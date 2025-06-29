from intervention_utils import intervension
from utils import (
    get_params_fit, prefix_to_infix,
    make_human_readable, simplify_with_timeout, generate_permutations
)
from functools import partial
import torch
import numpy as np
import multiprocessing
from tqdm.auto import tqdm
from collections import defaultdict
import signal
import sympy as sp
from utils import evaluate_formula_samples
from gen_dataset import calculate_accuracy
import pandas as pd
import re


def evaluate_candidate(candidate_prefix, candidate_infix, datapoint, eq_simpl, X, y):
    """
    Evaluates one candidate equation (in infix form) against several criteria.
    Returns a tuple (category, final_equation, candidate_prefix) if the candidate matches,
    or None if it does not.
    Any exceptions are logged into the global failed_equations list.
    """
    
    # 1. If the candidate exactly matches the original infix
    if candidate_infix == datapoint["equation"]:
        return "no_help", candidate_infix, candidate_prefix

    # 2. Check permutation with c = 0
    try:
        perms0 = generate_permutations(candidate_infix, [0])
        if eq_simpl == perms0[0]:
            datapoint["c"] = perms0
            return "zero", candidate_infix, candidate_prefix
    except Exception as e:
        failed_equations.append((datapoint["equation"], candidate_infix, "zero", str(e)))

    # 3. Check permutation with c = 1
    try:
        perms1 = generate_permutations(candidate_infix, [1])
        if eq_simpl == perms1[0]:
            datapoint["c"] = perms1
            return "one", candidate_infix, candidate_prefix
    except Exception as e:
        failed_equations.append((datapoint["equation"], candidate_infix, "one", str(e)))

    # 4. Check permutation with c = [0,1]
    try:
        perms_both = generate_permutations(candidate_infix, [0, 1])
        if isinstance(perms_both, list):
            for perm in perms_both:
                if eq_simpl == perm:
                    datapoint["c"] = perm
                    return "zeroone", candidate_infix, candidate_prefix
            # If no single permutation matches, still keep the list
            datapoint["c"] = perms_both
    except Exception as e:
        failed_equations.append((datapoint["equation"], candidate_infix, "zeroone", str(e)))

    # 5. Check pointwise evaluation:
    c_val = datapoint.get("c")
    if isinstance(c_val, str):
        try:
            y_hat = evaluate_formula_samples(c_val, X)
            acc = calculate_accuracy(y, y_hat, threshold=0.05)
            if acc >= 95:
                return "point", c_val, candidate_prefix
        except Exception as e:
            failed_equations.append((datapoint["equation"], c_val, "point_str", str(e)))
    if isinstance(c_val, list):
        for perm in c_val:
            try:
                y_hat = evaluate_formula_samples(perm, X)
            except Exception as e:
                failed_equations.append((datapoint["equation"], perm, "point_list", str(e)))
                continue
            acc = calculate_accuracy(y, y_hat, threshold=0.05)
            if acc >= 95:
                datapoint["c"] = perm  # Save the single permutation that passed
                return "point", perm, candidate_prefix

    # If none of the tests passed, return None.
    return None


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    cfg, params_fit = get_params_fit(beam_size=32, max_len=60)
    iclass = intervension()
    model = iclass.nnModel
    fitfunc = partial(model.fitfunc, cfg_params=params_fit, max_len=60)
    fitfuncskeleton = partial(model.fitfunc, cfg_params=params_fit, return_skeleton=True, max_len=60)

    data = np.load("data/Arco/Datasets/expressions_with_prefix_1000000_constatson_False.npy", allow_pickle=True)

    results = defaultdict(list)
    failed_equations = []

    NUMBER_TO_EVAL = 1_000
    with tqdm(total=NUMBER_TO_EVAL, desc="Processing datapoints") as pbar:
        for datapoint in data:
            pbar.set_description(
                f"Progress: no_help {len(results['no_help'])}, "
                f"zero {len(results['zero'])}, "
                f"one {len(results['one'])}, "
                f"zeroone {len(results['zeroone'])}, "
                f"point {len(results['point'])}, "
                f"bfgs {len(results['bfgs'])}, "
                f"incorrect {len(results['incorrect'])}, "
                f"cnr {len(results['cnr'])}"
            )
            if sum(len(v) for v in results.values()) >= NUMBER_TO_EVAL:
                break

            try:
                infix, vars, prefix = datapoint.values()
                X, y = iclass.get_input(infix, 200, n_variables=len(vars))
            except Exception as e:
                failed_equations.append((infix, None, "input_generation", str(e)))
                continue

            eq_simpl = str(sp.simplify(infix))
            pred_prefixes, _ = fitfuncskeleton(X, y)
            candidate_found = False
            potential_cnr_candidate = None

            for beam_idx, (_, pred_prefix) in enumerate(pred_prefixes):
                candidate_prefix = iclass.decode_ids(pred_prefix.cpu().numpy())
                try:
                    candidate_infix = make_human_readable(prefix_to_infix(candidate_prefix))
                except Exception as e:
                    failed_equations.append((infix, candidate_prefix, "decode", str(e)))
                    continue

                try:
                    candidate_infix = str(simplify_with_timeout(candidate_infix, 1))
                except Exception:
                    pass

                if candidate_infix == "<S>":
                    if potential_cnr_candidate is None:
                        potential_cnr_candidate = (candidate_prefix, candidate_infix, beam_idx)
                    continue

                eval_result = evaluate_candidate(candidate_prefix, candidate_infix, datapoint, eq_simpl, X, y)
                if eval_result:
                    category, final_eq, final_prefix = eval_result
                    results[category].append({
                        "datapoint": datapoint,
                        "recreation": {"equation": final_eq, "prefix": final_prefix, "beam_idx": beam_idx}
                    })
                    candidate_found = True
                    break

            # If no non-<S> candidate was found, but we did see a "<S>" candidate,
            # use that and label it "cnr".
            if not candidate_found and potential_cnr_candidate is not None:
                candidate_prefix, candidate_infix, beam_idx = potential_cnr_candidate
                results["cnr"].append({
                    "datapoint": datapoint,
                    "recreation": {"equation": candidate_infix, "prefix": candidate_prefix, "beam_idx": beam_idx}
                })
                candidate_found = True

            if candidate_found:
                pbar.update(1)
                continue

            # BFGS fallback:
            try:
                pred = fitfunc(X, y)
            except Exception as e:
                failed_equations.append((infix, None, "bfgs_fit", str(e)))
                pbar.update(1)
                continue

            bfgs_found = False
            for permutation in pred[0].get("all_bfgs_preds", []):
                try:
                    y_hat = evaluate_formula_samples(permutation, X)
                    acc = calculate_accuracy(y, y_hat, threshold=0.05)
                    if acc >= 95:
                        results["bfgs"].append(datapoint)
                        bfgs_found = True
                        break
                except Exception as e:
                    failed_equations.append((infix, permutation, "bfgs_eval", str(e)))

            if bfgs_found:
                pbar.update(1)
                continue

            results["incorrect"].append(datapoint)
            pbar.update(1)

    print("\nFailed Equations:")
    for failed in failed_equations:
        print(failed)

    np.savez("data/Arco/Datasets/results_1000_full_eval.npz", **results)
