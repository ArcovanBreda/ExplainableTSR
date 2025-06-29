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
import sympy as sp
from utils import evaluate_formula_samples
from gen_dataset import calculate_accuracy
import pandas as pd
import re


def evaluate_candidate(candidate_prefix, candidate_infix, datapoint, eq_simpl, X, y):
    if candidate_infix == "<S>":
        return "cnr", candidate_infix, candidate_prefix

    if candidate_infix == datapoint["equation"]:
        return "no_help", candidate_infix, candidate_prefix

    try:
        perms0 = generate_permutations(candidate_infix, [0])
        if eq_simpl == perms0[0]:
            datapoint["c"] = perms0
            return "zero", candidate_infix, candidate_prefix
    except Exception:
        pass

    try:
        perms1 = generate_permutations(candidate_infix, [1])
        if eq_simpl == perms1[0]:
            datapoint["c"] = perms1
            return "one", candidate_infix, candidate_prefix
    except Exception:
        pass

    try:
        perms_both = generate_permutations(candidate_infix, [0, 1])
        if isinstance(perms_both, list):
            for perm in perms_both:
                if eq_simpl == perm:
                    datapoint["c"] = perm
                    return "zeroone", candidate_infix, candidate_prefix
            datapoint["c"] = perms_both
    except Exception:
        pass

    c_val = datapoint.get("c")
    if isinstance(c_val, str):
        try:
            y_hat = evaluate_formula_samples(c_val, X)
            acc = calculate_accuracy(y, y_hat, threshold=0.05)
            if acc >= 95:
                return "point", c_val, candidate_prefix
        except Exception:
            pass
    if isinstance(c_val, list):
        for perm in c_val:
            try:
                y_hat = evaluate_formula_samples(perm, X)
                acc = calculate_accuracy(y, y_hat, threshold=0.05)
                if acc >= 95:
                    datapoint["c"] = perm
                    return "point", perm, candidate_prefix
            except Exception:
                pass
    return None


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    data = np.load("data/Arco/Datasets/expressions_with_prefix_1000000_constatson_True.npy", allow_pickle=True)
    NUMBER_TO_EVAL = 1_000

    for beam_size in [2**i for i in range(3, 4)]:
        cfg, params_fit = get_params_fit(beam_size=beam_size, max_len=60)
        iclass = intervension()
        model = iclass.nnModel
        fitfunc = partial(model.fitfunc, cfg_params=params_fit, max_len=60)
        fitfuncskeleton = partial(model.fitfunc, cfg_params=params_fit, return_skeleton=True, max_len=60)

        results = defaultdict(list)
        failed_equations = []

        with tqdm(total=NUMBER_TO_EVAL, desc=f"Processing datapoints (Beam {beam_size})") as pbar:
            for datapoint in data:
                if sum(len(v) for v in results.values()) >= NUMBER_TO_EVAL:
                    break
                try:
                    infix, vars, prefix = datapoint.values()
                    X, y = iclass.get_input(infix, 200, n_variables=len(vars))
                except Exception:
                    continue

                eq_simpl = str(sp.simplify(infix))
                pred_prefixes, _ = fitfuncskeleton(X, y)
                candidate_found = False

                for beam_idx, (_, pred_prefix) in enumerate(pred_prefixes):
                    candidate_prefix = iclass.decode_ids(pred_prefix.cpu().numpy())
                    try:
                        candidate_infix = make_human_readable(prefix_to_infix(candidate_prefix))
                    except Exception:
                        continue

                    try:
                        candidate_infix = str(simplify_with_timeout(candidate_infix, 1))
                    except Exception:
                        pass

                    eval_result = evaluate_candidate(candidate_prefix, candidate_infix, datapoint, eq_simpl, X, y)
                    if eval_result:
                        category, final_eq, final_prefix = eval_result
                        results[category].append({
                            "datapoint": datapoint,
                            "recreation": {"equation": final_eq, "prefix": final_prefix, "beam_idx": beam_idx}
                        })
                        candidate_found = True
                        break

                if candidate_found:
                    pbar.update(1)
                    continue

                try:
                    pred = fitfunc(X, y)
                except Exception:
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
                    except Exception:
                        pass

                if bfgs_found:
                    pbar.update(1)
                    continue

                results["incorrect"].append(datapoint)
                pbar.update(1)

        np.savez(f"data/Arco/Datasets/results_1000_beam_{beam_size}_constants.npz", **results)
        print(f"Saved results for beam size {beam_size}")
