from intervention_utils import intervension
from utils import (
    get_params_fit, prefix_to_infix, infix_to_prefix, 
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


def replace_variables(equation: str, translation_dict: dict) -> str:
    for old, new in sorted(translation_dict.items(), key=lambda x: -len(x[0])):
        equation = re.sub(rf'\b{old}\b', new, equation)
    return equation


def evaluate_candidate(candidate_prefix, candidate_infix, datapoint, eq_simpl, X, y):
    """
    Evaluates one candidate equation (in infix form) against several criteria.
    Returns a tuple (category, final_equation, candidate_prefix) if the candidate matches,
    or None if it does not.
    Any exceptions are logged into the global failed_equations list.
    """
    # 1. If the candidate is the special token "<S>"
    if candidate_infix == "<S>":
        return "cnr", candidate_infix, candidate_prefix

    # 2. If the candidate exactly matches the original infix
    if candidate_infix == datapoint["equation"]:
        return "no_help", candidate_infix, candidate_prefix

    # 3. Check permutation with c = 0
    try:
        perms0 = generate_permutations(candidate_infix, [0])
        if eq_simpl == perms0[0]:
            datapoint["c"] = perms0
            return "zero", candidate_infix, candidate_prefix
    except Exception as e:
        failed_equations.append((datapoint["equation"], candidate_infix, "zero", str(e)))

    # 4. Check permutation with c = 1
    try:
        perms1 = generate_permutations(candidate_infix, [1])
        if eq_simpl == perms1[0]:
            datapoint["c"] = perms1
            return "one", candidate_infix, candidate_prefix
    except Exception as e:
        failed_equations.append((datapoint["equation"], candidate_infix, "one", str(e)))

    # 5. Check permutation with c = [0,1]
    try:
        perms_both = generate_permutations(candidate_infix, [0, 1])
        if isinstance(perms_both, list):
            for perm in perms_both:
                if eq_simpl == perm:
                    datapoint["c"] = perm
                    return "zeroone", candidate_infix, candidate_prefix
            datapoint["c"] = perms_both
    except Exception as e:
        failed_equations.append((datapoint["equation"], candidate_infix, "zeroone", str(e)))

    # 6. Check pointwise evaluation:
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

    # Load and preprocess the dataset.
    file_path = "data/Arco/Datasets/FeynmanEquations.csv"
    df = pd.read_csv(file_path)
    df = df.drop(columns=['Filename', 'Number', 'Output'])
    df = df.dropna(subset=["# variables"])
    df["# variables"] = df["# variables"].astype(int)
    data_dict = df.to_dict(orient="records")

    # Manually adjust some datapoints.
    data_dict[21]["# variables"] = 3
    data_dict[22]["# variables"] = 4
    data_dict[38]["# variables"] = 4
    data_dict[90]["# variables"] = 4

    counter = 0
    translation_names = ["x_1", "x_2", "x_3"]
    var_dict_names = ["v1", "v2", "v3"]
    data = []
    for datapoint in data_dict:
        if counter >= 1000:
            break
        counter += 1

        formula, n_vars = datapoint["Formula"], datapoint["# variables"]
        if n_vars > 3:
            continue

        var_names, supports = [], []
        for i in range(n_vars):
            var_name = datapoint[f"{var_dict_names[i]}_name"]
            lower, upper = datapoint[f"{var_dict_names[i]}_low"], datapoint[f"{var_dict_names[i]}_high"]
            var_names.append((var_name, translation_names[i]))
            supports.append((lower, upper))

        corrected_formula = replace_variables(formula, dict(var_names))
        prefix = infix_to_prefix(corrected_formula)

        try:
            X, y = iclass.get_input(
                target_eq=corrected_formula, 
                number_of_points=20, 
                n_variables=n_vars,
                supports=supports
            )
        except Exception as e:
            print(f"Failed on datapoint {counter}:")
            print(formula, n_vars)
            print(corrected_formula)
            print(var_names, supports)
            print("Error:", e)
            continue

        data.append({
            "equation": corrected_formula,
            "variables": set([name[1] for name in var_names]),
            "prefix": prefix,
            "X": X,
            "y": y
        })

    results = {
        "no_help": [],
        "zero": [],
        "one": [],
        "zeroone": [],
        "point": [],
        "bfgs": [],
        "incorrect": [],
        "cnr": []
    }
    # Global list to track any failed evaluations.
    failed_equations = []

    # Main processing loop.
    with tqdm(total=len(data), desc="Processing datapoints") as pbar:
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
            infix = datapoint["equation"]
            prefix = datapoint["prefix"]
            X = datapoint["X"]
            y = datapoint["y"]
            eq_simpl = str(sp.simplify(infix))
            pred_prefixes, _ = fitfuncskeleton(X, y)
            candidate_found = False

            for beam_idx, (_, pred_prefix) in enumerate(pred_prefixes):
                candidate_prefix = iclass.decode_ids(pred_prefix.cpu().numpy())
                try:
                    candidate_infix = make_human_readable(prefix_to_infix(candidate_prefix))
                except Exception as e:
                    failed_equations.append((infix, candidate_prefix, "decode", str(e)))
                    continue
                try:
                    candidate_infix = str(simplify_with_timeout(candidate_infix, 1))
                except Exception as e:
                    pass

                eval_result = evaluate_candidate(candidate_prefix, candidate_infix, datapoint, eq_simpl, X, y)
                if eval_result is not None:
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

            # Try a BFGS evaluation if beam candidates did not succeed.
            try:
                pred = fitfunc(X, y)
            except Exception as e:
                failed_equations.append((infix, None, "bfgs_fit", str(e)))
                pbar.update(1)
                continue

            pred_funcs = pred[0].get("all_bfgs_preds", [])
            bfgs_found = False
            for permutation in pred_funcs:
                try:
                    y_hat = evaluate_formula_samples(permutation, X)
                except Exception as e:
                    failed_equations.append((infix, permutation, "bfgs_eval", str(e)))
                    continue
                acc = calculate_accuracy(y, y_hat, threshold=0.05)
                if acc >= 95:
                    datapoint["c"] = permutation
                    results["bfgs"].append(datapoint)
                    bfgs_found = True
                    break
            if bfgs_found:
                pbar.update(1)
                continue

            results["incorrect"].append(datapoint)
            pbar.update(1)

    # Save the results.
    np.savez(
        "data/Arco/Datasets/results_feynman_full_eval_20samples.npz",
        recreated_skeleton=results['no_help'],
        recreated_0=results['zero'],
        recreated_1=results['one'],
        recreated_01=results['zeroone'],
        recreated_point_eval=results['point'],
        recreated_BFGS=results['bfgs'],
        not_recreated=results['incorrect'],
        unable_to_recreate=results['cnr']
    )
