import os
import sys
# os.chdir("/home/arco/Downloads/Master/MscThesis/ExplainableDSR")
sys.path[-1] = "/home/arco/Downloads/Master/MscThesis/ExplainableDSR/"

from intervention_utils import intervension
from utils import get_params_fit, prefix_to_infix, infix_to_prefix, make_human_readable, simplify_with_timeout, generate_permutations, same_recons, check_same_decode
from functools import partial
import torch
import numpy as np
import multiprocessing
multiprocessing.set_start_method("spawn")
from tqdm.auto import tqdm
from collections import defaultdict
import signal
import sympy as sp
from utils import evaluate_formula_samples
from gen_dataset import calculate_accuracy
import matplotlib.pyplot as plt
from collections import Counter
from gen_dataset import make_corrupted
import argparse

# Argument Parser
parser = argparse.ArgumentParser(description="Run expression transformations with specified parameters.")
parser.add_argument("--n_corr_equations", type=int, default=1000, help="Number of corrupted equations to generate.")
parser.add_argument("--number_of_points", type=int, default=200, help="Number of points used for evaluation.")
parser.add_argument("--character_to_include", type=str, required=True, help="Character to be included in expressions.")
parser.add_argument("--character_to_change_to", type=str, required=True, help="Character to change 'character_to_include' into.")
parser.add_argument("--characters_to_exclude", type=str, nargs="+", required=True, help="Characters to exclude from expressions.")
parser.add_argument("--top", type=int, default=3, help="Number of points used for evaluation.")


args = parser.parse_args()

# Load dataset
cfg, params_fit = get_params_fit(beam_size=8, max_len=60)
iclass = intervension()
model = iclass.nnModel
fitfunc = partial(model.fitfunc, cfg_params=params_fit, max_len=60)
fitfuncskeleton = partial(model.fitfunc, cfg_params=params_fit, max_len=60, return_skeleton=True)

# dataset = np.load("data/Arco/Datasets/expressions_with_prefix_1000000_constatson_False.npy", allow_pickle=True)
dataset = np.load("data/Arco/CircuitFinding/OLD/dataset_log_1000.npz", allow_pickle=True)["correct"]

# PARAMETERS (from command-line arguments)
N_CORR_EQUATIONS = args.n_corr_equations
NUMBER_OF_POINTS = args.number_of_points
character_to_include = args.character_to_include
character_to_change_to = args.character_to_change_to
characters_to_exclude = args.characters_to_exclude

corr = []
incorr = []
incorr_corr = []
failed_equations = []
skipped = 0

print(f"Running with:\n"
      f"- Character to include: {character_to_include}\n"
      f"- Character to change to: {character_to_change_to}\n"
      f"- Characters to exclude: {characters_to_exclude}\n"
      f"- Number of corrupted equations: {N_CORR_EQUATIONS}\n"
      f"- Number of points: {NUMBER_OF_POINTS}")


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")

def run_with_timeout(func, args=(), kwargs=None, timeout_seconds=5):
    if kwargs is None:
        kwargs = {}
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = func(*args, **kwargs)
    except TimeoutException:
        result = "timeout"
    finally:
        signal.alarm(0)
    return result

def process_datapoint(datapoint):
    equation = datapoint["equation"]; variables = datapoint["variables"]; prefix= datapoint["prefix"]
    if not character_to_include in prefix or any(char in prefix for char in characters_to_exclude):
        return "skip"
    X, y = iclass.get_input(equation, n_variables=len(variables), number_of_points=NUMBER_OF_POINTS)
    eq_simpl = str(simplify_with_timeout(equation))
    pred_prefixes, _ = fitfuncskeleton(X, y)
    correct, failed = process_beam_candidates(pred_prefixes, iclass, eq_simpl, X, y, datapoint, equation)
    failed_equations.extend(failed)
    
    if not correct:
        values = [t.item() for t in pred_prefixes[0][1].cpu()]
        decoded = iclass.decode_ids(values)

        incorr.append(datapoint)
        return "incorrect"

    corr_eq = simplify_with_timeout(make_human_readable(prefix_to_infix(make_corrupted(prefix, character_to_include, character_to_change_to))))
    X_corr, y_corr = iclass.get_input(str(corr_eq), n_variables=len(variables), number_of_points=NUMBER_OF_POINTS)
    pred_prefixes, _ = fitfuncskeleton(X_corr, y_corr)
    correct_corr, failed = process_beam_candidates(pred_prefixes, iclass, eq_simpl, X_corr, y_corr, datapoint, equation)
    failed_equations.extend(failed)
    if not correct_corr:
        incorr_corr.append(datapoint)
        return "incorrect_corr"
    datapoint["corrupt"] = corr_eq
    datapoint["pred"] = correct[0]["recreation"]["prefix"]
    datapoint["predCorr"] = correct_corr[0]["recreation"]["prefix"]
    datapoint["predInfix"] = correct[0]["recreation"]["equation"]
    datapoint["predCorrInfix"] = correct_corr[0]["recreation"]["equation"]

    formula = datapoint["pred"]
    formula_corrupted = datapoint["predCorr"]
    char = character_to_include if character_to_include != "log" else "ln"
    same, eq = check_same_decode(formula, formula_corrupted, sign=char)

    if same:
        next_pred_gt = model.decode_one_step(encoder=iclass.Model(X, y), input_tokens=iclass.encode_ids(eq)).cpu()
        top = torch.argsort(next_pred_gt, descending=True)[0:args.top].numpy()
        decoded = iclass.decode_ids(top)

        if character_to_include in decoded or (character_to_include == "log" and "ln" in decoded):
            datapoint["reconstructed_as"] = "correctly_predicted"
            corr.append(datapoint)
            return "correct"
        else:
            datapoint["reconstructed_as"] = "same_decode"
            corr.append(datapoint)
            return "same"
    else:
        datapoint["reconstructed_as"] = "not_same_decode"
        corr.append(datapoint)
        return "not_same"


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
            return "zero", candidate_infix, candidate_prefix
    except Exception as e:
        failed_equations.append((datapoint["equation"], candidate_infix, "zero", str(e)))

    # 3. Check permutation with c = 1
    try:
        perms1 = generate_permutations(candidate_infix, [1])
        if eq_simpl == perms1[0]:
            return "one", candidate_infix, candidate_prefix
    except Exception as e:
        failed_equations.append((datapoint["equation"], candidate_infix, "one", str(e)))

    # 4. Check permutation with c = [0,1]
    try:
        perms_both = generate_permutations(candidate_infix, [0, 1])
    except Exception as e:
        failed_equations.append((datapoint["equation"], candidate_infix, "zeroone", str(e)))
        return None

    if isinstance(perms_both, list):
        for perm in perms_both:
            if eq_simpl == perm:
                return "zeroone", candidate_infix, candidate_prefix
        # If no single permutation matches, still keep the list

    # 5. Check pointwise evaluation:
    if isinstance(perms_both, str):
        try:
            y_hat = evaluate_formula_samples(perms_both, X)
            acc = calculate_accuracy(y, y_hat, threshold=0.05)
            if acc >= 95:
                return "point", perms_both, candidate_prefix
        except Exception as e:
            failed_equations.append((datapoint["equation"], perms_both, "point_str", str(e)))
    if isinstance(perms_both, list):
        for perm in perms_both:
            try:
                y_hat = evaluate_formula_samples(perm, X)
            except Exception as e:
                failed_equations.append((datapoint["equation"], perm, "point_list", str(e)))
                continue
            acc = calculate_accuracy(y, y_hat, threshold=0.05)
            if acc >= 95:
                return "point", perm, candidate_prefix

    # If none of the tests passed, return None.
    return None


def process_beam_candidates(pred_prefixes, iclass, eq_simpl, X, y, datapoint, equation):
    corr = []
    failed_equations = []

    for beam_idx, (_, pred_prefix) in enumerate(pred_prefixes):
        candidate_prefix = iclass.decode_ids(pred_prefix.cpu().numpy())

        try:
            candidate_infix = make_human_readable(prefix_to_infix(candidate_prefix))
        except Exception as e:
            failed_equations.append((equation, candidate_prefix, "decode", str(e)))
            continue

        try:
            candidate_infix = str(simplify_with_timeout(candidate_infix, 1))
        except Exception:
            pass

        eval_result = evaluate_candidate(candidate_prefix, candidate_infix, datapoint, eq_simpl, X, y)
        if eval_result:
            category, final_eq, final_prefix = eval_result
            corr.append({
                "datapoint": datapoint,
                "recreation": {"equation": final_eq, "prefix": final_prefix, "beam_idx": beam_idx, "category": category}
            })
            return corr, failed_equations
    return corr, failed_equations

save_threshold = 25
characters_to_exclude_str = "-".join(characters_to_exclude)
prev_save = 0

with tqdm(total=N_CORR_EQUATIONS, desc=f"Processing datapoints") as pbar:
    counter = 0
    same_reconstructed = 0
    correctly_predicted = 0
    start_from = 689
    for datapointidx, datapoint in enumerate(dataset[start_from:], start=start_from):
        if counter >= N_CORR_EQUATIONS:
            break

        pbar.set_description(
            f"Progress: idx: {datapointidx}, Same Reconstruction: {same_reconstructed}, "
            f"Incorrect: {len(incorr)}, "
            f"Failed: {int(len(failed_equations) /10 )}, "
            f"Skipped: {skipped}"
        )
        try:
            result = run_with_timeout(process_datapoint, args=(datapoint,), timeout_seconds=10)
        except Exception:
            print("failed")
            continue

        if result == "timeout":
            print("timeout")
            skipped += 1
        elif result == "skip":
            skipped += 1
        elif result == "same":
            same_reconstructed += 1
        elif result == "correct":
            counter += 1
            correctly_predicted += 1
            pbar.update(1)

        # Auto-save every 25 new 'correctly_predicted' datapoints
        if correctly_predicted % 25 == 0 and prev_save != correctly_predicted:
            save_path = f"data/Arco/CircuitFinding/dataset_{character_to_include}_{correctly_predicted}_{character_to_change_to}_{characters_to_exclude_str}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, correct=corr, incorrect=incorr, incorrect_corrupted=incorr_corr, datapointidx=datapointidx)
            print(f"Auto-saved {correctly_predicted} correctly predicted datapoints to {save_path}")
            prev_save = correctly_predicted

datapath = f"data/Arco/CircuitFinding/dataset_{character_to_include}_{correctly_predicted}_{character_to_change_to}_{characters_to_exclude_str}"
np.savez(datapath, correct=corr, incorrect=incorr, incorrect_corrupted=incorr_corr)
print(f"\n\n\nDONE\nSaved to {datapath}")
