import argparse
import numpy as np
import torch
from collections import defaultdict
from functools import partial
import multiprocessing
from tqdm.auto import tqdm
import sympy as sp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import (
    get_params_fit, make_human_readable, simplify_with_timeout,
    same_recons, prefix_to_infix,
    get_attr_from_path, get_head_index, evaluate_formula_samples,
    get_layer_names_new
)
from NeSymReS.DataGeneration.gen_dataset import calculate_accuracy
from intervention_utils import intervension
from torch.utils.data import TensorDataset
import glob
import argparse


def load_dataset(operation, base_path=None, eval="TRAIN"):
    if base_path is None:
        path = os.path.join("data", "Arco", "CircuitFinding")
    else:
        path = os.path.join(base_path, "data", "Arco", "CircuitFinding")
    abs_path = os.path.abspath(path)

    file_path = glob.glob(f"{abs_path}/CD_{eval}_{operation}_*.npy")[0]
    data = np.load(file_path, allow_pickle=True)
    return data


def clone_tensor_dict(d):
    """Recursively detach, clone and move tensors to CPU."""
    result = {}
    for k, sub in d.items():
        result[k] = {kk: vv.detach().clone().cpu() for kk, vv in sub.items()}
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--operation", type=str, default="sin", help="Target operation to resample")
    parser.add_argument("--subset", type=str, default="correct", help="Subset of data to use")
    parser.add_argument("--num_points", type=int, default=200, help="Number of points to sample for input")
    parser.add_argument("--max_datapoints", type=int, default=100, help="Limit number of datapoints (debugging)")
    parser.add_argument("--data_root", type=str, default="data/Arco/CircuitFinding", help="Root directory of the dataset")
    parser.add_argument("--save_root", type=str, default="data/Arco/Datasets/Probing", help="Directory to save results")
    parser.add_argument("--n", type=int, default=1000, help="Number of formulas in dataset file")
    parser.add_argument("--idx", type=int, default=1, help="index to use for the operation")

    return parser.parse_args()

def evaluate_candidate(candidate_infix, datapoint, X, y):
    # based on NeSymRes DataGeneration Resample_Patching_dataset2.0.py
    """
    Evaluates one candidate equation (in infix form) against several criteria.
    Returns a tuple (category, final_equation, candidate_prefix) if the candidate matches,
    or None if it does not.
    Any exceptions are logged into the global failed_equations list.
    """

    # 1. If the candidate exactly matches the original infix
    if candidate_infix == datapoint["equation"]:
        return True

    # 2. Check pointwise evaluation:
    try:
        y_hat = evaluate_formula_samples(candidate_infix, X)
        acc = calculate_accuracy(y, y_hat, threshold=0.05)
        if acc >= 95:
            return True
        else:
            return False
    except Exception as e:
        return False


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

        eval_result = evaluate_candidate(candidate_infix, datapoint, X, y)
        if eval_result:
            return True

    return False


def get_dataset_activations(dataset, layer_path, model, iclass, f, t):
    data = []
    for counter, datapoint in enumerate(tqdm(dataset)):
        # 1. Get the datapoint
        infix, prefix = datapoint["equation"], datapoint["prefix"]
        n_vars = len(datapoint["variables"])
        X, y = iclass.get_input(target_eq=infix, n_variables=n_vars)

        # 2. Get the activations of the specified head using model.trace
        with model.trace(X, y):
            q = get_attr_from_path(model, f"{layer_path}.fc_q.output")
            q = q.save()
            k = get_attr_from_path(model, f"{layer_path}.fc_k.output").save()
            q = k.save()
            v = get_attr_from_path(model, f"{layer_path}.fc_v.output").save()
            v = v.save()
        output = torch.cat([q[0, :, f:t], k[0, :, f:t], v[0, :, f:t]], dim=-1)
        # 3. Average over the sequence dimension to get shape (64,)
        pooled_output = output.mean(dim=0).squeeze(0)
        # 4. Add to the list
        data.append(pooled_output)

    return data


def main():
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()

    magic_number = 500 if args.operation != "log" else 309

    full_name = get_layer_names_new()[args.idx]
    layer_path, head_idx = full_name.split("_")
    print(f"Creating dataset for: {layer_path} with head {head_idx}")

    cfg, params_fit = get_params_fit(beam_size=2, max_len=15)
    iclass = intervension()
    model = iclass.nnModel

    train = load_dataset(args.operation, eval="TRAIN")
    test = load_dataset(args.operation, eval="TEST")
    data = np.vstack((train, test))[:, 0]

    fitfuncskeleton = partial(model.fitfunc, cfg_params=params_fit, max_len=60, return_skeleton=True)
    f, t = get_head_index(torch.zeros((1,50,512)), int(head_idx), 8)

    print("Getting activations for correct dataset...")
    correct_dataset = get_dataset_activations(data[:magic_number], layer_path,
                                              model, iclass, f, t)

    print("Correct dataset created. Continuing with incorrect dataset...")

    try:
        save_path_equations = f"{args.save_root}/cached_datapoints_{magic_number}_{args.operation}_{args.subset}.npy"
        equations = np.load(save_path_equations, allow_pickle=True)
        print(f"Loaded {save_path_equations}.")

        print("Getting activations for incorrect dataset...")
        incorrect_dataset = get_dataset_activations(equations, layer_path,
                                              model, iclass, f, t)

    except Exception:
        print(f"Could not load {save_path_equations}, starting new equation finding objective.")
        all_generated_equations = np.load("data/Arco/Datasets/expressions_with_prefix_1000000_constatson_False.npy", allow_pickle=True)
        incorrect_equations = []
        incorrect_dataset = []

        with tqdm(total=magic_number) as pbar:
            for datapoint in all_generated_equations:
                if len(incorrect_dataset) >= magic_number:
                    break

                # 1. Get the datapoint
                infix, prefix = datapoint["equation"], datapoint["prefix"]
                n_vars = len(datapoint["variables"])

                # 2. Skip if prefix contains the operation we're filtering
                if args.operation in prefix:
                    continue

                # 3. Check if the model can recreate the datapoint
                X, y = iclass.get_input(infix, n_variables=n_vars)
                eq_simpl = str(sp.simplify(infix))
                pred_prefixes, _ = fitfuncskeleton(X, y)
                recreated = process_beam_candidates(pred_prefixes, iclass, eq_simpl, X, y, datapoint, infix)
                if not recreated:
                    continue

                # 4. Get the activations of the specified head
                with model.trace(X, y):
                    full_path = f"{layer_path}.output"
                    target = get_attr_from_path(model, full_path)
                    output = target.save()

                # 5. Average over the sequence dimension to get shape (64,)
                pooled_output = output.mean(dim=1).squeeze(0)[f:t]

                # 6. Add to the list
                incorrect_dataset.append(pooled_output)
                incorrect_equations.append(datapoint)
                pbar.update(1)

        save_path = f"{args.save_root}/cached_datapoints_{magic_number}_{args.operation}_{args.subset}"
        np.save(save_path, incorrect_equations)
        print(f"Saved cached equations to: {save_path}")

    all_activations = correct_dataset + incorrect_dataset
    all_labels = [1.0] * magic_number + [0.0] * magic_number

    X_tensor = torch.stack(all_activations)  # shape: (N, 64)
    y_tensor = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1)

    save_path = f"{args.save_root}/cached_values_{len(X_tensor)}_{args.operation}_{args.subset}_{layer_path}_{head_idx}.pt"
    torch.save((X_tensor, y_tensor), save_path)
    print(f"Saved cached values to: {save_path}")    


if __name__ == "__main__":
    main()
