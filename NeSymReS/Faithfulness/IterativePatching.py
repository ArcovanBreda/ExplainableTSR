import argparse
import numpy as np
import torch
import os
from collections import Counter
from tqdm.auto import tqdm
import sys
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import (
    same_recons, get_topk_acc, get_layer_names_new, get_layer_names,
    evaluate_performance, Logger, prepare_encoders,
)
from intervention_utils import intervension
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Script for iterative model intervention and evaluation.")

    parser.add_argument("--operation", type=str, default="sin", help="Operation type (e.g., 'sin').")
    parser.add_argument("--num_points", type=int, default=200, help="Number of points used for input preparation.")
    parser.add_argument("--max_iterations", type=int, default=117, help="Maximum number of iterations for exclusion.")
    parser.add_argument("--excluded_heads", type=int, nargs="*", default=[], 
                        help="List of excluded heads (space-separated). Example: --excluded_heads 1 3 5")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--max_samples", type=int, default=100, help="how many samples to eval on.")
    parser.add_argument("--patch_type", type=str, default="mean", help="Patch type either mean or resample.")
    parser.add_argument("--patch_type_CTR", type=str, default="cos", help="Patch type token to replace for loading dataset.")
    parser.add_argument("--patch_type_use_resample_CTR", action="store_true", help="Disable using resample mean; use default patch type instead.")
    parser.add_argument("--evaluation_type", type=str, default="functional", help="Evaluate on model or functional faitfulness")
    parser.add_argument("--patch_type_subset", type=str, default="TRAIN", help="Patch type from the subset (default or same_decode)")

    return parser.parse_args()


def setup_logging(args):
    os.makedirs("logs", exist_ok=True)

    excluded_heads_str = str(len(args.excluded_heads))
    random_seed_str = str(args.random_seed) if args.random_seed is not None else "none"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"logs/IterativePatching_{args.operation}_{args.patch_type}_{args.evaluation_type}_{random_seed_str}_{args.num_points}_{args.max_iterations}_{excluded_heads_str}_{args.max_samples}_{timestamp}.txt"

    sys.stdout = Logger(filename)

    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("\n")


def load_dataset(operation, base_path=None):
    if base_path is None:
        path = os.path.join("data", "Arco", "CircuitFinding")
    else:
        path = os.path.join(base_path, "data", "Arco", "CircuitFinding")
    abs_path = os.path.abspath(path)

    file_path = glob.glob(f"{abs_path}/CD_TRAIN_{operation}_*.npy")[0]
    data = np.load(file_path, allow_pickle=True)
    print("length dataset:", len(data))
    return data


def initialize_model(base_path=None):
    if base_path is None:
        path = ""
    else:
        path = os.path.join(base_path)
    abs_path = os.path.abspath(path)
    intervention_class = intervension(abs_path + "/jupyter/100M/eq_setting.json",
                                      abs_path + "/jupyter/100M/config.yaml",
                                      abs_path + "/weights/100M.ckpt")
    model = intervention_class.nnModel
    return intervention_class, model


def prepare_inputs(intervention_class, dataset, num_points, max_samples=100):
    Xs, ys, eq_untill_target, equations = [], [], [], []
    for datapoint, eq in dataset[:max_samples]:
        X, y = intervention_class.get_input(
            str(datapoint["equation"]),
            n_variables=len(datapoint["variables"]),
            number_of_points=num_points
        )
        Xs.append(X)
        ys.append(y)
        equations.append(str(datapoint["equation"]))
        eq_untill_target.append(eq)

    Xs = np.array(Xs)
    ys = np.array(ys)

    return (
        torch.tensor(Xs, dtype=torch.float32),
        torch.tensor(ys, dtype=torch.float32),
        np.array(equations, dtype=object),
        np.array(eq_untill_target, dtype=object)
    )


def main(args):
    setup_logging(args)

    if args.random_seed is None:
        print("Seed was not set correctly, exiting")
        exit()        

    np.random.seed(args.random_seed)

    print("Initial setup...")
    layer_names = get_layer_names_new()
    same_reconstructions_top3 = load_dataset(args.operation)
    intervention_class, model = initialize_model()
    Xs, ys, equations, eqs_untill_target = prepare_inputs(intervention_class, 
                                                          same_reconstructions_top3, 
                                                          args.num_points, 
                                                          max_samples=args.max_samples)
    patch_type = args.patch_type

    print("Loading datasets...")
    # Load patch data
    if args.patch_type == "mean":
        mean_patches = np.load("data/Arco/MeanPatching/mean_patching_10000.npy", allow_pickle=True).item()
    elif args.patch_type == "resample":
        data_name = f"data/Arco/ResamplePatching/cached_values_{args.max_samples}_{args.operation}_{args.patch_type_CTR}_{args.patch_type_subset}.npz"
        if args.patch_type_use_resample_CTR:
            print("Using CTR ONLY")
            mean_patches = np.load(data_name, allow_pickle=True)["resample_patch_CTR"]
        else:
            mean_patches = np.load(data_name, allow_pickle=True)["resample_patch_mean"]
    else:
        raise NotImplementedError

    # Baseline
    print("Preparing baseline performance...")
    encoders = prepare_encoders(layer_names, [], mean_patches, patch_type, model, Xs, ys)
    ranks, counter, logit_score = evaluate_performance(model,
                                                         intervention_class,
                                                         encoders,
                                                         equations,
                                                         eqs_untill_target,
                                                         args.operation,
                                                         return_activation=True)

    base_top1 = get_topk_acc(counter, 1)
    base_top2 = get_topk_acc(counter, 2)
    base_top3 = get_topk_acc(counter, 3)
    base_logit_score = logit_score
    print(f"Original model:   Top-1: {base_top1:.3f}, Top-2: {base_top2:.3f}, Top-3: {base_top3:.3f}, logit score: {logit_score.mean():.3f}")

    # performance of model with circuit only
    excluded = set(args.excluded_heads)
    encoders = prepare_encoders(layer_names, excluded, mean_patches, patch_type, model, Xs, ys)
    ranks, counter, logit_score = evaluate_performance(model,
                                                intervention_class,
                                                encoders,
                                                equations,
                                                eqs_untill_target,
                                                args.operation,
                                                return_activation=True)

    top1 = get_topk_acc(counter, 1)
    top2 = get_topk_acc(counter, 2)
    top3 = get_topk_acc(counter, 3)
    print(f"Baseline circuit: Top-1: {top1:.3f}, Top-2: {top2:.3f}, Top-3: {top3:.3f}, logit score: {logit_score.mean():.3f}")

    print("\nInitializing iterative search:\n")
    if args.random_seed == 999:
        iteration_list = np.arange(0, args.max_iterations, 1)[::-1]
    else:
        iteration_list = np.random.permutation(np.arange(1, args.max_iterations, 1))

    forward = True
    while True:
        new_exclusions = set()
        search_order = iteration_list if forward else reversed(iteration_list)

        for current_idx in search_order:
            if current_idx in excluded:
                continue
            test_encoders = prepare_encoders(layer_names, list(excluded) + [current_idx], mean_patches, patch_type, model, Xs, ys)
            ranks, counter, logit_score = evaluate_performance(model,
                                                         intervention_class,
                                                         test_encoders,
                                                         equations,
                                                         eqs_untill_target,
                                                         args.operation,
                                                         return_activation=True)

            if (
                (args.evaluation_type == "functional" and
                get_topk_acc(counter, 1) >= base_top1 - 0.1 and
                get_topk_acc(counter, 2) >= base_top2 - 0.1 and
                get_topk_acc(counter, 3) >= base_top3 - 0.1)
                or
                (args.evaluation_type == "model" and
                logit_score.mean() >= base_logit_score.mean() - 0.1)
            ):
                excluded.add(current_idx)
                new_exclusions.add(current_idx)
                print(f"\nExcluded head {layer_names[current_idx]} ({current_idx}): Current exclusion list {sorted(excluded)}")
                print(f"Length circuit: {args.max_iterations - len(excluded)}")
            else:
                print(current_idx, end=" ")
        iteration_list = [i for i in iteration_list if i not in new_exclusions]  # Remove newly excluded heads

        if not new_exclusions:
            complete = set(range(0, args.max_iterations))
            circuit = list(complete - set(excluded))
            print(f"\nCircuit length: {len(circuit)}, Not important length: {len(excluded)}")
            print(f"Circuit: {circuit}")
            print(f"Ran with seed: {args.random_seed}")
            break  # No new exclusions, stop the process

        forward = not forward  # Reverse search direction
        print("\n\nReversing order\n\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)