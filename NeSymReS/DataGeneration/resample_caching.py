import argparse
import numpy as np
import torch
from collections import defaultdict
from functools import partial
import multiprocessing
import tqdm.auto as tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import (
    get_params_fit, get_layer_outputs, layer_to_head,
    same_recons, prefix_to_infix,
    all_unary_operators, all_binary_operators
)
from intervention_utils import intervension
import glob


def clone_tensor_dict(d):
    """Recursively detach, clone and move tensors to CPU."""
    result = {}
    for k, sub in d.items():
        result[k] = {kk: vv.detach().clone().cpu() for kk, vv in sub.items()}
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Resample Patching Mean Activation")
    parser.add_argument("--operation", type=str, default="sin", help="Target operation to resample")
    parser.add_argument("--control", type=str, default="cos", help="Control operation for comparison")
    parser.add_argument("--subset", type=str, default="TRAIN", help="Subset of data to use")
    parser.add_argument("--num_points", type=int, default=200, help="Number of points to sample for input")
    parser.add_argument("--max_datapoints", type=int, default=100, help="number of datapoints")
    parser.add_argument("--data_root", type=str, default="data/Arco/CircuitFinding", help="Root directory of the dataset")
    parser.add_argument("--save_root", type=str, default="data/Arco/ResamplePatching", help="Directory to save results")
    parser.add_argument("--n", type=int, default=1000, help="Number of formulas in dataset file")
    return parser.parse_args()


def load_dataset(operation, base_path=None, subset="TRAIN"):
    if base_path is None:
        path = os.path.join("data", "Arco", "CircuitFinding")
    else:
        path = os.path.join(base_path, "data", "Arco", "CircuitFinding")

    abs_path = os.path.abspath(path)
    file_path = glob.glob(f"{abs_path}/CD_{subset}_{operation}_*.npy")[0]
    data = np.load(file_path, allow_pickle=True)
    print("length dataset:", len(data))
    return data


def main():
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()

    iclass = intervension()
    model = iclass.nnModel
    model.to("cpu")

    dataset = load_dataset(args.operation, None, args.subset)
    unary_ops = all_unary_operators()
    binary_ops = all_binary_operators()
    resample_ops = unary_ops if args.operation in unary_ops else binary_ops
    resample_ops.remove(args.operation)

    all_data = []
    CTR_results = []

    for counter, (datapoint, _) in enumerate(tqdm.tqdm(dataset, total=len(dataset))):
        if counter >= args.max_datapoints:
            break

        prefix = datapoint["prefix"]
        n_vars = len(datapoint["variables"])
        op_index = prefix.index(args.operation)

        running_sums = defaultdict(dict)
        running_counts = defaultdict(dict)
        CTR_result = None

        for resample_op in resample_ops:
            modified_prefix = prefix.copy()
            modified_prefix[op_index] = resample_op

            X, y = iclass.get_input(prefix_to_infix(modified_prefix), n_variables=n_vars, number_of_points=args.num_points)
            layer_outputs = get_layer_outputs(X, y, model)
            activations = layer_to_head(layer_outputs)

            if resample_op == args.control:
                CTR_result = clone_tensor_dict(activations)

            for layer, neurons in activations.items():
                for neuron, tensor in neurons.items():
                    if not torch.isnan(tensor).any():
                        if neuron not in running_sums[layer]:
                            running_sums[layer][neuron] = tensor.detach().clone().cpu()
                            running_counts[layer][neuron] = 1
                        else:
                            running_sums[layer][neuron] += tensor.detach().clone().cpu()
                            running_counts[layer][neuron] += 1

        # Average activations
        avg_activations = defaultdict(dict)
        for layer, neurons in running_sums.items():
            for neuron, total in neurons.items():
                count = running_counts[layer][neuron]
                avg_activations[layer][neuron] = (total / count).detach().clone().cpu()

        all_data.append(avg_activations)
        CTR_results.append(CTR_result)

    # Save results
    save_path = f"{args.save_root}/cached_values_{len(all_data)}_{args.operation}_{args.control}_{args.subset}.npz"
    np.savez(save_path, resample_patch_mean=all_data, resample_patch_CTR=CTR_results)
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
