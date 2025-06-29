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


def clone_tensor_dict(d):
    """Recursively detach, clone and move tensors to CPU."""
    result = {}
    for k, sub in d.items():
        result[k] = {kk: vv.detach().clone().cpu() for kk, vv in sub.items()}
    return result


def transform_expression(datapoint, expr_type="monomial"):
    """
    Transforms the expression in `datapoint["model_prefix"]` based on the specified type.

    For "monomial": Replaces first and second 'mul' with 'add' (returns two variants).
    For "posynomial": Replaces all 'add' with 'mul' (returns one variant).
    """
    original = datapoint["model_prefix"]

    if expr_type == "monomial":
        try:
            first_mul_idx = original.index('mul')
            second_mul_idx = original.index('mul', first_mul_idx + 1)
        except ValueError:
            print("Less than two 'mul' operators found.")
            return None

        variant1 = original.copy()
        variant2 = original.copy()
        variant1[first_mul_idx] = "add"
        variant2[second_mul_idx] = "add"
        return [(variant1, first_mul_idx), (variant2, second_mul_idx)]

    elif expr_type == "posynomial":
        transformed = ['mul' if token == 'add' else token for token in original]
        add_index = original.index("add")
        return [(transformed, add_index)]

    else:
        raise NotImplementedError(f"Unknown expression type: {expr_type}")


def parse_args():
    parser = argparse.ArgumentParser(description="Resample Patching Mean Activation")
    parser.add_argument("--operation", type=str, default="monomial", help="choose between monomial and posynomial")
    parser.add_argument("--subset", type=str, default="correct", help="Subset of data to use")
    parser.add_argument("--num_points", type=int, default=200, help="Number of points to sample for input")
    parser.add_argument("--max_datapoints", type=int, default=100, help="number of datapoints")
    parser.add_argument("--data_root", type=str, default="data/Arco/CircuitFinding", help="Root directory of the dataset")
    parser.add_argument("--save_root", type=str, default="data/Arco/ResamplePatching", help="Directory to save results")
    parser.add_argument("--n", type=int, default=1000, help="Number of formulas in dataset file")
    return parser.parse_args()


def main():
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()

    cfg, params_fit = get_params_fit(beam_size=2, max_len=15)
    iclass = intervension()
    model = iclass.nnModel
    model.to("cpu")

    filename = f"{args.data_root}/dataset_{args.operation}_{args.n}.npz"
    print(f"\n\nLoading: {filename}")
    data = np.load(filename, allow_pickle=True)
    dataset = data[args.subset]
    print(f"Total datapoints in subset '{args.subset}': {len(dataset)}")


    reconstructions = [d for d in dataset if d["recreation_posynomial"] == 'both']
    print(f"Filtered same reconstructions: {len(reconstructions)}")

    all_data = []
    CTR_results = []
    
    # Determine resample operators
    unary_ops = all_unary_operators()
    binary_ops = all_binary_operators()
    resample_ops = unary_ops if args.operation in unary_ops else binary_ops
    # resample_ops.remove("div") # weg voor mono
    resample_ops.remove("sub") # weg voor poly

    for counter, datapoint in enumerate(tqdm.tqdm(reconstructions, total=args.max_datapoints)):
        if counter >= args.max_datapoints:
            break

        prefix = datapoint["model_prefix"]
        n_vars = len(datapoint["variables"])

        exprs = transform_expression(datapoint, expr_type=args.operation)

        for (exp, idx) in exprs:
            running_sums = defaultdict(dict)
            running_counts = defaultdict(dict)
            CTR_result = None
            resample_ops_clone = resample_ops.copy()
            resample_ops_clone.remove("mul" if exp[idx] == 'add' else 'mul')
            for resample_op in resample_ops_clone:
                modified_prefix = prefix.copy()
                modified_prefix[idx] = resample_op

                X, y = iclass.get_input(prefix_to_infix(modified_prefix[1:]), n_variables=n_vars, number_of_points=args.num_points)
                layer_outputs = get_layer_outputs(X, y, model)
                activations = layer_to_head(layer_outputs)

                if resample_op == ("add" if exp[idx] == 'add' else 'mul'):
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

            avg_activations = defaultdict(dict)
            for layer, neurons in running_sums.items():
                for neuron, total in neurons.items():
                    count = running_counts[layer][neuron]
                    avg_activations[layer][neuron] = (total / count).detach().clone().cpu()

            all_data.append(avg_activations)
            CTR_results.append(CTR_result)

    # Save results
    save_path = f"{args.save_root}/cached_values_{len(all_data)}_{args.operation}_{args.subset}.npz"
    np.savez(save_path, resample_patch_mean=all_data, resample_patch_CTR=CTR_results)
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
