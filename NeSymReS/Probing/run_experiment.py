import concurrent.futures
import multiprocessing
import sys
import os
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import get_layer_names_new
from NeSymReS.Probing.probe_head import main
multiprocessing.set_start_method('spawn', force=True)


def run_experiment(idx, data_path, save_name):
    try:
        print(f"Running for idx: {idx}")
        full_name = get_layer_names_new()[idx]
        data_path_idx = data_path + full_name + ".pt"
        print(f"Data path for experiment {idx}: {data_path_idx}")

        test_accuracies = main(data_path_idx, seeds=10, epochs=200,
                               patience=200, lr=1e-4, batch_size=32,
                               hidden_dim=10, save_path=f"pictures/probing/{save_name}_{idx}")

        print(f"Test accuracies for idx {idx}: {test_accuracies}")
        return (idx, test_accuracies)
    except Exception as e:
        print(f"Error occurred for idx {idx}: {e}")
        return (idx, f"Error: {e}")


def main_function(selected_indices, data_path, save_name):
    all_results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(run_experiment, idx, data_path, save_name) for idx in selected_indices]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            all_results.append(result)

    sorted_results = sorted(all_results, key=lambda x: selected_indices.index(x[0]))

    print("\n\nSorted Test Accuracies from all runs:")
    for idx, accuracy in sorted_results:
        print(f"Idx: {idx}, Test Accuracy: {accuracy}")

    complete_save_name = "data/Arco/Probing/" + save_name
    np.save(complete_save_name, np.array(sorted_results, dtype=object))
    print(f"Results saved to {complete_save_name}.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run probing experiments on selected indices")
    parser.add_argument('--indices', type=str, required=True,
                        help="Comma-separated list of selected indices (e.g., 12,88,6)")
    parser.add_argument('--datapath', type=str, required=True,
                        help="Prefix path to cached dataset files")
    parser.add_argument('--save', type=str, required=True,
                        help="Filename (without extension) to save the results")

    args = parser.parse_args()
    indices = list(map(int, args.indices.split(',')))
    main_function(indices, args.datapath, args.save)
