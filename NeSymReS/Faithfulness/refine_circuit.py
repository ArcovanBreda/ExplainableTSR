from utils import (
    get_attr_from_path, get_params_fit, same_recons, 
    get_topk_acc, from_names_to_encoder_info, get_layer_names_new, 
    apply_patches_to_model, get_encoders
)
from intervention_utils import intervension
import numpy as np
import torch
from collections import Counter
from tqdm.auto import tqdm

OPERATION = 'sin'
NUMBER_OF_POINTS = 200
MAX_ITERATIONS = 119
EXCLUDED_HEADS =  []


def load_dataset(operation):
    data = np.load(f"data/Arco/CircuitFinding/dataset_{operation}_1000.npz", allow_pickle=True)
    dataset = data["correct"]
    same_reconstructions = same_recons(dataset, operation)
    return [s for s in same_reconstructions if s[0]["reconstructed_as"] == "correctly_predicted"]


def initialize_model():
    intervention_class = intervension()
    model = intervention_class.nnModel
    return intervention_class, model


def prepare_inputs(intervention_class, dataset, num_points, max_samples=100):
    Xs, ys, equations = [], [], []
    for datapoint, eq in tqdm(dataset[:max_samples], desc="Preparing inputs"):
        X, y = intervention_class.get_input(
            str(datapoint["equation"]),
            n_variables=len(datapoint["variables"]),
            number_of_points=num_points
        )
        Xs.append(X)
        ys.append(y)
        equations.append(eq)
    return Xs, ys, equations


def evaluate_performance(model, intervention_class, encoders, equations, target_operation):
    performance = Counter()
    for encoder, eq in zip(encoders, equations):
        next_pred = model.decode_one_step(
            encoder=encoder,
            input_tokens=intervention_class.encode_ids(eq)
        ).cpu()
        sorted_pred = intervention_class.decode_ids(torch.argsort(next_pred, descending=True).numpy())
        found_idx = np.where(np.array(sorted_pred) == target_operation)[0][0] if target_operation in sorted_pred else -1
        performance[found_idx + 1] += 1
    return performance


def refine_circuit(initial_non_important, intervention_class, model, Xs, ys, equations, layer_names, mean_patches):
    """
    Refines the circuit by iteratively removing non-important heads and testing replacements.
    """
    non_important = set(initial_non_important)
    alternatives = []

    print(f"Initial non-important heads: {sorted(non_important)}")

    while True:
        improved = False
        for head in list(non_important):
            # Temporarily remove this head
            temp_non_important = non_important - {head}
            print(f"\nTesting removal of head {head}...")

            # Calculate the total number of valid candidates
            valid_candidates = [
                candidate for candidate in range(MAX_ITERATIONS)
                if candidate not in temp_non_important and candidate not in EXCLUDED_HEADS
            ]
            total_valid = len(valid_candidates)

            patch_info = from_names_to_encoder_info(layer_names, temp_non_important, mean_patches)
            patch_info = [(name, info["headidx"], info["new_output"]) for name, info in patch_info.items()]
            edit_model = apply_patches_to_model(model, patch_info)

            # Test adding new heads with a progress bar
            for candidate in tqdm(valid_candidates, desc="Testing candidates", total=total_valid):
                # Create a new set of non-important heads including the candidate
                test_non_important = temp_non_important | {candidate}

                # Apply patches and evaluate performance
                patch_info = from_names_to_encoder_info(layer_names, [candidate], mean_patches)
                patch_info = [(name, info["headidx"], info["new_output"]) for name, info in patch_info.items()]
                test_model = apply_patches_to_model(edit_model, patch_info)
                test_encoders = get_encoders(test_model, Xs, ys)
                test_perf = evaluate_performance(model, intervention_class, test_encoders, equations, OPERATION)

                # Check if performance criteria are met
                if all([
                    get_topk_acc(test_perf, 1) >= 0.55,
                    get_topk_acc(test_perf, 2) >= 0.68,
                    get_topk_acc(test_perf, 3) >= 0.9
                ]):
                    print(f"Found replacement: head {candidate} can be added if head {head} is removed")
                    non_important = test_non_important
                    improved = True
                    break  # Restart the search with the new non-important set

            if improved:
                break

        if not improved:
            break  # No further improvements found

    print(f"\nFinal non-important heads: {sorted(non_important)}")
    return non_important, alternatives


def main(iteration_list=None):
    # Initial setup
    print("Initial setup...")
    names = get_layer_names_new()
    same_reconstructions_top3 = load_dataset(OPERATION)
    intervention_class, model = initialize_model()
    Xs, ys, equations = prepare_inputs(intervention_class, same_reconstructions_top3, NUMBER_OF_POINTS)

    # Load precomputed results
    print("Loading datasets...")
    patch_results = np.abs(np.load("data/Arco/ResamplePatching/INVERTED_results_sin_100.npy"))
    mean_patches = np.load("data/Arco/MeanPatching/mean_patching_10000.npy", allow_pickle=True).item()
    layer_names = get_layer_names_new()

    # Baseline with excluded heads
    print("Preparing baseline performance...")
    patch_info = from_names_to_encoder_info(layer_names, EXCLUDED_HEADS, mean_patches)
    edited_model = apply_patches_to_model(model, [(name, info["headidx"], info["new_output"]) for name, info in patch_info.items()])

    # Performance evaluation
    encoders = get_encoders(edited_model, Xs, ys)
    baseline_perf = evaluate_performance(model, intervention_class, encoders, equations, OPERATION)

    for k in range(1, 11):
        print(f"Top-{k} accuracy of sin mean patching: {get_topk_acc(baseline_perf, k):.3f}")

    # Initial non-important heads
    refined_non_important, alternatives = refine_circuit(EXCLUDED_HEADS, intervention_class, edited_model, Xs, ys, equations, layer_names, mean_patches)

    print("\nRefinement complete!")
    print(f"Minimal non-important heads: {sorted(refined_non_important)}")
    print(f"Alternative circuits: {alternatives}")


if __name__ == "__main__":
    main()