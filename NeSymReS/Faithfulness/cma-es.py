import argparse
import numpy as np
import torch
import cma
from collections import Counter
from tqdm.auto import tqdm
import multiprocessing as mp
from functools import partial
from datetime import datetime

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import (
    same_recons, get_topk_acc, get_layer_names_new,
    Logger, evaluate_performance, prepare_encoders
)
from intervention_utils import intervension
import glob

THRESHOLD_TOP1 = 999
THRESHOLD_TOP2 = 999
THRESHOLD_TOP3 = 999

# Heavy penalty for non-feasible solutions
PENALTY = 1e2

def parse_args():
    parser = argparse.ArgumentParser(description="Script for minimal circuit search with CMA-ES.")
    parser.add_argument("--operation", type=str, default="sin", help="Operation type (e.g., 'sin').")
    parser.add_argument("--num_points", type=int, default=200, help="Number of points used for input preparation.")
    parser.add_argument("--max_iterations", type=int, default=117, 
                       help="Dimension of the search space (usually equals the number of heads).")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--max_samples", type=int, default=100, help="Number of samples to evaluate on.")
    parser.add_argument("--cma_max_evals", type=int, default=10000, help="Maximum number of CMA-ES evaluations.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers.")
    parser.add_argument("--patch_type", type=str, default="mean", help="Patch type either mean or resample.")
    parser.add_argument("--patch_type_subset", type=str, default="TRAIN", help="Patch type from the subset (default or same_decode)")
    parser.add_argument("--patch_type_CTR", type=str, default="cos", help="Patch type token to replace for loading dataset.")
    parser.add_argument("--patch_type_use_resample_CTR", action="store_true", help="Disable using resample mean; use default patch type instead.")
    parser.add_argument("--evaluation_type", type=str, default="functional", help="Evaluate on model or functional faitfulness")

    return parser.parse_args()


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


def initialize_model():
    intervention_class = intervension()
    model = intervention_class.nnModel
    return intervention_class, model


def prepare_inputs(intervention_class, dataset, num_points, max_samples=100):
    Xs, ys, eq_untill_target, equations = [], [], [], []
    for datapoint, eq in tqdm(dataset[:max_samples], desc="Preparing inputs"):
        X, y = intervention_class.get_input(
            str(datapoint["equation"]),
            n_variables=len(datapoint["variables"]),
            number_of_points=num_points
        )
        Xs.append(X)
        ys.append(y)
        equations.append(str(datapoint["equation"]))
        eq_untill_target.append(eq)
    return Xs, ys, equations, eq_untill_target


def evaluate_candidate(x, names, intervention_class, model, Xs, ys, equations, 
                       equations_untill_target_, mean_patches, target_operation,
                       patch_type, evaluation_type):
    x = np.array(x)
    excluded = list(np.where(x > 0.5)[0])

    encoders = prepare_encoders(names, excluded, mean_patches, patch_type, model, Xs, ys)

    ranks, counter, logit_score = evaluate_performance(model, intervention_class,
                                             encoders, equations,
                                             equations_untill_target_,
                                             target_operation,
                                             return_activation=True)

    top1 = get_topk_acc(counter, 1)
    top2 = get_topk_acc(counter, 2)
    top3 = get_topk_acc(counter, 3)

    n_kept = len(names) - len(excluded)

    penalty = 0
    if evaluation_type == "functional":
        if top1 < THRESHOLD_TOP1:
            penalty += (THRESHOLD_TOP1 - top1) * PENALTY
        if top2 < THRESHOLD_TOP2:
            penalty += (THRESHOLD_TOP2 - top2) * PENALTY
        if top3 < THRESHOLD_TOP3:
            penalty += (THRESHOLD_TOP3 - top3) * PENALTY
    elif evaluation_type == "model":
        if logit_score.mean() < THRESHOLD_ACTIVATION:
           penalty += (THRESHOLD_ACTIVATION - logit_score.mean()) * PENALTY
    else:
        raise NotImplementedError

    return (n_kept + penalty, top1, top2, top3, logit_score, n_kept, excluded)


def init_worker(data, threshold_top1, threshold_top2, threshold_top3, threshold_activation):
    global worker_names, worker_Xs, worker_ys, worker_equations, worker_mean_patches
    global worker_operation, worker_intervention_class, worker_model
    global worker_equations_untill_target, worker_patch_type, worker_evaluation_type
    global THRESHOLD_TOP1, THRESHOLD_TOP2, THRESHOLD_TOP3, THRESHOLD_ACTIVATION

    worker_names = data['names']
    worker_Xs = data['Xs']
    worker_ys = data['ys']
    worker_equations = data['equations']
    worker_mean_patches = data['mean_patches']
    worker_operation = data['operation']
    worker_equations_untill_target = data['eq_untill_target']
    worker_intervention_class, worker_model = initialize_model()
    worker_patch_type = data['patch_type']
    worker_evaluation_type = data['evaluation_type']

    THRESHOLD_TOP1 = threshold_top1
    THRESHOLD_TOP2 = threshold_top2
    THRESHOLD_TOP3 = threshold_top3
    THRESHOLD_ACTIVATION = threshold_activation


def parallel_evaluator(x_index):
    x, index = x_index
    result = evaluate_candidate(
        x, worker_names, worker_intervention_class, worker_model,
        worker_Xs, worker_ys, worker_equations, worker_equations_untill_target,
        worker_mean_patches, worker_operation, worker_patch_type, worker_evaluation_type,
    )
    return index, result


def setup_logging(args):
    os.makedirs("logs", exist_ok=True)

    random_seed_str = str(args.random_seed) if args.random_seed is not None else "none"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"logs/cma-es_{args.operation}_{args.patch_type}_{args.evaluation_type}_{args.cma_max_evals}_{random_seed_str}_{args.num_points}_{args.max_iterations}_{args.max_samples}_{args.num_workers}_{timestamp}.txt"
    sys.stdout = Logger(filename)

    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("\n")


def main(args):
    global THRESHOLD_TOP1, THRESHOLD_TOP2, THRESHOLD_TOP3
    
    setup_logging(args)
    np.random.seed(args.random_seed)

    # Initial setup
    print("Initial setup...")
    names = get_layer_names_new()
    n_heads = len(names)
    dataset = load_dataset(args.operation)
    intervention_class, model = initialize_model()
    Xs, ys, equations, eq_untill_target = prepare_inputs(intervention_class, dataset, args.num_points, max_samples=args.max_samples)

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

    # Baseline performance
    print("Evaluating baseline performance with full circuit...")
    encoders = prepare_encoders(names, [], mean_patches, args.patch_type, model, Xs, ys)
    ranks, counter, logit_score = evaluate_performance(model,
                                                         intervention_class,
                                                         encoders,
                                                         equations,
                                                         eq_untill_target,
                                                         args.operation,
                                                         return_activation=True)
    
    base_top1 = get_topk_acc(counter, 1)
    base_top2 = get_topk_acc(counter, 2)
    base_top3 = get_topk_acc(counter, 3)
    print(f"Baseline circuit ({n_heads} heads): Top-1: {base_top1:.3f}, Top-2: {base_top2:.3f}, Top-3: {base_top3:.3f}, logit score: {logit_score.mean():.3f}\n")

    THRESHOLD_TOP1 = base_top1 - 0.1
    THRESHOLD_TOP2 = base_top2 - 0.1
    THRESHOLD_TOP3 = base_top3 - 0.1
    THRESHOLD_ACTIVATION = logit_score.mean() - 0.1

    eval_counter = [0]
    scores = []
    best_solution = {
        'n_kept': [n_heads],
        'top1': [base_top1],
        'top2': [base_top2],
        'top3': [base_top3],
        'excluded': [[]],
        'x': [np.zeros(n_heads)]
    }

    # CMA-ES configuration
    intersection_circuit = list(range(117))
    x0 = np.ones(n_heads) * 0.6
    x0[intersection_circuit] = 0.4
    sigma0 = 0.1
    opts = {
        'maxfevals': args.cma_max_evals,
        'popsize': args.num_workers,
        'CMA_stds': [0.5] * n_heads,
        'bounds': [0, 1],
        'CMA_diagonal': 10,
        'verbose': -9,
        'seed': 42
    }

    print(f"Starting CMA-ES search with {n_heads} dimensions...")

    worker_data = {
        'names': names,
        'Xs': Xs,
        'ys': ys,
        'equations': equations,
        'mean_patches': mean_patches,
        'operation': args.operation,
        'eq_untill_target': eq_untill_target,
        'patch_type': args.patch_type,
        'evaluation_type': args.evaluation_type,
    }

    pool = mp.Pool(
    processes=args.num_workers,
    initializer=init_worker,
    initargs=(worker_data, THRESHOLD_TOP1, THRESHOLD_TOP2, THRESHOLD_TOP3, THRESHOLD_ACTIVATION)
)

    with tqdm(total=args.cma_max_evals, desc="CMA-ES Progress") as pbar:
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        try:
            while eval_counter[0] < args.cma_max_evals:
                solutions = es.ask()
                # Evaluate candidates in parallel
                indexed_solutions = [(x, i) for i, x in enumerate(solutions)]
                results = pool.map(parallel_evaluator, indexed_solutions)
                results.sort(key=lambda r: r[0])
                results = [r[1] for r in results]
                values = []
                for i, res in enumerate(results):
                    obj, top1, top2, top3, logit_score, n_kept, excluded = res
                    x = solutions[i]
                    eval_counter[0] += 1
                    
                    if eval_counter[0] % 1000 == 0:
                        save_name = f"data/Arco/CircuitFinding/cma-result_{args.operation}_{args.patch_type}_{args.evaluation_type}_{args.cma_max_evals}_{args.num_workers}_{args.max_samples}.npz"
                        covariance_matrix = es.C
                        np.savez(save_name,
                            mean=es.result.xbest,
                            covariance=covariance_matrix,
                            fbest=es.result.fbest,
                            scores=scores)
                        print(f"\n INTERMEDIATE SAVE: Saved results to {save_name}")

                    if (
                        (args.evaluation_type == "functional" and
                        top1 >= THRESHOLD_TOP1 and
                        top2 >= THRESHOLD_TOP2 and
                        top3 >= THRESHOLD_TOP3 and
                        n_kept < best_solution['n_kept'][0])
                        or
                        (args.evaluation_type == "model" and
                        logit_score.mean() >= THRESHOLD_ACTIVATION and
                        n_kept < best_solution['n_kept'][0])
                    ):

                        best_solution.update({
                            'n_kept': [n_kept],
                            'top1': [top1],
                            'top2': [top2],
                            'top3': [top3],
                            'logit_score': [logit_score.mean()],
                            'excluded': [excluded],
                            'x': [x.copy()]
                        })
                        print(f"\n New best @ eval {eval_counter[0]}: {n_kept} heads "
                                  f"(Top1={top1:.3f}, Top2={top2:.3f}, Top3={top3:.3f}), logit score: {logit_score.mean():.3f}")
                        print(f"Excluded heads: {sorted(excluded)}")
                        full_range = list(range(0, 117))
                        print("Circuit:", " ".join(map(str, sorted(set(full_range) - set(excluded)))), "\n")

                    pbar.update(1)
                    pbar.set_postfix({
                        'heads': n_kept,
                        'top1': f"{top1:.2f}",
                        'best': best_solution['n_kept'][0]
                    })
                    scores.append(obj)
                    values.append(obj)
                es.tell(solutions, values)
        finally:
            pbar.close()
            pool.close()
            pool.join()

    print("\nOptimization complete")
    print(f"Best valid circuit: {best_solution['n_kept'][0]} heads")
    print(f"Performance: Top1={best_solution['top1'][0]:.3f}, "
          f"Top2={best_solution['top2'][0]:.3f}, Top3={best_solution['top3'][0]:.3f}")
    print(f"Excluded heads: {sorted(best_solution['excluded'][0])}")
    save_name = f"data/Arco/CircuitFinding/cma-result_{args.operation}_{args.patch_type}_{args.evaluation_type}_{args.cma_max_evals}_{args.num_workers}_{args.max_samples}.npz"
    covariance_matrix = es.C
    np.savez(save_name,
         mean=es.result.xbest,
         covariance=covariance_matrix,
         fbest=es.result.fbest,
         scores=scores)
    print(f"\nSaved results to {save_name}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    main(args)