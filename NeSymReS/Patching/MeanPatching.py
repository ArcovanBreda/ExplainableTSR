from tqdm.auto import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import get_params_fit, get_layer_names_new, prepare_encoders, evaluate_performance, get_topk_acc
from utils import from_names_to_encoder_info, apply_patches_to_model, get_encoders, get_layer_names

from functools import partial
import torch
from intervention_utils import intervension
import numpy as np
import torch.nn.functional as F
import glob


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


def initialize_model():
    intervention_class = intervension()
    model = intervention_class.nnModel
    return intervention_class, model

OPERATION = "exp"
N = 1000
SUBSET = "correct"

print("OPERATION:", OPERATION)

cfg, params_fit = get_params_fit(beam_size=2, max_len=15)
iclass = intervension()
model = iclass.nnModel
fitfunc = partial(model.fitfunc, cfg_params=params_fit)
names = get_layer_names_new()


mean_patches = np.load("data/Arco/MeanPatching/mean_patching_10000.npy", allow_pickle=True).item()
dataset = load_dataset(operation=OPERATION)

print(dataset[0])

track_idx = iclass.encode_ids([OPERATION, OPERATION])[0]
not_able_to_prefix = 0

counter = 0  # for debugging
intervention_class, model = initialize_model()
Xs, ys, equations, eq_untill_target = prepare_inputs(intervention_class, dataset, 200, max_samples=100)

encoders = prepare_encoders(names, [], mean_patches, "mean", model, Xs, ys)
ranks, counter, logit_score_base = evaluate_performance(model, intervention_class,
                                            encoders, equations,
                                            eq_untill_target,
                                            OPERATION,
                                            return_activation=True)
base_top1 = get_topk_acc(counter, 1)
base_top2 = get_topk_acc(counter, 2)
base_top3 = get_topk_acc(counter, 3)
print(f"Top-1: {base_top1:.3f}, Top-2: {base_top2:.3f}, Top-3: {base_top3:.3f}, logit score: {logit_score_base.mean():.3f}\n")
print("-=+=-=+=-=+=-=+=-=+=-=+=-=+=-=+=-=+=-=+=-=+=-=+=-=+=-=+=-=+=-=+=-=+=-=+=-")

all_data = []
for idx in tqdm(range(0, 117)):
    encoders = prepare_encoders(names, [idx], mean_patches, "mean", model, Xs, ys)
    ranks, counter, logit_score = evaluate_performance(model, intervention_class,
                                             encoders, equations,
                                             eq_untill_target,
                                             OPERATION,
                                             return_activation=True)
    top1 = get_topk_acc(counter, 1)
    top2 = get_topk_acc(counter, 2)
    top3 = get_topk_acc(counter, 3)
    all_data.append([base_top1 - top1, base_top2 - top2, base_top2 - top2, logit_score_base.mean() - logit_score.mean()])

all_data = np.array(all_data)
num = len(dataset)
save_name = f'data/Arco/MeanPatching/results_MEAN_{OPERATION}_{num}.npy'
np.save(save_name, all_data)

print(f"Saved to {save_name}")