import tqdm.auto as tqdm
from utils import get_params_fit, get_layer_outputs, layer_to_head
from functools import partial
import torch
from intervention_utils import intervension
import numpy as np
from collections import defaultdict

cfg, params_fit = get_params_fit(beam_size=2, max_len=15)
iclass = intervension()
model = iclass.nnModel
fitfunc = partial(model.fitfunc, cfg_params=params_fit)
N_EQUATIONS = 10_000

# Load data
dataset = np.load("data/Arco/Datasets/expressions_1000000_constatson_False.npy", allow_pickle=True)

model.to("cpu")

running_sums = defaultdict(dict)
running_counts = defaultdict(dict)

for datapoint in tqdm.tqdm(dataset[:N_EQUATIONS]):
    clean, n_vars = datapoint["equation"], len(datapoint["variables"])
    Xcl, ycl = iclass.get_input(clean, n_variables=n_vars, number_of_points=200)
    layer_outputs = get_layer_outputs(Xcl, ycl, model, return_cpu=True)
    activations_clean = layer_to_head(layer_outputs)

    for key1, subdict in activations_clean.items():
        for key2, value in subdict.items():
            if not torch.isnan(value).any():
                if key2 not in running_sums[key1]:
                    # First valid occurrence: initialize the sum and count.
                    running_sums[key1][key2] = value.clone()
                    running_counts[key1][key2] = 1
                else:
                    running_sums[key1][key2] += value
                    running_counts[key1][key2] += 1

# Compute the averages from the running sums and counts.
avg_activations_cpu = defaultdict(dict)
for key1, subdict in running_sums.items():
    for key2, sum_value in subdict.items():
        count = running_counts[key1][key2]
        avg_activations_cpu[key1][key2] = sum_value / count

avg_activations_cpu = {k: dict(v) for k, v in avg_activations_cpu.items()}
np.save(f"data/Arco/MeanPatching/mean_patching_{N_EQUATIONS}.npy", avg_activations_cpu)
print(f"Files saved to data/Arco/MeanPatching/mean_patching_{N_EQUATIONS}.npy")
