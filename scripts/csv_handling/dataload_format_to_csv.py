import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Manager
import click
import warnings
from tqdm import tqdm
import json
import os
import time
import signal
from pathlib import Path
import pickle
from sympy import lambdify

import copyreg
import types
from itertools import chain
import traceback
import sympy as sp
import random
import hydra
from tqdm import tqdm

import sys
sys.path.append('/home/arco/Downloads/Master/MscThesis/ExplainableDSR/src')
from nesymres.dataset import generator
from nesymres.utils import create_env, load_metadata_hdf5, load_eq
from nesymres.dataset import data_utils 
from nesymres.dataset.sympy_utils import add_multiplicative_constants, add_additive_constants


def create_df(path,metadata,cfg, constats_on = False):
    rows = {"eq": [], "support": [], "num_points": []}
    for idx in tqdm(range(metadata.total_number_of_eqs)):
        eq = load_eq(path, idx, metadata.eqs_per_hdf)
        print(eq)
        w_const, wout_consts = data_utils.sample_symbolic_constants(eq,cfg.dataset_test.constants)
        print(wout_consts)
        print(w_const)
        print()
        if constats_on:
            dict_const = w_const
        else:
            dict_const = wout_consts
        eq_string = eq.expr.format(**dict_const)
        eq_string = str(sp.simplify(eq_string))
        print(eq_string)
        d = {}
        if not eq.support:
            for var in eq.variables:
                d[var] = cfg.dataset_test.fun_support
        rows["eq"].append(str(eq_string))
        rows["support"].append(str(d))
        rows["num_points"].append(cfg.dataset_test.max_number_of_points)
    dataset = pd.DataFrame(rows)
    return dataset


@hydra.main(config_name="../config")
def converter(cfg):
    df = pd.DataFrame()
    path = hydra.utils.to_absolute_path(cfg.raw_test_path)
    print(path)
    number_of_entries = cfg['raw_test_path'].split("/")[-1]

    os.makedirs(hydra.utils.to_absolute_path(f"test_set/{number_of_entries}"), exist_ok=True)

    metadata = load_metadata_hdf5(path)
    # df = create_df(path,metadata,cfg,constats_on = False)
    # df.to_csv(hydra.utils.to_absolute_path(f"test_set/{number_of_entries}/test_nc.csv"))
    df = create_df(path,metadata,cfg,constats_on = True)
    df.to_csv(hydra.utils.to_absolute_path(f"test_set/{number_of_entries}/test_wc.csv"))

    # dataset.to_csv(hydra.utils.to_absolute_path("test_set/test.csv"))
    # with open(hydra.utils.to_absolute_path("data/benchmark/test_csv"), "wb") as file:
    #     pickle.dump(dataset, file)

if __name__ == "__main__":
    converter()