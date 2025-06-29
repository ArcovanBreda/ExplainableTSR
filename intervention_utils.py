from src.nesymres.architectures.model import Model
from src.nesymres.utils import load_metadata_hdf5
from src.nesymres.dclasses import FitParams, BFGSParams
from functools import partial
import torch
from sympy import lambdify
import json
import omegaconf
from src.nesymres.architectures.model import EncoderOnly
from nnsight import NNsight
import nnsight
from sympy import sympify, Mul, Symbol, Function
from sympy.core.function import AppliedUndef
import sympy as sp
from utils import evaluate_formula_samples, infix_to_prefix 
import os

class intervension:
    def __init__(self, eq_setting_path="jupyter/100M/eq_setting.json",
                 cfg_path="jupyter/100M/config.yaml",
                 weights_path="weights/100M.ckpt") -> None:

        # Load config settings
        # print("Attempting to open:", os.path.abspath(eq_setting_path))
        with open(eq_setting_path, 'r') as json_file:
            self.eq_setting = json.load(json_file)
        # print("Attempting to open:", os.path.abspath(cfg_path)) 
        self.cfg = omegaconf.OmegaConf.load(cfg_path)
        self.cfg.inference.word2id = self.eq_setting["word2id"]
        self.cfg.inference.id2word = {int(k): v for k, v in self.eq_setting["id2word"].items()}
        self.cfg.inference.total_variables = self.eq_setting["total_variables"]

        self.id2word, self.word2id = self.get_translation()

        self.Model = EncoderOnly.load_from_checkpoint(weights_path,
                                                      cfg=self.cfg)
        self.Model.eval()
        if torch.cuda.is_available():
            print("Running on GPU")
            self.Model.cuda()
            self.device = 'cuda'
        else:
            print("Running on CPU")
            self.device = 'cpu'
        self.nnModel = NNsight(self.Model)

    def get_translation(self):
        id2word = self.eq_setting["id2word"]
        id2word['0'] = "<PAD>"
        id2word['1'] = "<S>"
        id2word['2'] = "<F>"
        word2id = self.eq_setting["word2id"]
        word2id["<S>"] = 1
        word2id["log"] = 17
        return id2word, word2id

    def decode_ids(self, ids):
        return [self.id2word.get(str(id), "NOTHING") for id in ids]

    def encode_ids(self, expr):
        return [self.word2id.get(str(word), "NOTHING") for word in expr]

    def decode_output(self, output):
        """
        returns the ints and characters according to the argmax pred.
        """
        output = torch.mean(output, dim=1).to("cpu")
        argmax_output = torch.argmax(output, dim=1)
        ids = argmax_output.tolist()
        # print(output.shape, argmax_output.shape)
        values = output[torch.arange(output.shape[0]), argmax_output].tolist()
        formula = self.decode_ids(ids)
        return (argmax_output, values), (formula, values)

    def get_input(self, target_eq, number_of_points=200, n_variables=3, seed=42, supports=None):
        """Given an equation string, return input/output pairs."""
        torch.manual_seed(seed)
        # Determine support ranges
        if supports is None:
            max_supp = self.cfg.dataset_train.fun_support["max"]
            min_supp = self.cfg.dataset_train.fun_support["min"]
            supports = [(min_supp, max_supp)] * len(self.eq_setting["total_variables"])

        # Ensure supports is for exactly n_variables, fill with zeros if needed
        supports = supports[:n_variables] + [(0, 0)] * (3 - len(supports))

        # Generate input values for each variable based on its support range
        X = torch.stack([
            torch.rand(number_of_points) * (max_val - min_val) + min_val
            for min_val, max_val in supports
        ], dim=1)

        # Ensure the tensor has the correct shape [number_of_points, n_variables]
        if X.shape[1] < 3:
            zeros_to_add = torch.zeros(number_of_points, 3 - X.shape[1])
            X = torch.cat([X, zeros_to_add], dim=1)

        # Compute output values
        y = evaluate_formula_samples(target_eq, X)

        return X, y
