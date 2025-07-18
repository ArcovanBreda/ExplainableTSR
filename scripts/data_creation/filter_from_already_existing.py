import click
import numpy as np
from nesymres.utils import load_eq, load_metadata_hdf5
from nesymres.dataset.data_utils import evaluate_fun
import pandas as pd
from collections import defaultdict
from nesymres.dataset.data_utils import create_uniform_support, sample_symbolic_constants, evaluate_fun, return_dict_metadata_dummy_constant
#from nesymres.benchmark import return_order_variables
from torch.distributions.uniform import Uniform
import torch
#from nesymres import dclasses
import multiprocessing
from tqdm import tqdm
import os
#from pathlib import Path
#import pickle
import warnings
from sympy import lambdify,sympify


def evaluate_validation_set(validation_eqs: pd.DataFrame, support) -> set:
    res = set()

    modules = {
    "add": np.add,
    "mul": np.multiply,
    "sub": np.subtract,
    "div": np.divide,
    "sqrt": np.sqrt,
    "pow2": lambda x: np.power(x, 2),
    "pow3": lambda x: np.power(x, 3),
    "pow4": lambda x: np.power(x, 4),
    "pow5": lambda x: np.power(x, 5),
    "log": np.log,
    "ln": np.log,
    "exp": np.exp,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "Abs": np.abs,
}

    for _, row in validation_eqs.iterrows():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            variables = [f"x_{i}" for i in range(1, 1 + support.shape[0])]
            # Define the modules dictionary to include numpy functions explicitly
            curr = np.array(lambdify(variables, row["eq"], modules=modules)(*support)).astype('float16')
            curr = tuple(x if not np.isnan(x) else "nan" for x in curr)
            res.add(curr)
    return res

class Pipeline:
    def __init__(self, data_path, metadata, support, target_image: list, validation_eqs: set):
        self.data_path = data_path
        self.metadata = metadata
        self.support = support
        self.target_image_l = target_image
        self.target_image = set(target_image)
        self.validation_eqs_l = validation_eqs
        self.validation_eqs = set(validation_eqs)

    def is_valid_and_not_in_validation_set(self, idx:int) -> bool:
        """
        Return True if an equation is not the validation set and is numerically meaningfull (i.e. values all differs from nan, -inf, +inf, all zeros),
               We test whether is in the validation dataset both symbolically and numerically
        Args:
            idx: index of the equation in the dataset
        
        """
        eq = load_eq(self.data_path, idx, self.metadata.eqs_per_hdf)
        # print("\n\n\n")
        # print("eq:", eq)

        dict_costs = return_dict_metadata_dummy_constant(self.metadata)
        consts = torch.stack([torch.ones([int(self.support.shape[1])])*dict_costs[key] for key in dict_costs.keys()])
        input_lambdi = torch.cat([self.support,consts],axis=0)
        assert input_lambdi.shape[0]  == len(self.metadata.total_coefficients) + len(self.metadata.total_variables)

        #Symbolic Checking
        const, dummy_const = sample_symbolic_constants(eq)
        eq_str = sympify(eq.expr.format(**dummy_const))
        if str(eq_str) in self.validation_eqs:
            # Skeleton in val
            return idx, False

        #Numerical Checking        
        args = [ eq.code,input_lambdi ]
        y = evaluate_fun(args)
        
        #Subtle bug tuple([np.nan]) == tuple([np.nan]) returns true however, tuple([np.nan+0]) == tuple([np.nan]) returns false. 
        #For avoiding missing numerical equivalences we convert all nan to string
        curr = [x if not np.isnan(x) else "nan" for x in y] 
        # y = evaluate_fun(args)
        val = tuple(curr)
        if val == tuple([]):
            # Not an equation
            return idx, False
        if val == tuple([float("-inf")]*input_lambdi.shape[-1]):
            # All Inf
            return idx, False
        if val == tuple([float("+inf")]*input_lambdi.shape[-1]):
            # All -inf 
           return idx, False
        if val == tuple([float(0)]*input_lambdi.shape[-1]):
            # All zeros
            return idx, False
        if val == tuple(["nan"]*input_lambdi.shape[-1]):
            # All nans
            return idx, False
        if val in self.target_image:
            # Numerically identical to val
            return idx, False
        return idx, True
    
        

@click.command()
@click.option("--data_path", default="data/raw_datasets/10000000/")
@click.option("--csv_path", default="data/benchmark/nc_old.csv")
@click.option("--debug/--no-debug", default=False)
def main(data_path, csv_path,debug):
    print("Loading metadata")
    metatada = load_metadata_hdf5(data_path)
    #data = load_dataset(data_path)
    if csv_path == "None":
        validation = pd.DataFrame(columns=["eq"])
    else:
        validation = pd.read_csv(csv_path)
        print(validation.info)
    sampling_distribution = Uniform(-25,25)
    num_p = 400
    support = create_uniform_support(sampling_distribution, len(metatada.total_variables), num_p)
    print("Creating image for validation set")
    target_image = evaluate_validation_set(validation, support)
    pipe = Pipeline(data_path, metatada, support, target_image, list(validation["eq"]))
    print("Starting finding out index of equations present in the validation set or wih numerical problems")
    total_eq = int(metatada.total_number_of_eqs)
    res = []
    if not debug:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            with tqdm(total=total_eq) as pbar:
                for evaled in p.imap_unordered(pipe.is_valid_and_not_in_validation_set, list(range(total_eq)),chunksize=10000):
                    pbar.update()
                    res.append(evaled)
    else:
        res = list(map(pipe.is_valid_and_not_in_validation_set, tqdm(range(total_eq))))
    
    # print(f"Total number of good equations {len([x for x in p if x[1]])}")
    print(len(res))
    print(os.path.join(data_path, "filtered"))
    np.save(os.path.join(data_path, "filtered"),res)

if __name__=="__main__":
    main()