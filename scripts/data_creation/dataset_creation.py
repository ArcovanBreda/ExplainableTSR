import sys
import os
# Get the absolute path of ../src
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..",))

temp = sys.path

import numpy as np
import multiprocessing
from multiprocessing import Manager
import click
import warnings
from tqdm import tqdm
import json
import time
import signal
from pathlib import Path
import pickle
from sympy import lambdify 
import copyreg
import types
from itertools import chain
import traceback
import h5py
import warnings

# sys.path = src_path
# print("hallo", sys.path)
sys.path.append('/home/arco/Downloads/Master/MscThesis/ExplainableDSR/')
from src.nesymres.dataset import generator
from src.nesymres import dclasses
from src.nesymres.utils import create_env, H5FilesCreator
from src.nesymres.utils import code_unpickler, code_pickler
sys.path = temp

class Pipepile:
    def __init__(self, env: generator.Generator, number_of_equations, eq_per_block, h5_creator:H5FilesCreator,  is_timer=False):
        self.env = env
        #manager = Manager()
        #self.cnt = manager.list()
        self.is_timer = is_timer
        self.number_of_equations = number_of_equations
        self.fun_args = ",".join(chain(list(env.variables),env.coefficients))
        self.eq_per_block = eq_per_block
        self.h5_creator=h5_creator

    def create_block(self,block_idx):
        print("hallo")
        block = []
        counter = block_idx
        hlimit = block_idx + self.eq_per_block
        while counter < hlimit and counter < self.number_of_equations:
            res = self.return_training_set(counter)
            block.append(res)
            counter = counter + 1
        # print("hallo", block)
        self.h5_creator.create_single_hd5_from_eqs((block_idx//self.eq_per_block, block))
        return 1

    def handler(self,signum, frame):
        raise TimeoutError

    def return_training_set(self, i) -> dclasses.Equation:
        np.random.seed(i)
        while True:
            try:
                res = self.create_lambda(np.random.randint(2**32-1))
                print(res)
                assert type(res) == dclasses.Equation
                return res
            except TimeoutError:
                signal.alarm(0)
                continue
            except generator.NotCorrectIndependentVariables:
                signal.alarm(0)
                continue
            except generator.UnknownSymPyOperator:
                signal.alarm(0)
                continue
            except generator.ValueErrorExpression:
                signal.alarm(0)
                continue
            except generator.ImAccomulationBounds:
                signal.alarm(0)
                continue
            except RecursionError:
                #Due to Sympy 
                signal.alarm(0)
                continue
            except KeyError:
                signal.alarm(0)
                continue
            except TypeError:
                signal.alarm(0)
            except Exception as E:
                continue


    def create_lambda(self, i):
        print(":", end=" ")
        if self.is_timer:
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(1)
        # print("-", end=" ")
        prefix, variables = self.env.generate_equation(np.random)
        print("sub sub")
        prefix = self.env.add_identifier_constants(prefix)
        consts =  self.env.return_constants(prefix)
        infix, _  = self.env._prefix_to_infix(prefix, coefficients=self.env.coefficients, variables=self.env.variables)
        consts_elemns = {y:y for x in consts.values() for y in x}
        constants_expression = infix.format(**consts_elemns)
        eq = lambdify(
            self.fun_args,
            constants_expression,
            modules=["numpy"],
        )
        res = dclasses.Equation(expr=infix, code=eq.__code__, coeff_dict=consts_elemns, variables=variables)
        print(res)
        # signal.alarm(0) 
        return res

# python3 scripts/data_creation/dataset_creation.py --number_of_equations 100 --debug

@click.command()
@click.option(
    "--number_of_equations",
    default=200,
    help="Total number of equations to generate",
)
@click.option(
    "--eq_per_block",
    default=5e4,
    help="Total number of equations to generate",
)
@click.option("--debug/--no-debug", default=True)
def creator(number_of_equations,eq_per_block, debug):
    copyreg.pickle(types.CodeType, code_pickler, code_unpickler) #Needed for serializing code objects
    total_number = number_of_equations
    cpus_available = multiprocessing.cpu_count()
    eq_per_block= min(total_number//cpus_available, int(eq_per_block))
    print("There are {} equations per block. The progress bar will have this resolution".format(eq_per_block) )
    warnings.filterwarnings("error")
    env, param, config_dict = create_env("dataset_configuration.json")
    if not debug:
        folder_path = Path(f"data/raw_datasets/{number_of_equations}") 
    else:
        folder_path = Path(f"data/raw_datasets/debug/{number_of_equations}")
    h5_creator = H5FilesCreator(target_path=folder_path)
    env_pip = Pipepile(env, 
                      number_of_equations=number_of_equations, 
                      eq_per_block=eq_per_block,
                      h5_creator=h5_creator,
                      is_timer=not debug)
    starttime = time.time()
    func = []
    res = []
    counter = []
    if not debug:
        try:
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                max_ = total_number
                with tqdm(total=max_) as pbar:
                    for f in p.imap_unordered(
                        env_pip.create_block, range(0, total_number, eq_per_block)
                    ):
                        pbar.update()
                        #res.append(f)
        except:
            print(traceback.format_exc())

    else:
        list(map(env_pip.create_block, tqdm(range(0, total_number, eq_per_block))))

    dataset = dclasses.DatasetDetails(
                               config=config_dict,
                               total_coefficients=env.coefficients,
                               total_variables=list(env.variables),
                               word2id=env.word2id,
                               id2word=env.id2word,
                               una_ops=env.una_ops,
                               bin_ops=env.una_ops,
                               rewrite_functions=env.rewrite_functions,
                               total_number_of_eqs=number_of_equations,
                               eqs_per_hdf=eq_per_block,
                               generator_details=param)
    print("Expression generation took {} seconds".format(time.time() - starttime))
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    t_hf = h5py.File(os.path.join(folder_path, "metadata.h5") , 'w')
    t_hf.create_dataset("other", data=np.void(pickle.dumps(dataset)))
    t_hf.close()


if __name__ == "__main__":
    creator()