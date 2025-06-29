from src.nesymres.utils import create_env
import numpy as np
from tqdm import tqdm
from sympy import lambdify
import sympy as sp
from itertools import chain
import hydra
import concurrent.futures

import sys
sys.path.append('/home/arco/Downloads/Master/MscThesis/ExplainableDSR/')
from src.nesymres import dclasses
from src.nesymres.dataset import data_utils 

import warnings
warnings.filterwarnings("ignore")

import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Simplification timed out")

def simplify_with_timeout(expr, timeout_seconds=5):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        simplified_expr = sp.simplify(expr)
    except TimeoutException:
        # Return None if simplification takes too long
        simplified_expr = None
    finally:
        signal.alarm(0)
    return simplified_expr



@hydra.main(config_name="scripts/config.yaml")
def main(cfg):
    # Config
    N = 1_000_000
    constats_on = True # if False: sin(x_1) + exp(x_2) if True: sin(x_1 + 0.543) + exp(x_2)*0.6
    path = "/home/arco/Downloads/Master/MscThesis/ExplainableDSR/"

    # create generator
    gen, _, _ = create_env(path + "dataset_configuration.json")
    fun_args = ",".join(chain(list(gen.variables), gen.coefficients))

    # generate expressions
    eq_strings = parallel_generate_expressions(N, gen, fun_args, constats_on, cfg)
    np.save(path + "data/Arco/Datasets/" + f"expressions_{N}_constatson_{constats_on}", eq_strings)


def generate_expression(counter, gen, fun_args, constats_on, cfg):
    try:
        np.random.seed(counter)
        prefix, infix, variables = gen.generate_equation(rng=np.random)
    except Exception:
        return None

    prefix = gen.add_identifier_constants(prefix)
    consts = gen.return_constants(prefix)
    infix, _ = gen._prefix_to_infix(prefix, coefficients=gen.coefficients, variables=gen.variables)
    consts_elemns = {y: y for x in consts.values() for y in x}
    constants_expression = infix.format(**consts_elemns)
    eq = lambdify(
        fun_args,
        constants_expression,
        modules=["numpy"],
    )
    eq = dclasses.Equation(expr=infix, code=eq.__code__, coeff_dict=consts_elemns, variables=variables)
    w_const, wout_consts = data_utils.sample_symbolic_constants(eq, cfg.dataset_test.constants)

    if constats_on:
        dict_const = w_const
    else:
        dict_const = wout_consts

    eq_string = eq.expr.format(**dict_const)
    eq_string = simplify_with_timeout(eq_string, timeout_seconds=1)

    if eq_string is None:
        return None

    return {"equation": str(eq_string), "variables": variables}


def worker(counter, gen, fun_args, constats_on, cfg):
    return generate_expression(counter, gen, fun_args, constats_on, cfg)


def parallel_generate_expressions(N, gen, fun_args, constats_on, cfg):
    eq_strings = []
    with tqdm(total=N, desc="Generating expressions") as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Pass extra arguments to worker using executor.map
            for eq_string in executor.map(worker, range(N),
                                            [gen]*N,
                                            [fun_args]*N,
                                            [constats_on]*N,
                                            [cfg]*N,
                                            chunksize=100):
                if eq_string:
                    eq_strings.append(eq_string)
                pbar.update(1)
    return eq_strings  # fast


if __name__ == "__main__":
    main()
