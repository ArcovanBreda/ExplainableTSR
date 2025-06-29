import numpy as np
from tqdm import tqdm
import math
import torch
import warnings
warnings.filterwarnings("ignore")
import sympy as sp
from sympy import lambdify
import re
import omegaconf
from src.nesymres.dclasses import FitParams, NNEquation, BFGSParams
import json
from src.nesymres.architectures.model import Model
from itertools import product
import copy
import signal
import sympy as sp
from collections import Counter
from functools import partial, reduce
import operator
import sys
import os
from datetime import datetime
import sympy as sp
from functools import reduce


def get_params_fit(eq_setting_path: str= 'jupyter/100M/eq_setting.json',
         cfg_path: str= 'jupyter/100M/config.yaml',
         weights_path: str="weights/100M.ckpt",
         max_len:int=None, beam_size:int=None):

    with open(eq_setting_path, 'r') as json_file:
        eq_setting = json.load(json_file)

    cfg = omegaconf.OmegaConf.load(cfg_path)
    if max_len is not None:
        cfg.architecture.length_eq = max_len
    if beam_size is not None:
        cfg.inference.beam_size = beam_size

    bfgs = BFGSParams(
        activated=cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )
    params_fit = FitParams(
        word2id=eq_setting["word2id"],
        id2word={int(k): v for k, v in eq_setting["id2word"].items()},
        una_ops=eq_setting["una_ops"],
        bin_ops=eq_setting["bin_ops"],
        total_variables=list(eq_setting["total_variables"]),
        total_coefficients=list(eq_setting["total_coefficients"]),
        rewrite_functions=list(eq_setting["rewrite_functions"]),
        bfgs=bfgs,
        beam_size=cfg.inference.beam_size
    )

    return cfg, params_fit


def evaluate_formula_samples(formula: str, X: torch.Tensor) -> torch.Tensor:
    """
    Evaluates a mathematical formula for corresponding samples of variable values.

    Args:
        formula (str): The mathematical formula as a string.
        X (torch.Tensor): Tensor containing variable values [rows, 3].

    Returns:
        torch.Tensor: A tensor of evaluated results for corresponding samples.
    """
    formula = make_human_readable(formula)
    func = lambdify(["x_1", "x_2", "x_3"], formula, "numpy")

    if not any(var in formula for var in ["x_1", "x_2", "x_3"]):
        # Formula does not depend on variables, return constant tensor
        result = func(0, 0, 0)  # Evaluate once for constant value
        return torch.full((X.shape[0],), result, dtype=torch.float32)

    X_numpy = X.cpu().numpy()
    y = func(X_numpy[:, 0], X_numpy[:, 1], X_numpy[:, 2])  

    return torch.tensor(y, dtype=torch.float32)


def infix_to_prefix(expression, flat=True):
    def flatten_list(nested_list):
        """Helper function to flatten nested lists."""
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(flatten_list(item))
            else:
                flat_list.append(item)
        return flat_list

    def nest_binary_operator(operator, args):
        """Recursively folds args left-to-right: op(op(a, b), c)..."""
        return reduce(lambda acc, x: [operator, acc, x], args)

    def parse_prefix(expr):
        """Recursive function to convert infix to prefix."""
        if expr.is_Atom:
            return str(expr)
        else:
            operator = str(expr.func.__name__).lower()

            # For binary operators (add, sub, mul, div, pow)
            if operator in {'add', 'sub', 'mul', 'div', 'pow'}:
                args = [parse_prefix(arg) for arg in expr.args]
                return nest_binary_operator(operator, args)

            # For unary operators (abs, cos, sin, tan, etc.)
            elif operator in {'abs', 'acos', 'asin', 'atan', 'cos', 'cosh', 'coth',
                              'exp', 'ln', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'log'}:
                return [operator, parse_prefix(expr.args[0])]

            else:
                return [operator] + [parse_prefix(arg) for arg in expr.args]

    expr = sp.sympify(expression, evaluate=False)
    prefix_expr = parse_prefix(expr)

    return flatten_list(prefix_expr) if flat else prefix_expr


def prefix_to_infix(prefix_expr):
    stack = []
    binary_operators = {'add', 'mul', 'pow', 'div', 'sub'}
    unary_operators = {'abs', 'acos', 'asin', 'atan', 'cos', 'cosh', 'coth', 'exp', 'ln', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'log'}

    for token in reversed(prefix_expr):
        if token not in binary_operators and token not in unary_operators:
            stack.append(token)
        elif token in binary_operators:
            if len(stack) < 2:
                raise ValueError(f"Insufficient operands for binary operator '{token}'")
            operand1 = stack.pop()
            operand2 = stack.pop()
            expression = f"({operand1} {token} {operand2})"
            stack.append(expression)
        elif token in unary_operators:
            if len(stack) < 1:
                raise ValueError(f"Insufficient operands for unary operator '{token}'")
            operand = stack.pop()
            expression = f"{token}({operand})"
            stack.append(expression)

    return stack[0]


def make_human_readable(infix_expr):
    operator_map = {
        'add': '+',
        'mul': '*',
        'pow': '**',
        'sub': '-',
        'cos': 'cos',
        'sin': 'sin',
        'tan': 'tan',
        'div': '/',
        'ln': 'log',
        'asin': 'arcsin',
        'E': "2.71828"
    }

    for operator, symbol in operator_map.items():
        infix_expr = re.sub(rf'\b{operator}\b', symbol, infix_expr)
    return infix_expr


def get_head_index(output, head_idx, num_heads):
    """
    Calculate the index range for a specific attention head.
    Parameters:
    output (tensor or int): The total size of the output tensor or the number of elements (for an integer).
    head_idx (int): The index of the attention head (0-based).
    num_heads (int): The total number of attention heads.
    Returns:
    tuple: A tuple (f, t) where 'f' is the starting index and 't' is the ending index of the specified head.
    """
    if type(output) == int:
        pass
    else:
        output = output.shape[-1] # take the last index

    if head_idx is None:
        return 0, output

    if head_idx >= num_heads or head_idx < 0:
            raise ValueError(f"Index out of range, should be between 0-{num_heads-1}") 

    head_size = output // num_heads
    f = head_idx * head_size
    t = f + head_size
    return f, t


def get_layer_outputs(X, y, model, return_cpu=False):
    outputs = {}
    with model.trace(X, y) as tracer:
        # 5 main layers
        for selfatt, mab in product(range(5), range(2)):
            layer_path = f"enc.selfatt[{selfatt}].mab{mab}"
            layer = _get_nested_attr(model, layer_path)

            outputs[layer_path] = {
                "q": layer.fc_q.output.save(),
                "k": layer.fc_k.output.save(),
                "v": layer.fc_v.output.save(),
                "fc_o": layer.fc_o.output.save(),
                "ln0": layer.ln0.output.save(),
                "ln1": layer.ln1.output.save(),
            }

        # 6th projection layer
        for mab in range(2):
            layer_path = f"enc.selfatt1.mab{mab}"
            layer = _get_nested_attr(model, layer_path)
            outputs[layer_path] = {
                    "q": layer.fc_q.output.save(),
                    "k": layer.fc_k.output.save(),
                    "v": layer.fc_v.output.save(),
                    "fc_o": layer.fc_o.output.save(),
                    "ln0": layer.ln0.output.save(),
                    "ln1": layer.ln1.output.save(),
                }
        # output layer
        layer_path = "enc.outatt.mab"
        layer = _get_nested_attr(model, layer_path)
        outputs[layer_path] = {
                "q": layer.fc_q.output.save(),
                "k": layer.fc_k.output.save(),
                "v": layer.fc_v.output.save(),
                "fc_o": layer.fc_o.output.save(),
                "ln0": layer.ln0.output.save(),
                "ln1": layer.ln1.output.save(),
            }

    return outputs


def layer_to_head(layer_outputs):
    activation = {}
    typs = ["fc_o", "q", "k", "v", "ln0", "ln1"]
    # 5 main layers
    for selfatt, mab, typ, head_idx in product(range(5), range(2), typs, range(8)):
        layer_path = f"enc.selfatt[{selfatt}].mab{mab}"
        output = layer_outputs[layer_path][typ]

        if typ in ["fc_o", "ln0", "ln1"]:
            if head_idx == 0 and typ == "fc_o":
                activation[layer_path] = {}
            activation[layer_path][typ] = output
            continue

        f, t = get_head_index(output, head_idx, 8)
        typ_name = f"{typ}_{head_idx}"
        activation[layer_path][typ_name] = output[..., f:t]

    # 6th projection layer
    for mab, typ, head_idx in product(range(2), typs, range(8)):
        layer_path = f"enc.selfatt1.mab{mab}"
        output = layer_outputs[layer_path][typ]

        if typ in ["fc_o", "ln0", "ln1"]:
            if head_idx == 0 and typ == "fc_o":
                activation[layer_path] = {}
            activation[layer_path][typ] = output
            continue

        f, t = get_head_index(output, head_idx, 8)
        typ_name = f"{typ}_{head_idx}"
        activation[layer_path][typ_name] = output[..., f:t]

    # output layer
    layer_path = "enc.outatt.mab"
    activation[layer_path] = {"fc_o": layer_outputs[layer_path]["fc_o"],
                              "ln0": layer_outputs[layer_path]["ln0"],
                              "ln1": layer_outputs[layer_path]["ln1"]}

    for head_idx, typ in product(range(8), typs[1:]):
        output = layer_outputs[layer_path][typ]
        f, t = get_head_index(output, head_idx, 8)
        typ_name = f"{typ}_{head_idx}"
        activation[layer_path][typ_name] = output[..., f:t]

    return activation


def change_one_head(MHA_output, head_idx=0, num_heads=8, new_output=None):
        """Changes the values of one attention head, because the heads are 
        concatinated, we index. If new_output==None we fill it with zeros
        WARNING: REDUNDANT"""

        head_size = int(MHA_output.shape[2] / num_heads)

        f, t = get_head_index(512, head_idx, num_heads)
        if new_output is None:  # fill with zeros
            new_output = torch.zeros((MHA_output.shape[0],
                                    MHA_output.shape[1],
                                    head_size))
        elif new_output.shape[0] != t-f: # if we have just given the entire layer, split it up just like output
            new_output = new_output[f:t]

        output = MHA_output
        output[:, :, f:t] = new_output
        return output


def prepare_batch(X, y, target, device="cuda"):
        """
        prepare a batch for the forward pass
        input:
            X: torch.tensor([number of points, number of variables])
            y: torch.tensor([number of points])
            target: list([id's of equation (see get_target)]) 
        output:
            batch: [batch0=X+y, batch1=target]
        """

        n_samples, n_vars = X.shape[0], X.shape[-1] + 1
        batch0 = torch.cat((X, y.unsqueeze(dim=1)), dim=1)
        batch0 = batch0.T.view(1, n_vars, n_samples).to(device)
        target = torch.tensor(target).view(1, len(target)).to(device)

        return [batch0, target]


def _get_nested_attr(obj, attr_path):
    """Retrieve a nested attribute from an object, specified by a dot-separated path."""
    for part in attr_path.split('.'):
        if '[' in part and ']' in part:
            attr_name, idx = part[:-1].split('[')
            obj = getattr(obj, attr_name)[int(idx)]
        else:
            obj = getattr(obj, part)
    return obj


def _set_nested_attr(obj, attr_path, value):
    """Set a nested attribute on an object, specified by a dot-separated path."""
    parts = attr_path.split('.')
    for part in parts[:-1]:  # Traverse up to the second-last part
        if '[' in part and ']' in part:
            attr_name, idx = part[:-1].split('[')
            obj = getattr(obj, attr_name)[int(idx)]
        else:
            obj = getattr(obj, part)
    # Set the attribute on the last part
    last_part = parts[-1]
    if '[' in last_part and ']' in last_part:
        attr_name, idx = last_part[:-1].split('[')
        getattr(obj, attr_name)[int(idx)] = value
    else:
        setattr(obj, last_part, value)


def edit_encoder(X, y, layer_paths, head_indices, model, new_outputs=None):
    """
    Edit the outputs of specified layers in the model.

    Input:
        model: NNisight model.
        X: torch.tensor([number of points, number of variables])
        y: torch.tensor([number of points])
        layer_paths: A list of strings representing the paths to the layers to be edited,
                     e.g., ['enc.selfatt[0].mab0', 'enc.selfatt[1].mab0'].
        head_indices: A list of head indices to modify for each layer.
        new_outputs: A list of new outputs for the layers.

    Output:
        The edited output of the model.
    """
    # make it work with 1 or multiple input types
    if type(layer_paths) == str:
        layer_paths = [layer_paths]
    if type(new_outputs) == str:
        new_outputs = [new_outputs]
    elif new_outputs is None:
         new_outputs = [None] * len(layer_paths)
    if type(head_indices) != list:
        head_indices = [head_indices] * len(layer_paths)

    # Retrieve and modify outputs from specified layers
    modified_outputs = {}
    for layer_path, head_idx, new_output in zip(layer_paths, head_indices, new_outputs):
        layer = _get_nested_attr(model, layer_path)
        with model.trace(X, y) as tracer:
            output = layer.output.save() 

        # Modify the output for the given head
        modified_outputs[layer_path] = change_one_head(output, head_idx=head_idx, new_output=new_output)

    # Edit the model with the modified outputs
    with model.edit() as model_edit:
        for layer_path, new_output in modified_outputs.items():
            _set_nested_attr(model_edit, layer_path + ".output", new_output)

    # Get the edited output of the model
    with model_edit.trace(X, y):
        output_edit = model_edit.output.save()

    return output_edit


def get_nested_attr(obj, attr_path):
    """
    Handles both dot-separated attributes and indexed attributes (e.g., 'layer[0].weight').
    """
    attrs = attr_path.split('.')
    for i, attr in enumerate(attrs):
        if "[" in attr and "]" in attr:
            base_attr, index = attr.split("[")
            index = int(index[:-1])
            obj = getattr(obj, base_attr)[index]
        else:
            obj = getattr(obj, attr)
    return obj


def get_attr_from_path(obj, path):
    current = obj
    components = path.split('.')
    for comp in components:
        match = re.match(r'(\w+)\[(-?\d+)\]$', comp)
        if match:
            name = match.group(1)
            index = int(match.group(2))
            current = getattr(current, name)
            current = current[index]
        else:
            current = getattr(current, comp)
    return current


def check_tolerance(og, edited, tol=0.1):
        """
        Check if the edited loss is within the specified tolerance percentage of the original loss.

        Args:
            og (float): Original loss.
            edited (float): Edited loss.
            tol (float): Tolerance percentage as a decimal (e.g., 0.1 for 10%).

        Returns:
            bool: True if the edited loss is within the tolerance, False otherwise.
        """
        # Calculate the tolerance range (e.g., ± 10% of og)
        tolerance_limit = og * tol

        # Check if the difference between original and edited loss is within the tolerance limit
        return abs(og - edited) <= tolerance_limit


def check_size(variable):
    import sys
    return sys.getsizeof(variable) / 1024


def check_same_decode(form1, form2, sign='sin'):
    """
    Return True if everything in the formula until the first occurrence of `sign` is the same in both formulas.
    Return False if `sign` is not found in either form1 or form2.
    """
    if sign not in form1:
        return False, None

    index = form1.index(sign)
    return form1[:index] == form2[:index], ["<S>"] + form1[:index]


def same_recons(dataset, target_operation):
    same_reconstructions = []
    for datapoint in dataset:
        formula = datapoint["pred"]
        formula_corrupted = datapoint["predCorr"]

        if target_operation == "sin-cos-tan":
            operation = find_target_operation(str(datapoint["equation"]))
        else:
            operation = target_operation

        same, eq = check_same_decode(formula, formula_corrupted, sign=operation)
        if same:
            same_reconstructions.append([datapoint, eq])
    return same_reconstructions


def get_all_names():
    names = [f"L{l+1}-MAB{m+1}_{name}" for l in range(5) for m in range(2) for name in (['MLP'] + [f'H{i+1}' for i in range(8)])]
    return names


def get_layer_names():
    "NOTE we do not take into account the ln0 ln1"
    base_names = [f"enc.selfatt[{l}].mab{m}" for l in range(5) for m in range(2)]
    base_names == ["enc.selfatt1.mab0", "enc.selfatt1.mab1", "enc.outatt.mab"]
    suffixes = ['MLP'] + [str(i) for i in range(8)]
    return np.array([f"{base}_{suffix}" for base in base_names for suffix in suffixes])


def get_layer_names_new():
    "NOTE we do not take into account the ln0 ln1"
    base_names = ["enc.selfatt1.mab0", "enc.selfatt1.mab1", "enc.outatt.mab"]
    base_names += [f"enc.selfatt[{l}].mab{m}" for l in range(5) for m in range(2)]
    suffixes = ['MLP'] + [str(i) for i in range(8)]
    return np.array([f"{base}_{suffix}" for base in base_names for suffix in suffixes])


def replace_constants(prefix_formula):
    return ['c' if re.fullmatch(r'-?\d+/\d+|-?\d+\.\d+|-?\d+', token) else token for token in prefix_formula]


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
        simplified_expr = None
    finally:
        signal.alarm(0)
    return simplified_expr


def replace_c_unique(expr, target):
    """
    Recursively traverse the expression tree and replace every occurrence of `target`
    with a new unique symbol. Returns the new expression and the list of dummy symbols.
    """
    dummy_symbols = []
    counter = [0]            

    def rec(e):
        if e.is_Atom:
            if e == target:
                new_sym = sp.symbols(f"{target}_{counter[0]}")
                counter[0] += 1
                dummy_symbols.append(new_sym)
                return new_sym
            else:
                return e
        new_args = tuple(rec(arg) for arg in e.args)
        return e.func(*new_args)

    new_expr = rec(expr)
    return new_expr, dummy_symbols


def generate_permutations(expr_str, constants, symbol='c'):
    """
    Generate all permutations by replacing each occurrence of `symbol`
    with one of the values in `constants`. If `symbol` is not found,
    return the input as a single-item list.
    """
    expr = sp.sympify(expr_str)
    target = sp.Symbol(symbol)

    if symbol not in expr_str:
        return [expr_str]

    new_expr, dummy_symbols = replace_c_unique(expr, target)

    permutations = []
    for values in product(constants, repeat=len(dummy_symbols)):
        subs_dict = dict(zip(dummy_symbols, values))
        permuted_expr = new_expr.subs(subs_dict)
        permutations.append(str(permuted_expr))

    return permutations


def flatten(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


def get_topk_acc(counter: Counter, threshold: int) -> float:
    total_count = sum(counter.values())
    below_threshold = sum(v for k, v in counter.items() if k <= threshold)  
    return below_threshold / total_count if total_count > 0 else 0.0  # Proportion


def from_name_to_layer(name, mean_patches):
    splitted_name = name.split(".")
    mab, head_idx = splitted_name[-1].split("_")
    new_name = ".".join(splitted_name[:-1] + [mab])
    if head_idx == "MLP":
        return new_name + ".fc_o", None, mean_patches[new_name]["fc_o"]
    else:
        new_name = new_name
        values = [mean_patches[new_name][f"q_{head_idx}"], 
                  mean_patches[new_name][f"k_{head_idx}"], 
                  mean_patches[new_name][f"v_{head_idx}"]]
        names = [new_name + ".fc_q", new_name + ".fc_k", new_name + ".fc_v"]
        head_idx = [int(head_idx)] * 3
        return names, head_idx, values


def from_names_to_encoder_info(names, indices, mean_patches):
    info_dict = dict()
    for i, name in enumerate(names):
        if i not in indices:
            continue
        _, head = name.split("_")
        n, h, v = from_name_to_layer(name, mean_patches)

        if head == "MLP":
            info_dict[n] = {"headidx": None, "new_output": v}
        else:
            f, t = get_head_index(512, h[0], 8)

            for na, va in zip(n, v):
                rangee = list(range(f, t))
                if info_dict.get(na, None) is not None:
                    info_dict[na]["headidx"] += [rangee]
                    info_dict[na]["new_output"] += [va]
                else:
                    # Initialize the new entry with the given headidx and new_output
                    info_dict[na] = {"headidx": [rangee], "new_output": [va]}
    return info_dict


def apply_patches_to_model(model, patch_info):
    with model.edit() as test_model:
        for layer_path, head_idxs, new_outputs in patch_info:
            full_path = f"{layer_path}.output"
            target = get_attr_from_path(model, full_path)

            if head_idxs is None: # MLP
                target[:] = new_outputs
            else: # Attention head
                for h, o in zip(head_idxs, new_outputs):
                    target[..., :, h[0]:h[-1]+1] = o
    return test_model


def get_encoders(model, Xs, ys):
    encoders = []
    for X_, y_ in zip(Xs, ys):
            with model.trace(X_, y_):
                output = model.output.save()
            encoders.append(output)
    return encoders


def evaluate_performance(model,
                         intervention_class,
                         encoders,
                         equations,
                         eqs_untill_target,
                         target_operation,
                         return_activation=False,
                         debug=False):
    ranks = []
    softmax_activations = []

    if target_operation == "log":
        target_operation = "ln"
    elif target_operation == "monomial":
        target_operation = "mul"

    for encoder, eq, full_eq in zip(encoders, eqs_untill_target, equations):
        input_tokens = intervention_class.encode_ids(eq)
        next_pred = model.decode_one_step(
            encoder=encoder,
            input_tokens=input_tokens
        ).cpu()

        next_pred_f = torch.softmax(next_pred, dim=-1)
        idxs = torch.argsort(next_pred, descending=True).numpy().astype(int)
        sorted_pred = intervention_class.decode_ids(idxs)

        if target_operation == "sin-cos-tan":
            operation = find_target_operation(full_eq)
        elif target_operation == "posynomial":
            operation = full_eq[len(eq)]
        else:
            operation = target_operation

        # Find rank or assign 58 (not found)
        try:
            rank = np.where(np.array(sorted_pred) == operation)[0][0]
        except IndexError:
            rank = 58

        ranks.append(rank + 1)  # +1 to match Top-k convention (1-indexed)
        softmax_activations.append(next_pred_f[idxs][rank].item())

    counter = Counter(ranks)

    if debug:
        return ranks, counter, np.array(softmax_activations)
    elif return_activation:
        return ranks, counter, np.array(softmax_activations)
    else:
        return ranks, counter


def find_target_operation(equation):
    match = re.search(r"(sin|cos|tan)", equation)

    if match:
        operation = match.group(0)
    else:
        operation = "unknown"
        print(f"Equation: {equation}, Operation: {operation}")
    return operation


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def all_operators():
    return ['<PAD>', '<S>', '<F>', 'c', 'x_1', 'x_2', 'x_3', 'abs', 
            'acos', 'add', 'asin', 'atan', 'cos', 'cosh', 'coth', 
            'div', 'exp', 'ln', 'mul', 'pow', 'sin', 'sinh', 'sqrt', 
            'tan', 'tanh', '-3', '-2', '-1', '0', '1', '2', '3']


def all_unary_operators():
    return ['abs', 'cos', 'cosh', 'exp', 'ln', 'log', 
            'sin', 'sinh', 'sqrt', 'tan', 'tanh']


def all_binary_operators():
    return['add', 'div', 'mul', 'pow', 'sub']


def prepare_encoders(names, indices, mean_patches, patch_type, model, Xs, ys):
    if patch_type == "mean":
        patch_info = from_names_to_encoder_info(names, indices, mean_patches)
        patch_info = [(name, info["headidx"], info["new_output"]) for name, info in patch_info.items()]
        edited_model = apply_patches_to_model(model, patch_info)
        encoders = get_encoders(edited_model, Xs, ys)
        return encoders

    elif patch_type == "resample":
        encoders = []

        for (mean_patch, X, y) in zip(mean_patches, Xs, ys):
            patch_info = from_names_to_encoder_info(names, indices, mean_patch)
            patch_info = [(name, info["headidx"], info["new_output"]) for name, info in patch_info.items()]
            edited_model = apply_patches_to_model(model, patch_info)
            encoders.append(get_encoders(edited_model, [X], [y])[0])
        return encoders
    else:
        raise NotImplementedError
