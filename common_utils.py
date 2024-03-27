import dataclasses
import random
import inspect
from typing import Any, Union, List, Dict, Iterable, Callable, Sequence, Optional

import numpy as np
import torch
from torch import Tensor


__all__ = [
    "to", 
    "as_list",
    "fix_seed",
    "maybe_first_element",
    "make_batch_iterator"
]


def to(data: Any, *args, **kwargs):
    '''
    # adopted from https://github.com/Yura52/delu/blob/main/delu/_tensor_ops.py
    TODO
    '''
    def _to(x):
        return to(x, *args, **kwargs)

    if isinstance(data, Tensor):
        return data.to(*args, **kwargs)
    elif isinstance(data, (tuple, list, set)):
        return type(data)(_to(x) for x in data)
    elif isinstance(data, dict):
        return type(data)((k, _to(v)) for k, v in data.items())
    elif dataclasses.is_dataclass(data):
        return type(data)(**{k: _to(v) for k, v in vars(data).items()}) 
    # do nothing if provided value is not tensor or collection of tensors
    else:
        return data

def as_list(x: Union[str, List[str]]):
    if x is None:
        return x
    if isinstance(x, (list, tuple)):
        return x
    return [x]

def maybe_first_element(x):
    if isinstance(x, Sequence):
        x = x[0]
    return x

def extract_into_tensor(tensor_list: List[torch.Tensor], indices: Iterable[int], device=None):
    extracted_items = [maybe_first_element(tensor_list[i]) for i in indices]
    return torch.cat(extracted_items, dim=0).to(device)

def cast_if_float(x: torch.Tensor, dtype: Optional[torch.dtype] = None):
    if isinstance(x, (torch.FloatTensor, torch.HalfTensor)):
        return x.to(dtype)
    return x

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def filter_kwarg_dict(fn_or_method: Callable, kwarg_dict: Dict[str, Any]) -> Dict[str, Any]:
    fn_or_method_keys = inspect.signature(fn_or_method).parameters.keys()
    return {k: v for k, v in kwarg_dict.items() if k in fn_or_method_keys}

def collate_fn(
    input_args: List[Sequence[Any]], 
    input_kwargs: List[Dict[str, Any]], 
    targets: List[torch.Tensor],
    float_dtype: Optional[torch.dtype] = None
):
    batch_input_args = []
    batch_input_kwargs = {}
    # 1) prepare input args
    tensor_arg_pos = []
    for j, inp_arg in enumerate(input_args[0]):
        if isinstance(inp_arg, torch.Tensor):
            tensor_arg_pos.append(j)
            batch_input_args.append([inp_arg])
        else:
            batch_input_args.append(inp_arg)
    for inp_args in input_args[1:]:
        for j in tensor_arg_pos:
            batch_input_args[j].append(inp_args[j])
    # postprocess
    for j in tensor_arg_pos:
        batch_input_args[j] = cast_if_float(torch.cat(batch_input_args[j], dim=0), float_dtype)
    # 2) prepare input kwargs
    tensor_kwarg_keys = []
    for j, (k, v) in enumerate(input_kwargs[0].items()):
        if isinstance(v, torch.Tensor):
            tensor_kwarg_keys.append(k)
            batch_input_kwargs[k] = [v]
        else:
            batch_input_kwargs[k] = v
    for inp_kwargs in input_kwargs[1:]:
        for k in tensor_kwarg_keys:
            batch_input_kwargs[k].append(inp_kwargs[k])
    # postprocess
    for k in tensor_kwarg_keys:
        batch_input_kwargs[k] = torch.cat(batch_input_kwargs[k], dim=0)
    # 3) prepare targets
    batch_targets = cast_if_float(torch.cat(targets, dim=0), float_dtype)
    return batch_input_args, batch_input_kwargs, batch_targets


def make_batch_iterator(
    input_args: List[Sequence[Any]], 
    input_kwargs: List[Dict[str, Any]], 
    targets: List[torch.Tensor],
    batch_size: int,
    float_dtype: Optional[torch.dtype] = None
):
    all_batch_indices = []
    dataset_size = len(input_args)
    while True:
        if len(all_batch_indices) == 0:
            all_batch_indices = list(torch.randperm(dataset_size).chunk(dataset_size // batch_size))
        batch_indices = all_batch_indices.pop(0)
        yield collate_fn(
            [input_args[i] for i in batch_indices],
            [input_kwargs[i] for i in batch_indices],
            [targets[i] for i in batch_indices],
            float_dtype
        )
