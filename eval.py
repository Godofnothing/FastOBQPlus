from typing import List, Iterable, Union
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from common_utils import to
from model_utils import InputCollector, ForwardInterrupt


__all__ = ["eval_perplexity"]


@torch.no_grad()
def eval_perplexity(
    model: nn.Module, 
    data_loader: Iterable,
    block_modules: str = None,
    pre_block_modules: Union[str, List[str]] = None,
    post_block_modules: Union[str, List[str]] = None,
    device: str = 'cuda',
    cpu_offload: bool = False,
) -> float:
    # get blocks
    blocks = model.get_submodule(block_modules)
    # put first block in device
    blocks[0] = blocks[0].to(device)
    if cpu_offload:
        assert pre_block_modules is not None
        # load input embeddings or any other preprocessing step
        for module_name in pre_block_modules:
            module = model.get_submodule(module_name)
            module.to(device)

    if hasattr(model.config, 'use_cache'):
        use_cache = model.config.use_cache
        model.config.use_cache = False

    # Input preparation #
    blocks[0] = InputCollector(blocks[0])
    # TODO make namedtuple
    for (inp_args, inp_kwargs) in data_loader:
        try:
            model(
                *to(inp_args, device=device),
                **to(inp_kwargs, device=device),
            )
        except ForwardInterrupt:
            pass
    input_args = blocks[0].input_args
    input_kwargs = blocks[0].input_kwargs
    blocks[0] = blocks[0].module

    if cpu_offload:
        # offload input embeddings or any other preprocessing step
        for module_name in pre_block_modules:
            module = model.get_submodule(module_name)
            module.cpu()

    for i in trange(len(blocks), desc="Processing eval data"):
        block = blocks[i].to(device)

        for (inp_args, inp_kwargs) in zip(input_args, input_kwargs):
            out = block(*inp_args, **inp_kwargs)
            if isinstance(out, (list, tuple)):
                out = out[0]
            # change only first input argument
            inp_args[0].data = out

        blocks[i] = block.cpu()
        del block

    inputs = [inp_args[0] for inp_args in input_args]

    for module_name in post_block_modules:
        module = model.get_submodule(module_name)
        if cpu_offload:
            module = module.to(device)
        for inp in inputs:
            if cpu_offload:
                inp.data = inp.to(device)
            inp.data = module(inp)
            if cpu_offload:
                inp.data = inp.cpu()
        if cpu_offload:
            module = module.cpu()
        torch.cuda.empty_cache()

    logits = inputs
    num_sequences = len(logits)

    nlls = []
    for i, logits_batch in enumerate(logits):
        shift_logits = logits_batch[:, :-1, :].to(device).float()
        shift_labels = data_loader[i][1]['input_ids'][:, 1:].to(device)
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float()
        nlls.append(neg_log_likelihood)

    perplexity = torch.exp(torch.stack(nlls).sum() / num_sequences).item()

    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = use_cache

    return perplexity
