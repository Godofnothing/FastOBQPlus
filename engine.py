from typing import Iterable, Dict, List, Any, Optional
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import dist_utils
import model_utils
from common_utils import to, maybe_first_element, make_batch_iterator
from model_utils import InputCollector, ForwardInterrupt, LINEAR_LAYERS, select_layers
from quant_utils import QLinear

from fast_obq import FastOBQ


class Quantizer:

    def __init__(
        self, 
        model: nn.Module, 
        data_loader: Iterable,
        quantizable_modules: str,
        pre_block_modules: List[str],
        block_modules: str,
        obq_kwargs: Dict[str, Any] = {},
        device: Optional[torch.device] = None,
        cpu_offload_modules: bool = False,
        cpu_offload_activations: bool = False,
        finetune_kwargs: Dict[str, Any] = {},
        verbose: bool = False
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.quantizable_modules = quantizable_modules
        self.pre_block_modules = pre_block_modules
        self.block_modules = block_modules
        self.obq_kwargs = obq_kwargs
        self.device = device
        self.cpu_offload_modules = cpu_offload_modules
        self.cpu_offload_activations = cpu_offload_activations
        self.finetune_kwargs = finetune_kwargs
        self.verbose = verbose

    @torch.no_grad()
    def quantize(self, bits: int):
        device = self.device or next(self.model.parameters()).device
        # whether to finetune
        finetune = self.finetune_kwargs["epochs"] > 0
        # prepare pre blocks modules
        blocks = self._get_submodule(self.block_modules)
        pre_blocks = [self._get_submodule(module_name) for module_name in self.pre_block_modules]
        blocks[0] = blocks[0].to(device)
        for module in pre_blocks:
            module.to(device)
        # Cache
        if hasattr(self.model.config, 'use_cache'):
            use_cache = self.model.config.use_cache
            self.model.config.use_cache = False
        # Input preparation #
        blocks[0] = InputCollector(blocks[0], cpu_offload=self.cpu_offload_activations)
        # TODO make namedtuple
        for inp_args, inp_kwargs in self.data_loader:
            try:
                self.model(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            except ForwardInterrupt:
                pass
        input_args = blocks[0].input_args
        input_kwargs = blocks[0].input_kwargs
        blocks[0] = blocks[0].module

        if dist_utils.is_dist_available_and_initialized():
            dist.barrier()

        # offload pre_blocks
        if self.cpu_offload_modules:
            for module in pre_blocks:
                module.cpu()

        # Block pruning #
        for block_id, block in enumerate(blocks):
            # TODO change to logging
            if self.verbose:
                dist_utils.print_on_main(f"Processing {self.block_modules} {block_id}/{len(blocks)}.")
            block = block.to(device)
            # get layer prefix to select layers only within the block
            layer_prefix = f'{self.block_modules}.{block_id}.'
            layers = select_layers(self.model, layer_prefix, self.quantizable_modules, LINEAR_LAYERS)
            handles, hooks = self._prepare_hooks_and_handles(bits, layers)

            targets = []
            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
                if finetune:
                    out = maybe_first_element(out)
                    if self.cpu_offload_activations:
                        out = out.cpu()
                    targets.append(out)

            for _, h in hooks.items():
                h.remove()

            if dist_utils.is_dist_available_and_initialized():
                dist.barrier()

            self._quant_group(handles, bits)

            if finetune:
                self._finetune_block(block, input_args, input_kwargs, targets)

            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
                out = maybe_first_element(out)
                if self.cpu_offload_activations:
                    out = out.cpu()
                # change only first input argument
                if len(inp_args) > 0:
                    inp_args[0].data = out
                elif 'hidden_states' in inp_kwargs:
                    inp_kwargs['hidden_states'] = out
                else:
                    raise ValueError("Unsupported block input format.")

            if self.cpu_offload_modules:
                block = block.cpu()

            del handles
            del hooks
            torch.cuda.empty_cache()

        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = use_cache

    def _get_submodule(self, module_name: str):
        return self.model.get_submodule(module_name)

    def _prepare_hooks_and_handles(self, bits: int, layers: Dict[str, nn.Module]):
        handles = {}
        hooks = {}
        for layer_name, layer in layers.items():
            def update_handle_hook(name):
                def _hook(_, inp, out):
                    handles[name].update(inp[0])
                return _hook
            handles[layer_name] = self._create_handle(bits, layer)
            hooks[layer_name] = layer.register_forward_hook(update_handle_hook(layer_name))
        return handles, hooks

    def _create_handle(self, bits, layer):
        return FastOBQ(layer, bits=bits, **self.obq_kwargs)

    def _quant_group(self, handles: Dict[str, FastOBQ], bits: int):
        for handle_name, handle in handles.items():
            if self.verbose:
                dist_utils.print_on_main(f"Quantizing {handle_name}")
            qweight, scale, zero, perm = handle.quantize()
            qlayer = QLinear(
                qweight, 
                scale, 
                zero, 
                bias=handle.layer.bias, 
                perm=perm, 
                bits=8 if bits > 4 else 4
            )
            parent_name, child_name = handle_name.rsplit(".", 1)
            parent_module = self.model.get_submodule(parent_name)
            setattr(parent_module, child_name, qlayer)
            handle.reset()

    @torch.enable_grad()
    def _finetune_block(
        self, 
        block: nn.Module, 
        input_args: List[Any], 
        input_kwargs: Dict[str, Any], 
        targets: List[torch.Tensor]
    ):
        dist_utils.print_on_main("Finetuning")
        device = next(block.parameters()).device
        dtype = next(block.parameters()).dtype
        block.train()
        # cast to float32
        block.float()
        # init masks
        masks = [param.ne(0) for param in block.parameters()]
        # init DDP
        if dist_utils.is_dist_available_and_initialized():
            block = DDP(block, device_ids=[dist_utils.get_rank()])
        # init optimizer
        optimizer = torch.optim.Adam(
            block.parameters(),
            lr=self.finetune_kwargs["lr"],
            betas=(self.finetune_kwargs["adam_beta1"], self.finetune_kwargs["adam_beta2"]),
        )
        # init scaler
        scaler = torch.cuda.amp.GradScaler(enabled=self.finetune_kwargs["amp"])

        batch_size = self.finetune_kwargs["batch_size"]
        steps_per_epoch = len(input_args) // batch_size

        batch_iterator = make_batch_iterator(
            input_args, 
            input_kwargs, 
            targets, 
            batch_size,
            torch.float32
        )

        for epoch in trange(self.finetune_kwargs["epochs"], desc="Epoch", disable=not dist_utils.is_main()):
            epoch_loss = 0
            for step in trange(steps_per_epoch, desc="Step",  disable=not dist_utils.is_main(), leave=False):
                inp_args, inp_kwargs, target = next(batch_iterator)
                with torch.autocast(device_type='cuda', enabled=self.finetune_kwargs["amp"]):
                    # TODO dirty fix
                    if "cache_position" in inp_kwargs:
                        inp_kwargs.pop("cache_position")
                    out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
                    out = maybe_first_element(out)
                loss = F.mse_loss(out, target.to(device=device, dtype=torch.float32))
                # scaler and optimizer step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # mask params
                with torch.no_grad():
                    for param, mask in zip(block.parameters(), masks):
                        param.mul_(mask)
                epoch_loss += (loss.item() / steps_per_epoch)
            dist_utils.print_on_main(f"Train loss: {epoch_loss:.3e}")

        block = block.to(dtype)
        block.eval()
