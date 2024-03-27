import argparse
import time

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

import dist_utils
import data_utils
from engine import Quantizer
from eval import eval_perplexity

def parse_args():
    parser = argparse.ArgumentParser(description="One-shot pruning with parallel OBC.")
    # Model params
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=True,
        help="The name or path to the model being quantized",
    )
    parser.add_argument(
        '--tokenizer_name',
        type=str,
        default=None,
        help="The name or path to the tokenizer. By default use model tokenizer.",
    )
    parser.add_argument(
        '--quantizable_modules',
        type=str,
        required=True,
        help="Regex for modules to quantize",
    )
    parser.add_argument(
        '--pre_block_modules',
        nargs="+",
        type=str,
        required=True,
        help="Names of modules before transformer blocks",
    )
    parser.add_argument(
        '--block_modules',
        type=str,
        required=True,
        help="Name of transformer modules",
    )
    parser.add_argument(
        '--post_block_modules',
        nargs="+",
        type=str,
        required=True,
        help="Names of modules after transformer blocks",
    )
    # Data params
    parser.add_argument(
        '--dataset_name_or_path',
        type=str,
        required=True,
        help="The name or dataset or path used for calibration.",
    )
    parser.add_argument(
        '--sequence_length',
        default=2048,
        type=int,
        help="Length of calibration sequences."
    )
    parser.add_argument(
        '--num_sequences',
        default=128,
        type=int,
        help="Number of calibration sequences."
    )
    # Quantization params
    parser.add_argument(
        '--bits',
        default=4,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 16],
        type=float
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=None,
        help="How many weight columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument(
        "--act_order",
        action="store_true",
        help="Whether to permute in activation order.",
    )
    parser.add_argument(
        "--sym", 
        action="store_true", 
        help="Whether to use symmetric quantization"
    )
    parser.add_argument(
        "--perchannel",
        action="store_true",
        help="fit a unique quantizer to each output dim",
    )
    parser.add_argument(
        "--rel_damp", 
        type=float, 
        default=1e-2
    )
    parser.add_argument(
        "--block_size", 
        type=int, 
        default=128
    )
    # Logging params
    parser.add_argument(
        '--log_wandb',
        default=False,
        action="store_true",
        help="Log to W&B"
    )
    # Finetuning params
    parser.add_argument(
        '--epochs',
        default=0,
        type=int,
        help="Number of finetuning epochs"
    )
    parser.add_argument(
        '--batch_size',
        default=1,
        type=int,
        help="Finetuning batch size"
    )
    parser.add_argument(
        '--lr',
        default=1e-4,
        type=float,
        help="Finetuning learning rate"
    )
    parser.add_argument(
        '--adam_beta1',
        default=0.9,
        type=float,
        help="Finetuning adam_beta1."
    )
    parser.add_argument(
        '--adam_beta2',
        default=0.999,
        type=float,
        help="Finetuning adam_beta2."
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Whether to use amp on block finetuning'
    )
    parser.add_argument(
        '--pop_keys',
        nargs="+",
        type=str,
        default=[],
        help='Keys to pop from inputs during block tuning'
    )
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument(
        '--seed',
        default=0,
        type=int,
        help="random seed."
    )
    parser.add_argument(
        '--low_cpu_mem_usage',
        action='store_true',
        help='whether to load model with the use of `low_cpu_mem_usage`'
    )
    parser.add_argument(
        "--cpu_offload_modules", 
        action="store_true",
        help="whether to offload modules to CPU."
    )
    parser.add_argument(
        "--cpu_offload_activations", 
        action="store_true",
        help="whether to offload activations to CPU."
    )
    parser.add_argument(
        "--new_eval", 
        action="store_true",
        help="whether to use new evaluation setup."
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="whether to log progress."
    )
    # save params
    parser.add_argument(
        "--save", 
        type=str,
        default=None,
        help="where to save sparse model."
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # Distributed init
    if dist.is_available():
        dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    # init device
    device = f"cuda:{rank}"
    dtype = getattr(torch, args.dtype)

    # init W&B logger
    if args.log_wandb and dist_utils.is_main():
        wandb.init(config=args)

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=args.low_cpu_mem_usage
    )
    if not args.cpu_offload_modules:
        model = model.to(device)
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path, use_fast=False
    )
    # Data
    train_loader = data_utils.get_data_loader(
        dataset_name_or_path=args.dataset_name_or_path, 
        num_sequences=args.num_sequences,
        seed=args.seed,
        sequence_length=args.sequence_length,
        split="train",
        tokenizer=tokenizer
    )
    # take slice (if running on multiple workers)
    if dist_utils.is_dist_available_and_initialized():
        num_seq_per_rank = len(train_loader) // world_size
        train_loader = train_loader[rank * num_seq_per_rank: (rank + 1) * num_seq_per_rank]
    train_loader = [([], {'input_ids': input_ids}) for input_ids in train_loader]
    dist.barrier()
    # quantizer
    if args.bits <= 8:
        quantizer = Quantizer(
            model, 
            train_loader, 
            quantizable_modules=args.quantizable_modules,
            pre_block_modules=args.pre_block_modules,
            block_modules=args.block_modules,
            obq_kwargs=dict(
                rel_damp=args.rel_damp,
                block_size=args.block_size,
                perchannel=args.perchannel, 
                group_size=args.group_size,
                sym=args.sym,
                act_order=args.act_order,
            ),
            device=device,
            cpu_offload_modules=args.cpu_offload_modules,
            cpu_offload_activations=args.cpu_offload_activations,
            finetune_kwargs=dict(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                adam_beta1=args.adam_beta1,
                adam_beta2=args.adam_beta2,
                amp=args.amp,
                pop_keys=args.pop_keys
            ),
            verbose=args.verbose
        )
        dist.barrier()
        t1 = time.perf_counter()
        quantizer.quantize(args.bits)
        t2 = time.perf_counter()
        dist_utils.print_on_main(f"Pruning took {(t2 - t1)} s.")
    else:
        dist_utils.print_on_main("No pruning.")

    # Evaluation is done only on main process
    if dist_utils.is_main():
        dist_utils.print_on_main('---Evaluation after pruning---')
        # evaluating only on main process
        if args.new_eval:
            eval_datasets = ['wikitext2', 'ptb_new', 'c4_new']
        else:
            eval_datasets = ['wikitext2', 'ptb', 'c4']
        
        eval_stats = {}
        for eval_dataset_name in eval_datasets:
            test_loader = data_utils.get_data_loader(
                dataset_name_or_path=eval_dataset_name, 
                num_sequences=args.num_sequences,
                seed=args.seed,
                sequence_length=args.sequence_length,
                split="test",
                tokenizer=tokenizer
            )
            test_loader = [([], {'input_ids': input_ids}) for input_ids in test_loader]
            ppl = eval_perplexity(
                model, 
                test_loader, 
                args.block_modules,
                args.pre_block_modules,
                args.post_block_modules,
                device,
                cpu_offload=args.cpu_offload_modules
            )
            dist_utils.print_on_main(f'Dataset: {eval_dataset_name}\nPerplexity: {ppl:.2f}')
            eval_stats[f'eval/{eval_dataset_name}'] = ppl
        if args.log_wandb and dist_utils.is_main():
            wandb.log(eval_stats)
    # waiting for other process to complete
    dist.barrier()
    if args.save and dist_utils.is_main():
        model.save_pretrained(args.save)
        tokenizer.save_pretrained(args.save)

if __name__ == "__main__":
    main()
