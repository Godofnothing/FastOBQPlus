import os
import random

import torch
from datasets import load_dataset
from tqdm import trange

from dist_utils import print_on_main
from common_utils import fix_seed


def get_red_pajama(num_sequences, sequence_length, tokenizer, split="train"):
    print_on_main("Loading red_pajama from togethercomputer/RedPajama-Data-1T-Sample")
    assert split == "train", "Only train set is supported in RedPajama"
    train_dataset_raw = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    train_loader = []
    for _ in trange(num_sequences, desc="Making red_pajama calibration set", leave=False):
        while True:
            i = random.randint(0, len(train_dataset_raw) - 1)
            sample = tokenizer(train_dataset_raw[i]["text"], return_tensors="pt").input_ids
            if sample.shape[1] > sequence_length:
                break
        i = random.randint(0, max(0, sample.shape[1] - sequence_length - 1))
        j = i + sequence_length
        sample = sample[:, i:j]
        train_loader.append(sample)
    return train_loader


def get_wikitext2(num_sequences, sequence_length, tokenizer, split="train"):
    if split == "train":
        train_dataset_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        train_dataset_tok = tokenizer("\n\n".join(train_dataset_raw["text"]), return_tensors="pt").input_ids
        train_loader = []
        for _ in range(num_sequences):
            i = random.randint(0, train_dataset_tok.shape[1] - sequence_length - 1)
            j = i + sequence_length
            sample = train_dataset_tok[:, i:j]
            train_loader.append(sample)
        return train_loader
    else:
        test_dataset_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        test_dataset_tok = tokenizer("\n\n".join(test_dataset_raw["text"]), return_tensors="pt").input_ids
        num_test_sequences = test_dataset_tok.numel() // sequence_length
        test_loader = []
        for i in range(num_test_sequences):
            test_loader.append(test_dataset_tok[:, i * sequence_length: (i + 1) * sequence_length])
        return test_loader


def get_ptb(num_sequences, sequence_length, tokenizer, split="train"):
    if split == "train":
        train_dataset_raw = load_dataset("ptb_text_only", "penn_treebank", split="train")
        train_dataset_tok = tokenizer("\n\n".join(train_dataset_raw["sentence"]), return_tensors="pt")
        train_loader = []
        for _ in range(num_sequences):
            i = random.randint(0, train_dataset_tok.shape[1] - sequence_length - 1)
            j = i + sequence_length
            sample = train_dataset_tok[:, i:j]
            train_loader.append(sample)
        return train_loader
    else:
        valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")
        test_dataset_tok = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt").input_ids
        num_test_sequences = len(test_dataset_tok)
        test_loader = []
        for i in range(num_test_sequences):
            test_loader.append(test_dataset_tok[:, i * sequence_length: (i + 1) * sequence_length])
        return test_loader


def get_c4(num_sequences, sequence_length, tokenizer, split="train"):
    if split == "train":
        train_dataset_raw = load_dataset(
            "allenai/c4",
            "default",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        train_loader = []
        for _ in range(num_sequences):
            while True:
                i = random.randint(0, len(train_dataset_raw) - 1)
                sample = tokenizer(train_dataset_raw[i]["text"], return_tensors="pt").input_ids
                if sample.shape[1] >= sequence_length:
                    break
            i = random.randint(0, max(0, sample.shape[1] - sequence_length - 1))
            j = i + sequence_length
            sample = sample[:, i:j]
            train_loader.append(sample)
        return train_loader

    else:
        test_dataset_raw = load_dataset(
            "allenai/c4",
            "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        # fix seed to make evaluation determistic
        random.seed(0)
        test_loader = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(test_dataset_raw) - 1)
                sample = tokenizer(test_dataset_raw[i]["text"], return_tensors="pt").input_ids
                if sample.shape[1] >= sequence_length:
                    break
            i = random.randint(0, max(0, sample.shape[1] - sequence_length - 1))
            j = i + sequence_length
            test_loader.append(sample[:, i:j])
        return test_loader


def get_ptb_new(num_sequences, sequence_length, tokenizer, split="train"):
    if split == "train":
        train_dataset_raw = load_dataset("ptb_text_only", "penn_treebank", split="train")
        train_dataset_tok = tokenizer("\n\n".join(train_dataset_raw["sentence"]), return_tensors="pt")
        train_loader = []
        for _ in range(num_sequences):
            i = random.randint(0, train_dataset_tok.shape[1] - sequence_length - 1)
            j = i + sequence_length
            sample = train_dataset_tok[:, i:j]
            train_loader.append(sample)
        return train_loader
    else:
        valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")
        test_dataset_tok = tokenizer(" ".join(valdata["sentence"]), return_tensors="pt").input_ids
        num_test_sequences = len(test_dataset_tok)
        test_loader = []
        for i in range(num_test_sequences):
            test_loader.append(test_dataset_tok[:, i * sequence_length: (i + 1) * sequence_length])
        return test_loader


def get_c4_new(num_sequences, sequence_length, tokenizer, split="train"):
    if split == "train":
        train_dataset_raw = load_dataset(
            "allenai/c4",
            "default",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            verification_mode=False
        )
        train_loader = []
        for _ in range(num_sequences):
            while True:
                i = random.randint(0, len(train_dataset_raw) - 1)
                sample = tokenizer(train_dataset_raw[i]["text"], return_tensors="pt").input_ids
                if sample.shape[1] >= sequence_length:
                    break
            i = random.randint(0, max(0, sample.shape[1] - sequence_length - 1))
            j = i + sequence_length
            sample = sample[:, i:j]
            train_loader.append(sample)
        return train_loader
    else:
        test_dataset_raw = load_dataset(
            "allenai/c4",
            "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        test_dataset_tok = tokenizer(" ".join(test_dataset_raw[:1100]["text"]), return_tensors="pt").input_ids
        num_test_sequences = min(256, test_dataset_tok.numel() // sequence_length)
        test_loader = []
        for i in range(num_test_sequences):
            test_loader.append(test_dataset_tok[:, i * sequence_length: (i + 1) * sequence_length])
        return test_loader


def get_data_loader(
    dataset_name_or_path: str = None, 
    num_sequences: int = 128, 
    seed: int = 0, 
    sequence_length: int = 2048, 
    split: str = "train", 
    tokenizer = None
):
    fix_seed(seed)
    if dataset_name_or_path is None:
        print_on_main("Not loading any dataset. (OK if you use no compression or methods like RTN.)")
        return None
    elif os.path.isfile(dataset_name_or_path):
        print_on_main(f"Loading dataset from {dataset_name_or_path}.")
        data_loader = torch.load(dataset_name_or_path)[:num_sequences]
    else:
        assert tokenizer is not None
        if dataset_name_or_path.lower() == "wikitext2":
            data_loader = get_wikitext2(num_sequences, sequence_length, tokenizer, split=split)
        elif dataset_name_or_path.lower() == "pajama":
            data_loader = get_red_pajama(num_sequences, sequence_length, tokenizer, split=split)
        elif dataset_name_or_path.lower() == "ptb":
            data_loader = get_ptb(num_sequences, sequence_length, tokenizer, split=split)
        elif dataset_name_or_path.lower() == "ptb_new":
            data_loader = get_ptb_new(num_sequences, sequence_length, tokenizer, split=split)
        elif dataset_name_or_path.lower() == "c4":
            data_loader = get_c4(num_sequences, sequence_length, tokenizer, split=split)
        elif dataset_name_or_path.lower() == "c4_new":
            data_loader = get_c4_new(num_sequences, sequence_length, tokenizer, split=split)
        else:
            raise ValueError(
                f"Failed to load data from {dataset_name_or_path}.",
                "Check dataset dataset_name_or_path or path or use one of [wikitext2, pajama, ptb, ptb_new, c4, c4_new]",
            )

    print_on_main(f"Loaded data from {dataset_name_or_path}; {len(data_loader)=} sequences")
    return data_loader
