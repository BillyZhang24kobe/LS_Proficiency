from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

import pandas as pd
import ast
from torch.nn.utils.rnn import pad_sequence

from train import rank0_print, format_input_prompt, format_target_prompt


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    is_test=False,
) -> Dict:

    inputs = []
    targets = []
    for i, source in enumerate(sources):  # source is a tuple: (target_words, tagged_sentences, substitutes)
        target_word, tagged_sentence, substitutes = source
        substitutes = ', '.join(substitutes)
        inputs.append(format_input_prompt(target_word, tagged_sentence))
        targets.append(format_target_prompt(substitutes))

    # Apply prompt templates
    all_input_ids = []
    all_labels = []
    all_attention_mask = []

    for input_text, target_text in zip(inputs, targets):

        input_ids = tokenizer.encode(input_text, return_tensors='pt', add_special_tokens=True)[0]
        target_ids = tokenizer.encode(target_text, return_tensors='pt', add_special_tokens=False)[0]

        input_label_ids = torch.ones_like(input_ids) * -100
        target_label_ids = target_ids.clone()
        # mask out the target prompt words
        target_label_ids[:5] = -100

        final_input_ids = torch.cat([input_ids, target_ids], dim=-1)
        final_label_ids = torch.cat([input_label_ids, target_label_ids], dim=-1)

        all_input_ids.append(final_input_ids)
        all_labels.append(final_label_ids)
        all_attention_mask.append(torch.ones_like(final_input_ids))

    final_input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    print("input ids shape: ", final_input_ids.shape)
    final_labels = pad_sequence(all_labels, batch_first=True, padding_value=-100)
    final_attention_mask = pad_sequence(all_attention_mask, batch_first=True, padding_value=0).bool()

    return dict(
        input_ids=final_input_ids,
        labels=final_labels,
        attention_mask=final_attention_mask,
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [(t_word, t_sentence, subs) for t_word, t_sentence, subs in zip(raw_data['target_words'], raw_data['tagged_sentences'], raw_data['substitutes'].apply(lambda x : ast.literal_eval(x)))]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )
    

class TestDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(TestDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [(t_word, t_sentence, subs) for t_word, t_sentence, subs in zip(raw_data['target_words'], raw_data['Sentences'], raw_data['substitutes'].apply(lambda x : ast.literal_eval(x)))]
        data_dict = preprocess(sources, tokenizer, is_test=True)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    # train_json = json.load(open(data_args.data_path, "r"))
    train_csv = pd.read_csv(data_args.data_path, index_col=False)
    train_dataset = dataset_cls(train_csv, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def make_test_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (TestDataset)
    rank0_print("Loading data...")

    # train_json = json.load(open(data_args.data_path, "r"))
    test_csv = pd.read_csv(data_args.data_path, index_col=False)
    test_dataset = dataset_cls(test_csv, tokenizer=tokenizer)

    return dict(test_dataset=test_dataset)