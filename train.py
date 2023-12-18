# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

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

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# system_prompt = """You are about to perform a lexical substitution task, considering the proficiency level of the substitute compared to the target word in a sentence. The task is to generate a set of candidate substitutes seperated by commas for a target word in a given sentence. The target word is highlighted in the sentence, encompassed by two double asterisks. The candidate substitutes should be: \n a) common collocations or expressions in actual English use, \n b) grammatically correct, \n c) have an equal or higher language proficiency level compared to the target word."""

def format_input_prompt(target_word, sentence):
    return """You are about to perform a lexical substitution task, considering the proficiency level of the substitute compared to the target word in a sentence. The task is to generate a set of candidate substitutes seperated by commas for a target word in a given sentence. The target word is highlighted in the sentence, encompassed by two double asterisks. The candidate substitutes should be: \n a) common collocations or expressions in actual English use, \n b) grammatically correct, \n c) have an equal or higher language proficiency level compared to the target word. Target word: {} \n Sentence: {}""".format(target_word, sentence)

def format_target_prompt(substitutes):
    return "Substitutes: {}</s>".format(substitutes)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # sources = [(rows['target_words'], rows['tagged_sentences', rows['substitutes']]) for _, rows in raw_data.iterrows()]  ->  a list of triples
    # conv = get_conversation_template("vicuna")
    # roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

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


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = model.bfloat16()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)


if __name__ == "__main__":
    train()
