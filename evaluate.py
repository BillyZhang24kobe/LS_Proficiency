from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
import pandas as pd
from dataset import *
import ast
import spacy
import os

# global variables
nlp = spacy.load("en_core_web_sm")

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


def format_test_prompt(target_word, sentence):
    return """You are about to perform a lexical substitution task, considering the proficiency level of the substitute compared to the target word in a sentence. The task is to generate a set of candidate substitutes seperated by commas for a target word in a given sentence. The target word is highlighted in the sentence, encompassed by two double asterisks. The candidate substitutes should be: \n a) common collocations or expressions in actual English use, \n b) grammatically correct, \n c) have an equal or higher language proficiency level compared to the target word. Target word: {} \n Sentence: {} \n Substitutes:""".format(target_word, sentence)


class Evaluator(object):
    def __init__(self, args, model, tokenizer, eval_dataset, metrics='hard', print_results=False):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.metrics = metrics
        self.print_results = print_results

    def calculate_metrics(self, labels, preds):
        """ Calculate the precision, recall, and F1 score. """
        precision = 0
        recall = 0
        f1 = 0

        for label, pred in zip(labels, preds):
            try:
                precision += len(set(label).intersection(set(pred))) / len(set(pred))
            except Exception as e:
                print(e)
                print("Pred: ", pred)
                print("Label: ", label)
                precision += 0
            recall += len(set(label).intersection(set(pred))) / len(set(label))
            f1 += 2 * (len(set(label).intersection(set(pred))) / (len(set(pred)) + len(set(label))))

        precision /= len(labels)
        recall /= len(labels)
        f1 /= len(labels)

        return precision, recall, f1
    
    def print_prediction_results(self, preds):
        """ Print the prediction results. """
        eval_df = pd.read_csv(self.eval_dataset, index_col=False)
        
        # store the predicted substitutes in a txt file
        with open("predicted_substitutes.txt", "w") as f:
            for i, row in eval_df.iterrows():
                target_word = row['target_words']
                sentence = row['Sentences']
                f.write("Target word: " + target_word + "\n")
                f.write("Sentence: " + sentence + "\n")
                f.write("Predicted substitutes: " + str(preds[i]) + "\n")
                f.write("Gold substitutes: " + str(ast.literal_eval(row["substitutes"])) + "\n")
                f.write("--------------------------------------------------" + "\n")
                f.write("\n")

    def evaluate(self):
        """ Evaluate the model on the given dataset. """

        eval_df = pd.read_csv(self.eval_dataset, index_col=False)

        model_preds = []
        gold_labels = []

        # eval mode
        self.model.eval()

        # no gradient calculation
        with torch.no_grad():
            for _, row in eval_df.iterrows():
                target_word = row['target_words']
                sentence = row['Sentences']
                system_input = format_test_prompt(target_word, sentence)

                input_ids = tokenizer.encode(system_input, return_tensors='pt', add_special_tokens=True)
                input_ids = input_ids.cuda()  # TODO: change to a device
                # print("System input length: ", int(input_ids.ne(self.tokenizer.pad_token_id).sum()))

                # Generate the candidates.
                generated_ids = self.model.generate(
                    input_ids,
                    max_length=self.tokenizer.model_max_length,
                    temperature=0.2)
                
                # Decode the candidates.
                generated_texts = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True)
                
                # print(generated_texts)
                try:
                    pred = generated_texts[0].split("Substitutes: ")[1].split(", ")
                except Exception as e:
                    print(e)
                    print("Generated text: ", generated_texts)
                    pred = []
                model_preds.append(pred)
                gold_labels.append(ast.literal_eval(row["substitutes"]))

        # print the results if print_results is True
        if self.print_results:
            self.print_prediction_results(model_preds)

        # Compute precision, recall, and F1
        if self.metrics == 'soft':
            # get the lemma of the predicted substitutes and gold substitutes
            model_preds_lemma = []
            gold_labels_lemma = []
            
            for pred in model_preds:
                pred_lemma = []
                for p in pred:
                    doc = nlp(p)
                    pred_lemma.append(doc[0].lemma_)
                model_preds_lemma.append(pred_lemma)

            for gold in gold_labels:
                gold_lemma = []
                for g in gold:
                    doc = nlp(g)
                    gold_lemma.append(doc[0].lemma_)
                gold_labels_lemma.append(gold_lemma)


            precision, recall, f1 = self.calculate_metrics(gold_labels_lemma, model_preds_lemma)

            return precision, recall, f1
        
        elif self.metrics == 'hard':
            return  self.calculate_metrics(gold_labels, model_preds)

    def predict_single_turn(self, 
                             inputs: Tuple[str, str]):
        """ Predict substitutes given a Tuple of target word and sentence. """
        print("Predicting substitutes for target word: ", inputs[0])
        target_word, sentence = inputs
        system_input = format_test_prompt(target_word, sentence)

        # for input_text, target_text in zip(inputs, targets):

        input_ids = tokenizer.encode(system_input, return_tensors='pt', add_special_tokens=True)
        input_ids = input_ids.cuda()
        print("System input length: ", int(input_ids.ne(self.tokenizer.pad_token_id).sum()))

        # Generate the candidates.
        # eval mode
        self.model.eval()
        # no gradient calculation
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_length=self.tokenizer.model_max_length,
                temperature=0.2,)
            
        # Decode the candidates.
        generated_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)
        
        return generated_texts[0]
    

if __name__ == "__main__":

    # load model
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
    config.use_cache = True

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    device = torch.device('cuda:0')  
    model.to(device)
    print(f"Model moved to: {torch.cuda.current_device()}")
    model = model.bfloat16()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side='left',
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    # predict on single input
    # target_word = "obligatory"
    # sentence = "Even though it was an **obligatory** experience, I could take part in a community program"
    # inputs = (target_word, sentence)

    # evaluator = Evaluator(training_args, model, tokenizer, None)
    # generated_texts = evaluator.predict_single_turn(inputs)
    # print(generated_texts)

    # load evaluation data
    # data_module = make_test_data_module(tokenizer, data_args)

    # evaluate
    print("Evaluating...")
    evaluator = Evaluator(training_args, model, tokenizer, data_args.data_path, metrics='hard', print_results=True)
    metrics = evaluator.evaluate()
    print("Precision: ", metrics[0])
    print("Recall: ", metrics[1])
    print("F1: ", metrics[2])