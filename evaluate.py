from dataclasses import dataclass, field
import math
from typing import Optional, Tuple

import torch
import transformers

import pandas as pd
from dataset import *
import ast
import spacy
from tqdm import tqdm

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


# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     cache_dir: Optional[str] = field(default=None)
#     optim: str = field(default="adamw_torch")
#     model_max_length: int = field(
#         default=512,
#         metadata={
#             "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
#         },
#     )


def format_test_prompt(target_word, sentence):
    return """You are about to perform a lexical substitution task, considering the proficiency level of the substitute compared to the target word in a sentence. The task is to generate a set of candidate substitutes seperated by commas for a target word in a given sentence. The target word is highlighted in the sentence, encompassed by two double asterisks. The candidate substitutes should be: \n a) common collocations or expressions in actual English use, \n b) grammatically correct, \n c) have an equal or higher language proficiency level compared to the target word. Target word: {} \n Sentence: {} \n Substitutes:""".format(target_word, sentence)


class Evaluator(object):
    def __init__(self, args, model, eval_dataset, tokenizer=None, evaluate_all_metrics=False, metrics='hard', print_results=False, aspect='acc', topk=10):
        """ Initialize the Evaluator. 
        Args:
            args: TrainingArguments
            model: Pretrained model or model_name ('gpt-4', 'ChatGPT', 'para-ls', 'bert-ls')
            tokenizer: Pretrained tokenizer
            eval_dataset: Path to the evaluation dataset
            metrics: 'hard' or 'soft'
            print_results: True or False
            aspect: 'acc' or 'prof'
            topk: top k candidates to be considered as substitutes
        """
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.evaluate_all_metrics = evaluate_all_metrics
        self.eval_dataset = eval_dataset
        self.metrics = metrics
        self.print_results = print_results
        self.aspect = aspect
        self.topk = topk

    def calculate_metrics(self, labels, preds):
        """ Calculate the precision, recall, and F1 score. 
        Args:
            labels: gold labels -> list
            preds: predicted substitutes -> list
        Returns:
            precision: precision score
            recall: recall score
            f1: F1 score
        """

        # Verify if labels and preds have the same length
        if len(labels) != len(preds):
            raise ValueError("The length of labels and preds must be the same")
        
        numerator = 0
        p_denominator = 0
        r_denominator = 0

        for label, pred in zip(labels, preds):
            if len(label) == 0: continue
            pred = pred[:self.topk]
            num_acceptable = len(set(label).intersection(set(pred)))
            numerator += num_acceptable
            p_denominator += len(set(pred))
            r_denominator += min(self.topk, len(set(label)))

        precision = numerator / p_denominator
        recall = numerator / r_denominator
        f1 = (2 * precision * recall) / (precision + recall)

        return precision, recall, f1
    
    def print_prediction_results(self, preds):
        """ Print the prediction results. """
        eval_df = pd.read_csv(self.eval_dataset, index_col=False)
        
        print("Printing prediction results...")
        # store the predicted substitutes in a txt file
        with open(self.args.model_name_or_path+"-"+self.metrics+"-"+self.aspect+"-"+"predicted_substitutes.txt", "w") as f:
            for i, row in eval_df.iterrows():
                target_word = row['target word']
                sentence = row['Sentence']
                f.write("Target word: " + target_word + "\n")
                f.write("Sentence: " + sentence + "\n")
                if self.aspect == 'acc':
                    f.write("Gold substitutes: " + str(ast.literal_eval(row["acc_subs"])) + "\n")
                elif self.aspect == 'prof':
                    f.write("Gold substitutes: " + str(ast.literal_eval(row["prof_acc_subs"])) + "\n")
                f.write("Predicted substitutes: " + str(preds[i]) + "\n")
                f.write("--------------------------------------------------" + "\n")
                f.write("\n")

    def evaluate_soft_metrics(self, labels, preds):
        # get the lemma of the predicted substitutes and gold substitutes
            model_preds_lemma = []
            gold_labels_lemma = []
            
            for pred in preds:
                pred_lemma = []
                for p in pred:
                    doc = nlp(p)
                    pred_lemma.append(doc[0].lemma_)
                model_preds_lemma.append(pred_lemma)

            for gold in labels:
                gold_lemma = []
                for g in gold:
                    doc = nlp(g)
                    gold_lemma.append(doc[0].lemma_)
                gold_labels_lemma.append(gold_lemma)


            precision, recall, f1 = self.calculate_metrics(gold_labels_lemma, model_preds_lemma)

            return precision, recall, f1
    
    def evaluate_hard_metrics(self, labels, preds):
        return self.calculate_metrics(labels, preds)

    def get_gold_labels(self):
        """ Get the gold labels from the evaluation dataset. """
        eval_df = pd.read_csv(self.eval_dataset, index_col=False)
        gold_labels = []

        for i in tqdm(range(len(eval_df))):
            row = eval_df.iloc[i]
            if self.aspect == 'acc':
                gold_labels.append(ast.literal_eval(row["acc_subs"]))
            elif self.aspect == 'prof':
                gold_labels.append(ast.literal_eval(row["prof_acc_subs"]))
        
        return gold_labels

    def evaluate(self):
        """ Evaluate the model on the given dataset. """
        model_preds = []

        eval_df = pd.read_csv(self.eval_dataset, index_col=False)
        
        if self.model in ['gpt-4', 'gpt-4-32', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-1106-32', 'gpt-3.5-turbo', 'para-ls', 'bert-ls']:
            pred_df = pd.read_csv("outputs/"+self.model+'_'+self.eval_dataset.split('/')[-1], index_col=False)

            for i in tqdm(range(len(eval_df))):
                gold_row = eval_df.iloc[i]
                pred_row = pred_df.iloc[i]

                pred = pred_row['Substitutes'].split(", ")
                model_preds.append(pred)
        else:
            # eval mode
            self.model.eval()

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # no gradient calculation
            with torch.no_grad():
                for i in tqdm(range(len(eval_df))):
                    row = eval_df.iloc[i]
                    target_word = row['target word']
                    sentence = row['Sentence']
                    system_input = format_test_prompt(target_word, sentence)

                    input_ids = tokenizer.encode(system_input, return_tensors='pt', add_special_tokens=True)
                    input_ids = input_ids.cuda()

                    # Generate the candidates.
                    generated_ids = self.model.generate(
                        input_ids,
                        max_length=self.tokenizer.model_max_length,
                        temperature=0.2,
                        pad_token_id=self.tokenizer.pad_token_id)
                    
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
        
        # print the results if print_results is True
        if self.print_results:
            self.print_prediction_results(model_preds)

        if self.evaluate_all_metrics:
            # evaluate all metrics
            metrics_ = ["soft", "hard"]
            aspects_ = ["acc", "prof"]

            for metric in metrics_:
                for aspect in aspects_:
                    self.metrics = metric
                    self.aspect = aspect
                    gold_labels = self.get_gold_labels()
                    assert len(gold_labels) == len(model_preds)
                    if metric == 'soft':
                        precision, recall, f1 = self.evaluate_soft_metrics(gold_labels, model_preds)
                    elif metric == 'hard':
                        precision, recall, f1 = self.evaluate_hard_metrics(gold_labels, model_preds)
                    print("Metric: ", metric, "Aspect: ", aspect)
                    print("Precision: ", precision)
                    print("Recall: ", recall)
                    print("F1: ", f1)
                    print("--------------------------------------------------")

        else:
        # Compute precision, recall, and F1 seperately for each metrics
            gold_labels = self.get_gold_labels()
            if self.metrics == 'soft':
                return self.evaluate_soft_metrics(gold_labels, model_preds)
            
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
                temperature=0.2)
            
        # Decode the candidates.
        generated_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)
        
        return generated_texts[0]
    


if __name__ == "__main__":

    # load model
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments)
    )
    model_args, data_args = parser.parse_args_into_dataclasses()

    if model_args.model_name_or_path in ['gpt-4', 'gpt-4-32', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-1106-32', 'gpt-3.5-turbo', 'para-ls', 'bert-ls']:
        # evaluate
        print("Evaluating predictions from ", model_args.model_name_or_path)
        evaluator = Evaluator(model_args, model_args.model_name_or_path, data_args.data_path, evaluate_all_metrics=True, print_results=True)
        metrics = evaluator.evaluate()
    else:

        # Set RoPE scaling factor
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
        )

        # Load model and tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            trust_remote_code=model_args.trust_remote_code,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            padding_side='left',
            use_fast=False,
            trust_remote_code=model_args.trust_remote_code,
        )

        device = torch.device('cuda:0')  
        model.to(device)
        model = model.bfloat16()

        # evaluate
        print("Evaluating...")
        evaluator = Evaluator(model_args, model, data_args.data_path, tokenizer, evaluate_all_metrics=True, print_results=True)
        metrics = evaluator.evaluate()


        # predict on single input
        # target_word = "obligatory"
        # sentence = "Even though it was an **obligatory** experience, I could take part in a community program"
        # inputs = (target_word, sentence)

        # evaluator = Evaluator(training_args, model, tokenizer, None)
        # generated_texts = evaluator.predict_single_turn(inputs)
        # print(generated_texts)