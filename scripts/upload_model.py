from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/local/data/xuanming/models/output_llama2_7b_train_lr_1e_5/checkpoint-4149")

tokenzier = AutoTokenizer.from_pretrained("/local/data/xuanming/models/output_llama2_7b_train_lr_1e_5/checkpoint-4149")

model.push_to_hub("Columbia-NLP/llama-2-7b-hf-syn-ProLex")

tokenzier.push_to_hub("Columbia-NLP/llama-2-7b-hf-syn-ProLex")