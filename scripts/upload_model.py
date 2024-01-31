from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/local/data/xuanming/models/output_llama2_13b_train_combined_lr_1e_5/checkpoint-1149")

tokenzier = AutoTokenizer.from_pretrained("/local/data/xuanming/models/output_llama2_13b_train_combined_lr_1e_5/checkpoint-1149")

model.push_to_hub("Columbia-NLP/llama-2-13b-hf-comb-ProLex")

tokenzier.push_to_hub("Columbia-NLP/llama-2-13b-hf-comb-ProLex")