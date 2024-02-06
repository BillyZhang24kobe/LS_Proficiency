from datasets import load_dataset
# dev_dataset = load_dataset('csv', data_files='/home/billyzhang/lexical_substitution/LS_pro/LS_Proficiency/data/dev/ProLex_v1.0_dev.csv')
test_dataset = load_dataset('csv', data_files='/home/billyzhang/lexical_substitution/LS_pro/LS_Proficiency/data/test/ProLex_v1.0_test.csv')

# dev_dataset.push_to_hub("Columbia-NLP/ProLex")
test_dataset.push_to_hub("Columbia-NLP/ProLex")

