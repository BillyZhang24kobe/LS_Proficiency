from datasets import load_dataset
dataset = load_dataset('csv', data_files='path/to/your/dataset.csv')

dataset.push_to_hub("your-dataset-name")