import pandas as pd
import os
from tqdm import tqdm
# OpenAI credentials
import api_secrets
from openai import OpenAI
os.environ['OPENAI_API_KEY'] = api_secrets.openai_api_key

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    organization=api_secrets.openai_org
)

def formulate_prompt(target_word, sentence):
    return [
        {"role": "system", "content": "You are about to perform a lexical substitution task, considering the proficiency level of the substitute compared to the target word in a sentence. The task is to generate a set of candidate substitutes separated by commas for a target word in a given sentence. The target word is highlighted in the sentence, encompassed by two double asterisks."},
        {"role": "user", "content": "Target word: {} \n Sentence: {} \n Substitutes:".format(target_word, sentence)},
    ]

def generate_pred_gpt(data_df, output_path, model_name='gpt-4'):
    # In-context learning with GPT-4 to generate candidate substitutes for each target word
    pred = []

    for idx in tqdm(range(len(data_df))):
        row = data_df.iloc[idx]
        target_word = row['target word']
        sentence = row['Sentence']
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=formulate_prompt(target_word, sentence)
            )
            output = response.choices[0].message.content
            pred.append(output)
            print(output)
        except Exception as e:
            print(e)
            pred.append('')


    # Save predictions in a new csv
    data = {'target word': data_df['target word'], 'Sentence': data_df['Sentence'], 'Substitutes': pred}

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    test_data = pd.read_csv('data/test/test_final_cefr.csv', index_col=False)

    # print('Generating predictions for GPT-4...')
    # generate_pred_gpt(test_data, 'outputs/gpt-4_test_final_cefr.csv', model_name='gpt-4')

    print('Generating predictions for ChatGPT...')
    generate_pred_gpt(test_data, 'outputs/gpt-3.5-turbo-1106_test_final_cefr.csv', model_name='gpt-3.5-turbo-1106')