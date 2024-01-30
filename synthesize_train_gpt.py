import pandas as pd
import os
from utils import *
from tqdm import tqdm
import random
from openai import OpenAI

import api_secrets
os.environ['OPENAI_API_KEY'] = api_secrets.openai_api_key

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    organization=api_secrets.openai_org
)

# Define the items and their probabilities
items = {
    'noun': 25,
    'verb': 25,
    'adjective': 25,
    'adverb': 25
}

# Create a list where each item is repeated according to its probability
selection_pool = [item for item, count in items.items() for _ in range(count)]


# Function to randomly select an item based on the defined probabilities
def select_item():
    return random.choice(selection_pool)

# select target words and generate substitutions based on GPT-4
def formulate_prompt(query_pos, original_sentence):
    return [
        {"role": "system", "content": """You are about to synthesize data for a lexical substitution task, considering the proficiency level of the substitute compared to the target word in a sentence. Concretely, for each data point, I will first give you an original sentence. Then, you should follow the following steps to create a complete data point: 1) Based on the queried Part of Speech tag, select as unique as possible a content word as the target word to be substituted from the sentence (Content words include nouns, verbs, adjectives and adverbs). Do only select single word as the target; 3) generate up to five candidate substitutes (separated by commas) for the selected content word from Step 2; 4) final candidate substitutes after excluding candidates from Step 3 that are not common expressions in actual English use. \n You should make sure each of the generated substitutes follows exactly the following characteristics:  a) does NOT change the meaning and semantics of the sentence, and  b) is a common collocation or expression in actual English use (appears at least five times in Corpus of Contemporary American English), and c) in general, matches at least one connotation of the target word in the context sentence, and d) is grammatically correct, and e) has an equal or higher language proficiency level compared to the target word. Please use CEFR (Common European Framework of Reference for Languages) standard to describe the language proficiency of a word. The specification of CEFR levels (from the least proficient to the most proficient) is defined as follows: A1 (beginner), A2 (Elementary), B1 (Intermediate), B2 (Upper Intermediate), C1 (Advanced), C2 (Proficient). \n \n Here are some examples (Tagged sentence denotes sentence where the target word is surrounded by two double asterisks). Do not change the original sentence: \n Query Part of Speech tag: adverb \n Original Sentence: Students can learn to study independently from understanding ideas and concepts. \n Tagged Sentence: Students can learn to study **independently** from understanding ideas and concepts. \n Target word: independently (B2 - Upper Intermediate) \n Candidate Substitutes: autonomously (C2 - Proficient), individually (C1 - Advanced), solo (B2 - Upper Intermediate) \n Final Substitutes: autonomously, individually, solo \n \n Query Part of Speech tag: adjective \n Original Sentence:  It is because of this that various kinds of people with special knowledge can complement each other. \n Tagged Sentence: It is because of this that various kinds of people with **special** knowledge can complement each other. \n Target word: special (AI - beginner) \n Candidate Substitutes: specific (A2 - Elementary), distinctive (C1 - Advanced), exclusive (B2 - Upper Intermediate), unique (B2 - Upper Intermediate), particular (A2 - Elementary) \n Final Substitutes: specific, distinctive, unique, particular \n \n  Query Part of Speech tag: noun \n Original Sentence: At the start of a life a person doesn't have success yet, only during life does your action make your success. \n Tagged Sentence: At the start of a life a person doesn't have success yet, only during life does your **action** make your success. \n Target word: action (A1 - beginner) \n Candidate Substitutes: behavior (A2 - Elementary), conduct (B2 - Upper Intermediate), operation (B1 - Intermediate), undertaking (C1 - Advanced), activity (A1 - beginner) \n Final Substitutes: behavior, conduct, activity \n \n Query Part of Speech tag: verb \n Original Sentence: It has no arguments to support it and is terribly broad. \n Tagged Sentence: It has no arguments to **support** it and is terribly broad. \n Target word: support (A2 - Elementary) \n Candidate Substitutes: back (B2 - Upper Intermediate), substantiate (C2 - Proficient), uphold (C1 - Advanced), justify (B2 - Upper Intermediate) \n Final Substitutes: back, substantiate, justify \n \n Query Part of Speech tag: adverb \n Original Sentence: many old people have diseases that rob them of their health and make them unable. \n -> POS does not exist in the sentence. \n \n Please note that if there are no words that correspond to the queried part of speech tag in the original sentence, simply generate "POS does not exist in the sentence".  Do only select single word as the target. \n \n Now, please generate:"""},
        {"role": "user", "content": "Query Part of Speech tag: {} \n Original sentence: {}".format(query_pos, original_sentence)}
    ]


def generate_target_words_and_substitutes(data_df):
    '''
    Generate target words and their substitutes based on GPT-4
    '''
    original_sentences = []
    corrected_sentences = []
    tagged_sentences = []
    target_words = []
    pos = []
    candidate_substitutes = []
    final_substitutes = []
    proficiency_levels = []

    for i in tqdm(range(len(data_df))):
        row = data_df.iloc[i]

        # randomly select a POS tag
        query_pos = select_item()
        # print(query_pos)
        corrected_sentence = row['corrected_sentences']
        print(corrected_sentence)

        # formulate the prompt
        prompt = formulate_prompt(query_pos, corrected_sentence)

        proficiency_levels.append(row['essay_proficiency'])
        pos.append(query_pos)
        original_sentences.append(row['original_sentences'])
        corrected_sentences.append(corrected_sentence)
        
        # initialize all items with empty string
        tagged_sentence_toadd = ''
        target_word_toadd = ''
        candidate_substitutes_toadd = ''
        final_substitutes_toadd = ''
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=prompt
            )
            output = response.choices[0].message.content
            
            if 'POS does not exist in the sentence' in output:
                print('POS does not exist in the sentence. The skipped sentence is {}'.format(corrected_sentence)) 
                tagged_sentences.append(tagged_sentence_toadd)
                target_words.append(target_word_toadd)
                candidate_substitutes.append(candidate_substitutes_toadd)
                final_substitutes.append(final_substitutes_toadd)
                continue

            items = output.split('\n')
            for item in items:
                if 'Tagged Sentence' in item:
                    tagged_sentence_toadd = item.split('Tagged Sentence: ')[1].strip()
                    # tagged_sentences.append(tagged_sentence)
                if 'Target word' in item:
                    target_word_toadd = item.split('Target word: ')[1].strip()
                    # target_words.append(target_word)
                if 'Candidate Substitutes' in item:
                    candidate_substitutes_toadd = item.split('Candidate Substitutes: ')[1].strip()
                    # candidate_substitutes.append(candidate_substitutes)
                if 'Final Substitutes' in item:
                    final_substitutes_toadd = item.split('Final Substitutes: ')[1].strip()
                    # final_substitutes.append(final_substitutes)

        except Exception as e:
            print(e)
        
        tagged_sentences.append(tagged_sentence_toadd)
        target_words.append(target_word_toadd)
        candidate_substitutes.append(candidate_substitutes_toadd)
        final_substitutes.append(final_substitutes_toadd)
    
    return target_words, pos, original_sentences, corrected_sentences, tagged_sentences, candidate_substitutes, final_substitutes, proficiency_levels


def get_candidate_substitutes_gpt4(data_df):
    # In-context learning with GPT-4 to generate candidate substitutes for each target word
    candidates = []

    for idx, row in data_df.iterrows():
        target_word = row['tagged_target_words']
        tagged_sentence = row['tagged_sentences']
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=formulate_prompt(tagged_sentence, target_word)
            )
            output = response.choices[0].message.content
            candidates.append(output)
            print(output)
        except Exception as e:
            print(e)
            candidates.append('')

    data_df['candidate_substitutions'] = candidates
    data_df.to_csv('../data/targetLemmas_low_medium_all_clean_candidates.csv', index=False)


selected_sents_df = pd.read_csv('data/raw/toefl_1500.csv', index_col=False)
target_words, pos, original_sentences, corrected_sentences, tagged_sentences, candidate_substitutes, final_substitutes, proficiency_levels = generate_target_words_and_substitutes(selected_sents_df)

# store the generated data into a dataframe
generated_df = pd.DataFrame({
    'target_words': target_words,
    'query_pos': pos,
    'original_sentences': original_sentences,
    'corrected_sentences': corrected_sentences,
    'tagged_sentences': tagged_sentences,
    'candidate_substitutes': candidate_substitutes,
    'final_substitutes': final_substitutes,
    'proficiency_levels': proficiency_levels
})

# drop rows with empty values
generated_df = generated_df.dropna()

# create a new df with target words, substitutes, query_pos, original_sentences, corrected_sentences, tagged_sentences, proficiency_levels
target_words = []
query_pos = []
original_sentences = []
corrected_sentences = []
tagged_sentences = []
substitutes = []
proficiency_levels = []
for idx, row in generated_df.iterrows():
    t_word = row['target_words'].split('(')[0].strip()
    subs = row['final_substitutes'].split(',')
    for i in range(len(subs)):
        subs[i] = subs[i].strip()
    # convert the substitutes string to list
    target_words.append(t_word)
    query_pos.append(row['query_pos'])
    original_sentences.append(row['original_sentences'])
    corrected_sentences.append(row['corrected_sentences'])
    tagged_sentences.append(row['tagged_sentences'])
    substitutes.append(subs)
    proficiency_levels.append(row['proficiency_levels'])

# create a new df for all_df
data = {
    'target_words': target_words,
    'substitutes': substitutes,
    'query_pos': query_pos,
    'original_sentences': original_sentences,
    'corrected_sentences': corrected_sentences,
    'tagged_sentences': tagged_sentences,
    'proficiency_levels': proficiency_levels
}

df = pd.DataFrame(data)
df.to_csv('data/train/synthetic_gpt4.csv', index=False)

