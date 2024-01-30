import pandas as pd
import os
from collections import Counter
import spacy
import re
# import gec_feedback
from utils import *
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import random
from collections import Counter

from nltk.corpus import words

os.environ['GPU_DEVICE'] = '7'

# OpenAI credentials - TODO: hide this to a seperate file before pushing to github
from openai import OpenAI

import api_secrets
os.environ['OPENAI_API_KEY'] = api_secrets.openai_api_key

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    organization=api_secrets.openai_org
)

test_df = pd.read_csv('../data/targetLemmas_low_medium_all_clean.csv')

# Define the items and their probabilities
items = {
    'noun': 25,
    'verb': 25,
    'adjective': 25,
    'adverb': 25
}

# Create a list where each item is repeated according to its probability
selection_pool = [item for item, count in items.items() for _ in range(count)]

def associate_words_POS_with_sentences(doc):
    '''
    Associate each (word, pos) with the sentences it come from. The word to be considered is content words (i.e. NOUN, ADJ, ADV, VERB)
    E.g. (word1, pos1): {sentences}
    '''
    wordPOS_sentence_association = {}
    
    for sentence in doc.sents:
        for word in sentence:
            if word.pos_ not in ['NOUN', 'ADJ', 'ADV', 'VERB']: continue  # only keep the content words
            if not word.is_stop and not word.is_punct and not '\n' in word.text and not '\t' in word.text and not ' ' in word.text:
                if (word.text, word.pos_) in wordPOS_sentence_association:
                    wordPOS_sentence_association[(word.text, word.pos_)].add(sentence.text.strip().replace('\n', ''))
                else:
                    wordPOS_sentence_association[(word.text, word.pos_)] = {sentence.text.strip().replace('\n', '')}
    
    return wordPOS_sentence_association

def associate_lemma_POS_with_sentences(doc):
    '''
    Associate each (lemma, pos) with the sentences it come from. The word to be considered is content words (i.e. NOUN, ADJ, ADV, VERB)
    E.g. (lemma1, pos1): {sentences}
    '''
    lemmaPOS_sentence_association = {}
    
    for sentence in doc.sents:
        for word in sentence:
            if word.pos_ not in ['NOUN', 'ADJ', 'ADV', 'VERB']: continue  # only keep the content words
            if not word.is_stop and not word.is_punct and not '\n' in word.text and not '\t' in word.text and not ' ' in word.text:
                if (word.lemma_, word.pos_) in lemmaPOS_sentence_association:
                    lemmaPOS_sentence_association[(word.lemma_, word.pos_)].add(sentence.text.strip().replace('\n', ''))
                else:
                    lemmaPOS_sentence_association[(word.lemma_, word.pos_)] = {sentence.text.strip().replace('\n', '')}
    
    return lemmaPOS_sentence_association

def select_target_words(data_df):

    target_wordsPOS = set()
    targetWordPOS_to_sentences = []  # list of list of tuples: [[((word1, pos1), {sentence set1})] ... ]  -> each essay's target words is a list

    for idx, row in data_df.iterrows():
        essay_path = essay_dir + row['Filename']
        with open(essay_path, 'r') as f:
            doc = nlp(f.read())
            wordPOS_sentence_association = associate_words_POS_with_sentences(doc)
            
            sorted_lt = sorted(wordPOS_sentence_association.items(), key=lambda item: len(item[1]), reverse=True)
            sorted_lt = [((word, pos), sents) for (word, pos), sents in sorted_lt if len(sents) >= 2 and (word, pos) not in target_wordsPOS]  # pick the frequency >= 2

            for (word, pos), sents in sorted_lt:
                target_wordsPOS.add((word, pos))

            targetWordPOS_to_sentences.append(sorted_lt)

    return target_wordsPOS, targetWordPOS_to_sentences


def select_target_lemmas(data_df, nlp, essay_dir):

    target_lemmaPOS = set()
    targetLemmaPOS_to_sentences = []  # list of list of tuples: [[((lemma1, pos1), {sentence set1})] ... ]  -> each essay's target words is a list

    for i in tqdm(range(len(data_df))):
        essay_path = essay_dir + data_df['Filename'].iloc[i]
        with open(essay_path, 'r') as f:
            essay = f.read()
            doc = nlp(essay)
            lemmaPOS_sentence_association = associate_lemma_POS_with_sentences(doc)
            
            sorted_lt = sorted(lemmaPOS_sentence_association.items(), key=lambda item: len(item[1]), reverse=True)
            sorted_lt = [((lemma, pos), sents) for (lemma, pos), sents in sorted_lt if len(sents) >= 3 and (lemma, pos) not in target_lemmaPOS]  # pick the frequency >= 3

            for (lemma, pos), sents in sorted_lt:
                target_lemmaPOS.add((lemma, pos))

            targetLemmaPOS_to_sentences.append(sorted_lt)

    return target_lemmaPOS, targetLemmaPOS_to_sentences


def is_valid_english_word(word):
    """
    Check if a word is a valid English word using the nltk corpus.

    Args:
    word (str): The word to check.

    Returns:
    bool: True if the word is valid, False otherwise.
    """
    word_list = set(words.words())
    return word.lower() in word_list


def grammar_correction(text, model, batch_size=4):

    text = re.sub(r"\n", " ", text)
    text = re.sub(r"，", ", ", text)
    text = re.sub(r"。", ". ", text)
    text = re.sub(r"！", "! ", text)
    text = re.sub(r"([.?!%$&#])([\S])", r"\1 \2", text)
    result = batch_controller_plus(sent_tokenize(text), model, batch_size)
    grammar_feedback_output = {}  # list of dicts
    for i, (o, c, edits, diff, explanation) in enumerate(zip(result["original_sents"], result["correct_sents"], result["edits"], result["diff"], result["explanation"])):
        if edits:
            original_sentence = o
            corrected_sentence = c
            # error_type = ''
            # feedback = ''
            fdiffs = []
            for type, content in diff:
                if type == 'n':
                    fdiffs.append({
                        "orig": content + ' ' if content[-1] in '.?!' else content,
                        "corr": '',
                        "has_error": False,
                        "error_type": ''
                    })
                elif type == 'e':
                    ex = explanation.pop(0) if len(explanation) > 0 else ''
                    error_type = content['edit'].type
                    fdiffs.append({
                        "orig": content['orig'],
                        "corr": content['corr'],
                        "has_error": True,
                        "error_type": error_type
                    })

            grammar_feedback_output = {
                "original_sentence": original_sentence,
                "corrected_sentence": corrected_sentence,
                "has_error": True,
                "fdiff": fdiffs,
            }
        else:
            grammar_feedback_output = {
                "original_sentence": o,
                "corrected_sentence": o,
                "has_error": False,
                "fdiff": []
            }

    final_output = []
    for fdiff in grammar_feedback_output['fdiff']:
        if not fdiff['has_error']:
            final_output.append(fdiff['orig'])
        else:
            final_output.append(fdiff['corr'])

    return ''.join(final_output)


#### keep the results in a seperate csv file, columns are: word, pos, original sentence, tagged sentence, essay proficiency
def target_dataset(targetLemmaPOS_to_sentences, gec_model, level='low'):
    
    print('Creating dataset for level {}...'.format(level))
    original_sentences = []  # can be used to locate where the lemma is
    corrected_sentences = []  # grammarly correct sentences from original sentences
    essay_proficiency = []
    test_set_sentences = set(test_df['original_sentences'].values)
    # train_sentences = set()

    for i in tqdm(range(len(targetLemmaPOS_to_sentences))):
        essay_item = targetLemmaPOS_to_sentences[i]
    
        for (lemma, pos), sents in essay_item:
            if not is_valid_english_word(lemma): continue

            # select sentences that are not in the test set
            for sent in sents:
                # print(sent)
                if sent in test_set_sentences: continue
                original_sentences.append(sent)
                try:
                    corr_sent = grammar_correction(sent, gec_model, batch_size=32)
                    corrected_sentences.append(corr_sent)
                except Exception as e:
                    print('Error in GPT2GEC.predict: {}'.format(e))
                    corrected_sentences.append('')

            
                essay_proficiency.append(level)

    return original_sentences, corrected_sentences, essay_proficiency


def generate_raw_data():

    gec_model = gec_feedback.GECFeedback()
    nlp = spacy.load('en_core_web_md')

    # metadata_path = '/local/data/xuanming/ETS_Corpus_of_Non-Native_Written_English/data/text/index.csv' # multilingual
    metadata_path = '/local-scratch1/data/xuanming/ETS_Corpus_of_Non-Native_Written_English/data/text/index.csv'  # communication
    index_df = pd.read_csv(metadata_path, index_col=False)

    low_df = index_df[index_df['Score Level'] == 'low']
    medium_df = index_df[index_df['Score Level'] == 'medium']

    #### Iterate through both low and medium samples
    essay_dir = '/local-scratch1/data/xuanming/ETS_Corpus_of_Non-Native_Written_English/data/text/responses/original/'  # communication

    print(low_df.shape)
    print(medium_df.shape)

    print('Selecting target words for low samples...')
    low_target_lemmaPOS, low_targetLemmaPOS_to_sentences = select_target_lemmas(low_df, nlp, essay_dir)
    print('Selecting target words for medium samples...')
    medium_target_lemmaPOS, medium_targetLemmaPOS_to_sentences = select_target_lemmas(medium_df, nlp, essay_dir)

    low_data = target_dataset(low_targetLemmaPOS_to_sentences, gec_model, level='low')
    medium_data = target_dataset(medium_targetLemmaPOS_to_sentences, gec_model, level='medium')

    for level in ['low', 'medium']:
        original_sentences, corrected_sentences, essay_proficiency = eval(f'{level}_data')
        
        new_data = {
            'original_sentences': original_sentences,
            'corrected_sentences': corrected_sentences,
            'essay_proficiency': essay_proficiency
        }

        new_df = pd.DataFrame(new_data)
        new_df.to_csv(f'../data/train_sents_{level}_all.csv', index=False)




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

# generate_raw_data()
# load all data
# data_df = pd.read_csv('../data/targetLemmas_low_medium_all_clean.csv')
# print(data_df.info())

selected_sents_df = pd.read_csv('../data/train_sents_low_medium_high_1500.csv', index_col=False)
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

# store the generated dataframe into csv
generated_df.to_csv('../data/train_low_medium_high_1500.csv', index=False)

