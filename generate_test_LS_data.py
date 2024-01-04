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
from nltk.corpus import words
from utils import *

os.environ['GPU_DEVICE'] = '7'

# OpenAI credentials
import api_secrets
from openai import OpenAI
os.environ['OPENAI_API_KEY'] = api_secrets.openai_api_key

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    organization=api_secrets.openai_org
)

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


def select_target_lemmas(data_df):

    target_lemmaPOS = set()
    targetLemmaPOS_to_sentences = []  # list of list of tuples: [[((lemma1, pos1), {sentence set1})] ... ]  -> each essay's target words is a list

    for i in tqdm(range(len(data_df))):
        # for i in tqdm(range(len(low_df))):
        # original_essay.append(open(essay_dir + low_df['Filename'].iloc[i]).read())
        essay_path = essay_dir + data_df['Filename'].iloc[i]
        with open(essay_path, 'r') as f:
            essay = f.read()
            # corrected_essay = grammar_correction(essay)
            # doc = nlp(corrected_essay)
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
def target_dataset(targetLemmaPOS_to_sentences, gec_model, nlp, level='low'):
    
    print('Creating dataset for level {}...'.format(level))
    t_lemmas = []
    t_pos = []
    original_sentences = []  # can be used to locate where the lemma is
    corrected_sentences = []  # grammarly correct sentences from original sentences
    tagged_sentences = []  # with the target word encompassed with * sign
    tagged_target_words = []  # tagged words from tagged sentences
    essay_proficiency = []

    for i in tqdm(range(len(targetLemmaPOS_to_sentences))):
        essay_item = targetLemmaPOS_to_sentences[i]
        for (lemma, pos), sents in essay_item:
            if not is_valid_english_word(lemma): continue

            t_lemmas.append(lemma)
            t_pos.append(pos)

            # select a random sentence from sents
            orig_sentence = random.choice(list(sents))
            original_sentences.append(orig_sentence)
            try:
                # orig_sentence = re.sub(r"\n", " ", orig_sentence)
                # orig_sentence = re.sub(r"，", ", ", orig_sentence)
                # orig_sentence = re.sub(r"。", ". ", orig_sentence)
                # orig_sentence = re.sub(r"！", "! ", orig_sentence)
                # orig_sentence = re.sub(r"([.?!%$&#])([\S])", r"\1 \2", orig_sentence)
                # sentence = sent_tokenize(orig_sentence)
                # result = gec_model.gec(sentence)
                # corr_sent = ' '.join(result['correct_sents'])
                corr_sent = grammar_correction(orig_sentence, gec_model, batch_size=32)
                corrected_sentences.append(corr_sent)
            except Exception as e:
                print('Error in GPT2GEC.predict: {}'.format(e))
                corrected_sentences.append('')

            
            tag_sentence = ''
            tag_word = ''
            doc = nlp(corr_sent)
            for w in doc:
                if w.lemma_ == lemma and w.pos_ == pos:
                    tag_sentence = corr_sent.replace(w.text, f'**{w.text}**')
                    tag_word = w.text
                    break
            
            tagged_sentences.append(tag_sentence)
            tagged_target_words.append(tag_word)
            
            essay_proficiency.append(level)

    return t_lemmas, t_pos, original_sentences, corrected_sentences, tagged_sentences, tagged_target_words, essay_proficiency


def generate_raw_data():

    gec_model = gec_feedback.GECFeedback()
    nlp = spacy.load('en_core_web_md')

    metadata_path = '/local/data/xuanming/ETS_Corpus_of_Non-Native_Written_English/data/text/index.csv' # multilingual
    metadata_path = '/local-scratch1/data/xuanming/ETS_Corpus_of_Non-Native_Written_English/data/text/index.csv'  # communication
    index_df = pd.read_csv(metadata_path, index_col=False)

    low_df = index_df[index_df['Score Level'] == 'low']
    medium_df = index_df[index_df['Score Level'] == 'medium']

    #### Iterate through both low and medium samples
    essay_dir = '/local-scratch1/data/xuanming/ETS_Corpus_of_Non-Native_Written_English/data/text/responses/original/'  # communication

    print(low_df.shape)
    print(medium_df.shape)

    print('Selecting target words for low samples...')
    low_target_lemmaPOS, low_targetLemmaPOS_to_sentences = select_target_lemmas(low_df, gec_model, nlp)
    print('Selecting target words for medium samples...')
    medium_target_lemmaPOS, medium_targetLemmaPOS_to_sentences = select_target_lemmas(medium_df, gec_model, nlp)

    low_data = target_dataset(low_targetLemmaPOS_to_sentences, level='low')
    medium_data = target_dataset(medium_targetLemmaPOS_to_sentences, level='medium')

    for level in ['low', 'medium']:
        t_lemmas, t_pos, original_sentences, corrected_sentences, tagged_sentences, tagged_target_words, essay_proficiency = eval(f'{level}_data')
        new_data = {
            'target lemma': t_lemmas,
            'pos': t_pos,
            'original_sentences': original_sentences,
            'corrected_sentences': corrected_sentences,
            'tagged_sentences': tagged_sentences,
            'tagged_target_words': tagged_target_words,
            'essay_proficiency': essay_proficiency
        }

        new_df = pd.DataFrame(new_data)
        new_df.to_csv(f'../data/targetLemmas_{level}_all.csv', index=False)


def formulate_prompt(tagged_sentence, target_word):
    return [
        {"role": "system", "content": "You are a helpful assistant to perform a lexical substitution task. Specifically, you will be given a tuple of text consisting of 1) context with target word indicated using asterisks, and a 2) natural language query. You should generate exactly five substitutes separated by commas. Do not generate the same word as the target word."},
        {"role": "user", "content": "I have completed the invoices for April, May and June and we owe Pasadena each month for a **total** of $3,615,910.62. I am waiting to hear back from Patti on May and June to make sure they are okay with her. \n \n Q: What are appropriate substitutes for **total** in the above text?"},
        {"role": "assistant", "content": "amount, sum, price, balance, gross"},
        {"role": "user", "content": "…I thought as much. Now leave, before I **call** the rats on you.” We left. \n \n Q: What are appropriate substitutes for **call** in the above text?"},
        {"role": "assistant", "content": "summon, order, rally, send, sic"},
        {"role": "user", "content": "The e-commerce free **zone** is situated in north Dubai, near the industrial free **zone** in Hebel Ali. \n \n Q: What are appropriate substitutes for **zone** in the above text?"},
        {"role": "assistant", "content": "sector, district, area, region, section"},
        {"role": "user", "content": "The state's action, the first in the nation, has the blessing of the American Psychological Association (APA), which considers prescriptive authority a **logical** extension of psychologists' role as health-care providers. \n \n Q: What are appropriate substitutes for **logical** in the above text?"},
        {"role": "assistant", "content": "rational, reasonable, sensible, justifiable, relevant"},
        {"role": "user", "content": "They nodded. “Oh, **totally**,” said the hunchback. “I get that all the time around here.” \n \n Q: What are appropriate substitutes for **totally** in the above text?"},
        {"role": "assistant", "content": "absolutely, for sure, surely, completely, definitely"},
        {"role": "user", "content": "{} \n \n Q: What are appropriate substitutes for **{}** in the above text?".format(tagged_sentence, target_word)},
    ]

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


def get_cefr_annotation(test_df):
    target_words = []
    context_sentences = []
    substitutes = []  # only one substitute in a row
    acc_substitutes = []
    unacc_substitutes = []
    CEFR_level_t_word = []
    CEFR_level_substitute = []

    for i in tqdm(range(len(test_df))):
        row = test_df.iloc[i]
        candidates = row['acceptable_substitutes']
        # convert the substitutes string to list
        candidates = candidates.replace('[', '').replace(']', '').replace("'", '').split(',')
        get_t_word_cefr = False
        for candidate in candidates:
            candidate = candidate.strip()
            # remove the period at the end of the candidate
            if candidate.endswith('.'):
                candidate = candidate[:-1]
        
            t_word = row['target word']
            target_words.append(t_word)
            context_sentences.append(row['Sentence'])
            substitutes.append(candidate)
            acc_substitutes.append(row['acceptable_substitutes'])
            unacc_substitutes.append(row['unacceptable_substitutes'])
            
            # get the CEFR level of the target word
            if not get_t_word_cefr:
                t_word_cefr = get_cefr_level(row['Sentence'], t_word)
                get_t_word_cefr = True
                if t_word_cefr is not None:
                    print("target word: {}, CEFR level: {}".format(t_word, t_word_cefr))
                    CEFR_level_t_word.append(t_word_cefr)
                else:
                    print("target word: {}, CEFR level: {}".format(t_word, 'None'))
                    CEFR_level_t_word.append('None')
            else:
                CEFR_level_t_word.append(t_word_cefr)

            # get the CEFR level of the substitute
            try:
                t_word_idx = find_word_index(row['Sentence'], "", t_word)
                context_sentence = replace_word(row['Sentence'], t_word_idx, candidate)
                substitute_cefr = get_cefr_level(context_sentence, candidate)
                print("substitute: {}, CEFR level: {}".format(candidate, substitute_cefr))
                if substitute_cefr is not None:
                    print("Correct substitute CEFR")
                    CEFR_level_substitute.append(substitute_cefr)
                else:
                    CEFR_level_substitute.append('None')
            except Exception as e:
                print(e)
                print("substitute: {}, CEFR level: {}".format(candidate, 'None'))
                CEFR_level_substitute.append('None')

    return target_words, context_sentences, substitutes, acc_substitutes, unacc_substitutes, CEFR_level_t_word, CEFR_level_substitute


def generate_final_data(generated_df, output_path):
    """
    Generate the final data for LS-Prof: (w, s, w^a, w^a_p)
    """

    target_words = []
    t_words_cefr = []
    sentences = []
    acceptable_substitutes = []
    unacceptable_substitutes = []
    prof_acceptable_substitutes = []
    prof_unacceptable_substitutes = []
    prof_acc_cefr_lst = []
    prof_unacc_cefr_lst = []
    t_sent_pair_set = set()
    for idx, row in generated_df.iterrows():
        t_word = row['target_words'].strip()
        sentence = row['Sentences'].strip()
        if (t_word, sentence) in t_sent_pair_set: continue
        # add the target word-sentence pair to the set
        t_sent_pair_set.add((t_word, sentence))

        target_words.append(t_word)
        t_words_cefr.append(row['CEFR_level_t_word'])
        sentences.append(sentence)
        acceptable_substitutes.append(row['acceptable_substitutes'])
        unacceptable_substitutes.append(row['unacceptable_substitutes'])

        # find all rows with the same target word-sentence pair
        same_tword_rows = generated_df[(generated_df['target_words'] == t_word) & (generated_df['Sentences'] == sentence)]
        same_tword_rows_no_phrase = same_tword_rows[~(same_tword_rows['CEFR_level_substitute'] == 'None')]
        same_tword_rows_phrase = same_tword_rows[same_tword_rows['CEFR_level_substitute'] == 'None']
        
        # get all prof_acceptable substitutes (CEFR_level_substitute >= CEFR_level_t_word)
        prof_acc_subs_no_phrase = same_tword_rows_no_phrase[same_tword_rows_no_phrase['CEFR_level_substitute'] >= same_tword_rows_no_phrase['CEFR_level_t_word']]['substitutes'].tolist()
        prof_acc_cefr_no_phrase = same_tword_rows_no_phrase[same_tword_rows_no_phrase['CEFR_level_substitute'] >= same_tword_rows_no_phrase['CEFR_level_t_word']]['CEFR_level_substitute'].tolist()
        prof_acc_subs_phrase = same_tword_rows_phrase['substitutes'].tolist()
        prof_acc_cefr_phrase = same_tword_rows_phrase['CEFR_level_substitute'].tolist()
        prof_acc_subs = prof_acc_subs_no_phrase + prof_acc_subs_phrase
        prof_acc_cefr = prof_acc_cefr_no_phrase + prof_acc_cefr_phrase

        # get all prof_unacceptable substitutes (CEFR_level_substitute < CEFR_level_t_word)
        prof_unacc_subs = same_tword_rows_no_phrase[same_tword_rows_no_phrase['CEFR_level_substitute'] < same_tword_rows_no_phrase['CEFR_level_t_word']]['substitutes'].tolist()
        prof_unacc_cefr = same_tword_rows_no_phrase[same_tword_rows_no_phrase['CEFR_level_substitute'] < same_tword_rows_no_phrase['CEFR_level_t_word']]['CEFR_level_substitute'].tolist()

        # append to the list
        prof_acceptable_substitutes.append(prof_acc_subs)
        prof_unacceptable_substitutes.append(prof_unacc_subs)
        prof_acc_cefr_lst.append(prof_acc_cefr)
        prof_unacc_cefr_lst.append(prof_unacc_cefr)

    # create a new df for all_df
    test_data = {
        'target word': target_words,
        'Sentence': sentences,
        'acc_subs': acceptable_substitutes,
        'unacc_subs': unacceptable_substitutes,
        'prof_acc_subs': prof_acceptable_substitutes,
        'prof_unacc_subs': prof_unacceptable_substitutes,
        't_words_cefr': t_words_cefr,
        'prof_acc_cefr': prof_acc_cefr_lst,
        'prof_unacc_cefr': prof_unacc_cefr_lst
    }

    df = pd.DataFrame(test_data)

    df.to_csv(output_path, index=False)

# load all data
# data_df = pd.read_csv('../data/targetLemmas_low_medium_all_clean.csv')
# print(data_df.info())

# get_candidate_substitutes_gpt4(data_df)


# load test data
test_df = pd.read_csv('../data/ann_data/test_set/test_688_acc_unacc_deleteEmptyAcc.csv', index_col=False)
target_words, context_sentences, substitutes, acc_substitutes, unacc_substitutes, CEFR_level_t_word, CEFR_level_substitute = get_cefr_annotation(test_df)

# store the generated data into a dataframe
generated_df = pd.DataFrame({
    'target_words': target_words,
    'CEFR_level_t_word': CEFR_level_t_word,
    'substitutes': substitutes,
    'CEFR_level_substitute': CEFR_level_substitute,
    'Sentences': context_sentences,
    'acceptable_substitutes': acc_substitutes,
    'unacceptable_substitutes': unacc_substitutes,
})

# drop the rows with null values
generated_df = generated_df.dropna()

# drop rows where CEFR_level_t_word is null
generated_df = generated_df[~(generated_df['CEFR_level_t_word'] == 'None')]

generated_df.to_csv('../data/ann_data/test_set/test_688_acc_unacc_deleteEmptyAcc_CEFR_unravel.csv', index=False)

generate_final_data(generated_df, '../data/ann_data/test_set/LS-Pro_test_688_final_cefr.csv')



