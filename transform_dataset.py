import pandas as pd
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize

def nltk_to_wordnet_pos(tag):
    if tag.startswith('NN'):
        return 'N'
    elif tag.startswith('VB'):
        return 'V'
    elif tag.startswith('JJ'):
        return "J"
    elif tag.startswith('RB'):
        return 'R'
    else:
        return ''

def find_word_index(tagged_sentence, target_word):
    '''
    Find the index of the target word in the tagged sentence or context sentence 
    '''
    tagged_sentence.replace(f"**{target_word}**", "*")
    # Splitting the sentence into words
    words_in_tagged = word_tokenize(tagged_sentence) # need to find a way to both split on punct and keep **asdf** intact
    # Finding the word index of the word surrounded by double asterisks
    for i, word in enumerate(words_in_tagged):
        if '*' in word:
            sentence = words_in_tagged[:i] + [target_word] + words_in_tagged[i+1:]
            pos_tags = nltk.pos_tag(sentence)
            for w, tag in pos_tags:
                if target_word == w:
                    return {'word_index': i, "pos": nltk_to_wordnet_pos(tag), "sent": " ".join(sentence)}
    return {'word_index': -1, "pos": None, "sent": None}


data = pd.read_csv('data/dev/LS-Pro_test_trial_68_final.csv')
data['instance'] = data.index
res = data.apply(lambda row: find_word_index(row['Sentence'], row['target word']), axis=1, result_type='expand')
data = pd.concat([data, res], axis='columns')
data['target_word'] = data['target word'] + "." + data['pos']
data.to_csv('data/dev/processed.tsv', sep='\t', index=False, header=False, columns=['target_word', 'instance', 'word_index', 'sent'])
