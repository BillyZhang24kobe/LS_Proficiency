import pandas as pd
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

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
    Find the index of the target word in the tagged sentence or context sentence.
    Also POS tags the target word for ParaLS input.
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

def prolex_to_parals_input():
    '''
    Convert the ProLex test set to the input format specified by the ParaLS repo.
    Saves the target with context at data/test/processed.tsv,
    and saves gold substitute files for both acceptable substitutes (data/test/gold_acc.tsv)
    and proficiency-oriented substitutes (data/test/gold_prof_acc.tsv).
    '''
    data = pd.read_csv('data/test/ProLex_v1.0_test.csv')
    data['instance'] = data.index
    res = data.apply(lambda row: find_word_index(row['Sentence'], row['target word']), axis=1, result_type='expand')
    data = pd.concat([data, res], axis='columns')
    data['target_word'] = data['target word'] + "." + data['pos']
    data.to_csv('data/test/processed.tsv', sep='\t', index=False, header=False, columns=['target_word', 'instance', 'word_index', 'sent'])
    data['sep'] = '::'
    data['acc_subs'] = data['acc_subs'].map(eval)
    data['acc_subs'] = data['acc_subs'].str.join(' 1;') + ' 1;'
    data['prof_acc_subs'] = data['prof_acc_subs'].map(eval)
    data['prof_acc_subs'] = data['prof_acc_subs'].str.join(' 1;') + ' 1;'
    data.to_csv('data/test/gold_acc.tsv', sep='\t', index=False, header=False, columns=['target_word', 'instance', 'sep', 'acc_subs'])
    data.to_csv('data/test/gold_prof_acc.tsv', sep='\t', index=False, header=False, columns=['target_word', 'instance', 'sep', 'prof_acc_subs'])

def parals_output_to_prolex():
    '''
    Convert the output from ParaLS to the ProLex format for evaluation.
    Uses the target with context file from data/test/processed.tsv
    and ParaLS outputs (best out of ten) copied over, and
    saves the converted output to outputs/ParaLS_test.csv.
    '''
    target = pd.read_csv('data/test/processed.tsv', sep='\t', header=None, names=['target_word', 'instance', 'word_index', 'sent'])
    output = pd.read_csv('baselines/ParaLS_results/test/lspro.out.embed.0.02.oot', sep=' ', header=None, names=['target_word', 'instance', 'sep', 'subs'])
    target = target.merge(output, on='instance', suffixes=(None, '_out'))
    detok = TreebankWordDetokenizer()
    target['target word'] = target['target_word'].str.split('.').str[0]
    target['sent'] = target['sent'].str.split(' ')
    target['Sentence'] = target.apply(lambda row: detok.detokenize(row['sent'][:row['word_index']] + [f"**{row['target word']}**"] + row['sent'][row['word_index']+1:]), axis=1)
    target['Substitutes'] = target['subs'].str.replace(';', ', ')
    target.to_csv('outputs/ParaLS_test.csv', index=False, columns=['target word','Sentence','Substitutes'])


if __name__ == '__main__':
    prolex_to_parals_input()
    # parals_output_to_prolex()