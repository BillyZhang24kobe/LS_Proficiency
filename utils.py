import requests
import re

# Cathoven AI credentials
import api_secrets
client_id = api_secrets.Client_ID
client_secret = api_secrets.Client_Secret


def find_word_index(tagged_sentence, context_sentence, target_word):
    '''
    Find the index of the target word in the tagged sentence or context sentence 
    '''
    # Splitting the sentence into words
    words_in_tagged = tagged_sentence.split()
    words_in_context = context_sentence.split()
    # Finding the word index of the word surrounded by double asterisks
    word_index_in_tagged = [i for i, word in enumerate(words_in_tagged) if "**" + target_word + "**" in word]
    word_index_in_context = [i for i, word in enumerate(words_in_context) if target_word in word]

    if word_index_in_tagged or word_index_in_context:
        if word_index_in_tagged:
            return word_index_in_tagged[0]
        else:
            return word_index_in_context[0]  
    else:
        return None
    

# replace target word with the candidate substitute given the target word index
def replace_word(context_sentence, target_idx, candidate_substitute):
    '''
    Replace the target word with the candidate substitute given the target word index
    '''
    # Splitting the sentence into words
    words = context_sentence.split()

    # Replace the word surrounded by double asterisks with the candidate substitute
    words[target_idx] = candidate_substitute.strip()
    # Joining the words back into a sentence
    new_sentence = " ".join(words)
    return new_sentence


def get_cefr_level(original_context, target_word):
    '''
    Get the CEFR level of the target word in the original context
    '''

    data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'keep_min': "true",  # Replace value with the actual value
        'return_final_levels': "true",  # Replace value with the actual value
        'text': original_context,
        'return_sentences': "true"  # Replace value with the actual value
    }

    try:
        response = requests.post('https://enterpriseapi.cathoven.com/cefr/process_text', json=data)
        response.raise_for_status()
        responseData = response.json()

        # find the index of the target word in the response
        word_index = None
        contain_more_than_one_sentence = False
        sentence_index = None  # the sentence that contains the target word
        # detect if original context contains more than one sentence
        if len(responseData['sentences']) > 1:
            contain_more_than_one_sentence = True
            for sid, obj in responseData['sentences'].items():
                if target_word in obj['word']:
                    words = obj['word']
                    word_index = [i for i, word in enumerate(words) if target_word == word]
                    sentence_index = sid
                    break
        else:
            words = responseData['sentences']['0']['word']
            word_index = [i for i, word in enumerate(words) if target_word == word]
        
        if word_index:
            t_word_index = word_index[0]  
            if contain_more_than_one_sentence and sentence_index:
                cefr_level = responseData['sentences'][sentence_index]['CEFR'][t_word_index]
            else:
                cefr_level = responseData['sentences']['0']['CEFR'][t_word_index]
        else:
            print("WARNING: Target word {} does not exist in the context sentence".format(target_word))
            print("Original context: {}".format(original_context))
            cefr_level = None
    except requests.exceptions.HTTPError as errh:
        print("Http Error:", errh)
        cefr_level = None
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:", errc)
        cefr_level = None
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
        cefr_level = None
    except requests.exceptions.RequestException as err:
        print("OOps: Something Else", err)
        cefr_level = None
    except Exception as e:
        print(e)
        cefr_level = None

    return cefr_level
