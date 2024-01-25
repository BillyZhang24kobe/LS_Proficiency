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

def formulate_32_shots_prompt(target_word, sentence):
    return [
        {"role": "system", "content": "You are about to perform a lexical substitution task, considering the proficiency level of the substitute compared to the target word in a sentence. The task is to generate a set of candidate substitutes separated by commas for a target word in a given sentence. The target word is highlighted in the sentence, encompassed by two double asterisks. \n Here are some examples: \n Target word: honest \n Sentence: This is not right as people think that the message sent to them is **honest**, thus they believe whatever they hear. \n Substitutes: truthful, sincere, genuine \n \n  Target word: fully \n Sentence: that provides them with access to enjoy their life **fully**. \n Substitutes: completely, totally, absolutely, wholly \n \n Target word: roads \n Sentence: the government may build more **roads** so that they provide translation. \n Substitutes: highways, tracks, routes, lanes \n \n Target word: backs \n Sentence: Few year **backs**, music industry was dominated by the Walkman from Sony. \n Substitutes: ago, previously, earlier, formerly \n \n Target word: For example \n Sentence: **For example**, let' s talk about the French Revolution. \n Substitutes: For instance, To illustrate \n \n Target word: better \n Sentence: But if you look into it, why do pairs of shoes coast so much, because it is an advertisement that Nike makes their shoes look **better** than everyone else's. \n Substitutes: superior, improved, exceptional \n \n Target word: any \n Sentence: Learning facts in a scientific field, like for example the resulting speed of an object, does not give **any** clue about the result of an experiment in another context. \n Substitutes: whatsoever, at all, absolutely \n \n Target word: rather \n Sentence: When they achieve one particular goal they proceed with a new goal **rather** than doing what they already know. \n Substitutes: preferably, instead, alternatively, ideally \n \n Target word: Balancing \n Sentence: **Balancing** between these methods is actually what we need. \n Substitutes: Harmonizing \n \n Target word: come \n Sentence: we leave like one day will **come** \n Substitutes: arrive, appear, emerge \n \n Target word: select \n Sentence: But If they could get a second subject they could **select** one of them after starting to work and get real information about their business life. \n Substitutes: choose, pick, decide on, opt for, single out \n \n Target word: idea \n Sentence: Once we have an **idea**, then we can easily understand the topic. \n Substitutes: concept, thought, notion, view, perception \n \n Target word: history \n Sentence: For **history** and math or other things to learn, the first thing is that you have to understand. \n Substitutes: background, past, chronicles \n \n Target word: bring \n Sentence: I think that the studies have to **bring** this balance between the general culture and the specialisation. \n Substitutes: establish, institute, foster, promote \n \n Target word: seldom \n Sentence: They are **seldom** educated to help others. \n Substitutes: rarely, infrequently, scarcely, hardly \n \n Target word: advertisement \n Sentence: With such aesthetically alluring visual pictures -with the help of modern technology, photography, or literature , an **advertisement** nearly blinds the reader depending on the reader's class, culture, or sex. \n Substitutes: promotion, publicity, broadcast, notice \n \n Target word: went \n Sentence: For example, I **went** on a school trip to China recently and we visited many famous gardens in China. \n Substitutes: journeyed, traveled, toured, roamed \n \n Target word: expect \n Sentence: Due to the efforts to tackle this problem one may **expect** a decrease in car usage over the next twenty years. \n Substitutes: anticipate, predict, forecast \n \n Target word: invention \n Sentence: For example, my father invested something 5 years ago, but the **invention** was not famous because, at that time, my father did not have money. \n Substitutes: creation, innovation, product, device \n \n Target word: help \n Sentence: In addition, most young people think that communities are not their best friends or relations, so they are not willing to **help** them. \n Substitutes: assist, support, aid, advance \n \n  Target word: jam \n Sentence: With the car, you would probably take much longer due to traffic **jam** or bad streets, while you could book a cheap flight and get to your destination much more relaxed and comfortable. \n Substitutes: congestion, gridlock, bottleneck, blockage \n \n  Target word: wonderful \n Sentence: So, if some wants to have a **wonderful**, memorable trip, he or she has to avoid a tour guide who probably takes his or her independence away. \n Substitutes: marvelous, fantastic, splendid, incredible, outstanding \n \n Target word: actually \n Sentence: If the advertisers are so keen on advertising such characteristics they should suggest to the manufacturers to **actually** integrate these characteristics into the product and then advertise them so they would not be cheating the customers but actually showing a true,realistic picture. \n Substitutes: truly, really, indeed, genuinely \n \n Target word: entrepreneurs \n Sentence: Imitative **entrepreneurs** are the ones who imitate the existing product. \n Substitutes: businessmen, pioneers, innovators \n \n Target word: however \n Sentence: At the first place I visited, **however**, I was so impressed by the beautiful scenery of a river and castle that I changed my plan and stayed there for one week. \n Substitutes: nonetheless, nevertheless, yet, still \n \n Target word: Finally \n Sentence: **Finally** those who are talented must come forward and they have to use the resources that they have. so that he will be in a good position. \n Substitutes: Ultimately, Eventually, Lastly, In the end \n \n Target word: take \n Sentence: To support this idea I will **take** the example of an old couple who will represent old people. \n Substitutes: utilize, employ, use, adopt, apply \n \n Target word: very \n Sentence: Other professions, such as teaching, engineering or accounting, need specialized knowledge, which is restricted to a field or few fields, but that is **very** thorough. \n Substitutes: extremely, incredibly, exceptionally, particularly \n \n Target word: tried \n Sentence: During his work, he **tried** thousand kinds of materials. \n Substitutes: experimented, tested, attempted \n \n Target word: worked \n Sentence: For example, five years ago I **worked** for an advertising company. \n Substitutes: toiled, labored, served \n \n Target word: believing \n Sentence: People should have a proper idea about the product before **believing** it by seeing a proper advertisement. \n Substitutes: trusting, accepting, endorsing, crediting \n \n Target word: Filling \n Sentence: This is the **Filling** of spare time, A man can only do one thing or job well but has a lot of habits. \n Substitutes: Occupying, Utilization, Use"},
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
                messages=formulate_32_shots_prompt(target_word, sentence)
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
    # test_data = pd.read_csv('data/dev/test_trial_68_final.csv', index_col=False)

    print('Generating predictions for GPT-4...')
    generate_pred_gpt(test_data, 'outputs/gpt-4-32_test_final_cefr.csv', model_name='gpt-4')

    print('Generating predictions for ChatGPT...')
    generate_pred_gpt(test_data, 'outputs/gpt-3.5-turbo-1106-32_test_final_cefr.csv', model_name='gpt-3.5-turbo-1106')