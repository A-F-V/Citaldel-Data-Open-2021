from re import T
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')
from nltk.stem import WordNetLemmatizer #lemitization
from nltk.corpus import words
from tqdm import tqdm
tqdm.pandas()



lemmatizer = WordNetLemmatizer()


lexicon = pd.read_csv('data/raw/lexicon.csv')
lexicon = lexicon.set_index('English (en)') #now can find word by indexing

words = words.words()
lexicon_words = set(lexicon.index)

packages = pd.read_csv('data/raw/packages.csv')

# group by test_id
tests = packages.groupby('test_id')

processed = pd.DataFrame(columns=['test_id', 'emotion','emotive_word_count','impressions','clicks'])

def safe_tokenize(x):
    try:
        return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(x)]
    except:
        return []

def to_emotive_lexicons(x):
    return list(filter(lambda word: word in lexicon_words, x))

def to_emotion_profile(x):
    return np.sum([np.array(lexicon.loc[word]) for word in x],axis=0)

processed['test_id'] = packages['test_id'].astype(str)
processed['impressions'] = packages['impressions']
processed['clicks'] = packages['clicks']

tokenized_words = packages['headline'].progress_apply(safe_tokenize)
emotive_tokenized_words = tokenized_words.progress_apply(to_emotive_lexicons)

processed['emotive_word_count'] = emotive_tokenized_words.progress_apply(len)
processed['emotion'] = emotive_tokenized_words.progress_apply(to_emotion_profile)
processed = processed[processed['emotive_word_count'] > 0]

print(processed.head())
processed.to_csv('data/processed/processed_x.csv')

#for i in range(len(gb)): # [test_id, [emotions], number_of_words, impressions, clicks]
#    if i% 1000 == 0: print(i)
#    gb1i1 = (gb1[i][1][['slug', 'impressions', 'clicks']]).values.tolist()
#    for j in range(len(gb1i1)):
#        emotions = np.array([0,0,0,0,0,0,0,0,0,0])
#        words = gb1i1[j][0].split('-')
#        nofwords = len(words) - 2
#        for k in words:
#            if k in lexicon1:
#                ind = lexicon1.index(k)
#                emotions += lexicon2[ind]
#        ans.append([gb1[i][0]] + emotions.tolist() + [nofwords, gb1i1[j][1], gb1i1[j][2]])
#arr = np.array(ans)
#np.save('datathon_data', arr)


