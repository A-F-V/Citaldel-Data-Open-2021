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

processed = pd.DataFrame(columns=['test_id', 'test_mean_impressions','test_mean_clicks','emotive_word_count','impressions','clicks'])

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
goodindices = processed['emotive_word_count'] > 0
emotions = pd.DataFrame(emotive_tokenized_words[goodindices].progress_apply(to_emotion_profile).to_list(),columns=['Positive','Negative','Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise','Trust'])
processed = processed[goodindices].reset_index(drop=True)
processed = pd.concat([processed,emotions],axis=1)

test_means = processed.groupby(['test_id']).mean()
processed['test_mean_impressions'] = processed['test_id'].progress_apply(lambda x: test_means['impressions'].loc[x])
processed['test_mean_clicks'] = processed['test_id'].progress_apply(lambda x: test_means['clicks'].loc[x])
processed.to_csv('data/processed/processed_xy.csv')

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





