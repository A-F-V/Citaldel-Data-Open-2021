from cmath import log
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

total_freq = 588090082941

lemmatizer = WordNetLemmatizer()
frequency = pd.read_csv('data/processed/unigram_lemmatized_freq.csv',index_col=0)

lexicon = pd.read_csv('data/raw/lexicon.csv')
lexicon = lexicon.set_index('English (en)') #now can find word by indexing

words = words.words()
lexicon_words = set(lexicon.index)

packages = pd.read_csv('data/raw/packages.csv')

processed = pd.DataFrame(columns=['test_id', 'test_mean_impressions','test_mean_clicks','emotive_word_count','token_word_count','impressions','clicks'])

def safe_tokenize(x):
    try:
        return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(x)]
    except:
        return []

def to_emotive_lexicons(x):
    return list(filter(lambda word: word in lexicon_words, x))

def freq(word):
    try:
        return frequency.loc[word].values[0]+1
    except:
        return 1

def to_emotion_profile(x):
    return np.sum([np.array(lexicon.loc[word])*(np.log(frequency.loc[word].values[0]+1)) for word in x],axis=0) 

processed['test_id'] = packages['test_id'].astype(str)
processed['impressions'] = packages['impressions']
processed['clicks'] = packages['clicks']

tokenized_words = packages['headline'].progress_apply(safe_tokenize)
processed['token_word_count'] = tokenized_words.progress_apply(len)

emotive_tokenized_words = tokenized_words.progress_apply(to_emotive_lexicons)
processed['emotive_word_count'] = emotive_tokenized_words.progress_apply(len)
goodindices = processed['emotive_word_count'] > 0
emotions = pd.DataFrame(emotive_tokenized_words[goodindices].progress_apply(to_emotion_profile).to_list(),columns=['Positive','Negative','Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise','Trust'])
processed = processed[goodindices].reset_index(drop=True)
processed = pd.concat([processed,emotions],axis=1)

test_means = processed.groupby(['test_id']).mean()
processed['test_mean_impressions'] = processed['test_id'].progress_apply(lambda x: test_means['impressions'].loc[x])
processed['test_mean_clicks'] = processed['test_id'].progress_apply(lambda x: test_means['clicks'].loc[x])
processed.to_csv('data/processed/processed_xy_augmented.csv')



