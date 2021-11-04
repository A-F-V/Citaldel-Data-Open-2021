import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer #lemitization
lemmatizer = WordNetLemmatizer()





lexicon = pd.read_csv('data/raw/lexicon.csv')
lexicon = lexicon.set_index('English (en)') #now can find word by indexing
words = set(lexicon.index.values)
packages = pd.read_csv('data/raw/packages.csv')

# group by test_id
tests = packages.groupby('test_id')

processed = pd.DataFrame(columns=['test_id', 'emotion', 'word_count','impressions','clicks'])

def safe_tokenize(x):
    try:
        l = word_tokenize(x)
        return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(x) if lemmatizer.lemmatize(word.lower()) in words]
    except:
        return []


processed['test_id'] = packages['test_id']
processed['impressions'] = packages['impressions']
processed['clicks'] = packages['clicks']

tokenized_words = packages['headline'].map(safe_tokenize)
print(tokenized_words)
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


