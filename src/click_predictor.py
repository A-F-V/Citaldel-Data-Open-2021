from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error



data = pd.read_csv('data/processed/processed_xy.csv')


# Preprocess data for training a model using unnormalized data
def unnormalized_predictor_process(data:pd.DataFrame):
    x = data[['test_mean_impressions','test_mean_clicks','Positive','Negative','Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise','Trust']]
    y = data[['impressions','clicks']]
    y.loc[:,'impressions'] = y['impressions']-x['test_mean_impressions']
    y.loc[:,'clicks'] = y['clicks']-x['test_mean_clicks']
    return x,y

def unnormalized_predictor_popularity_process(data:pd.DataFrame):
    x = data[['Positive','Negative','Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise','Trust']]
    x = x.join(pd.DataFrame(data['test_mean_clicks']/data['test_mean_impressions'],columns=['mean_popularity']))
    y = data['clicks']/data['impressions']
    print(x)
    print(y)
    return x,y

def normalized_predictor_popularity_process(data:pd.DataFrame):
    x = data[['Positive','Negative','Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise','Trust']]/2#/data['emotive_word_count']
    y = data['clicks']/data['impressions'] - data['test_mean_clicks']/data['test_mean_impressions']
    print(x)
    print(y)
    return x,y
#creates the two sets of data for training and testing
def train_test(data:pd.DataFrame, train_size:float=0.8):
    '''
    Returns a train and test set
    '''
    train_size = int(len(data)*train_size)
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]
    return train, test


def train_model_and_score(data:pd.DataFrame, model_trainer,preprocessor,scorer): #model_Trainer takes in trainx and trainy 
    '''
    Returns a trained model and score
    '''
    train, test = train_test(data)
    (trainx,trainy),(testx,testy) = preprocessor(train),preprocessor(test)
    model_trained = model_trainer(trainx,trainy)
    pred_test_y = model_trained.predict(testx)
    return model_trained, scorer(testy,pred_test_y)

def RFRegressor(x,y):
    return RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0).fit(x,y)


#model,score = train_model_and_score(data,RFRegressor,unnormalized_predictor_process,mean_squared_error)
#print(score**(0.5))

model,score = train_model_and_score(data,RFRegressor,normalized_predictor_popularity_process,mean_squared_error)
print(score**(0.5))

