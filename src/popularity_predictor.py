from cProfile import run
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt

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
    y = data['clicks']/data['impressions']-data['test_mean_clicks']/data['test_mean_impressions']
    return x,y


def normalized_predictor_popularity_process(data:pd.DataFrame):
    x = data[['Positive','Negative','Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise','Trust']].div(data['emotive_word_count'],axis=0)
    y = data['clicks']/data['impressions'] - data['test_mean_clicks']/data['test_mean_impressions']
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

#model,score = train_model_and_score(data,RFRegressor,unnormalized_predictor_process,mean_squared_error)
#print(score**(0.5))

def run_experiment(data,model,preprocessor,scorers,rname):
    with mlflow.start_run(run_name=rname):
        x, y = preprocessor(data)
        (trainx,testx),(trainy,testy) = train_test(x),train_test(y)
        model_trained = model.fit(trainx,trainy)
        pred_test_y = model_trained.predict(testx)
        for name in scorers:
            mlflow.log_metric(f"{name}",scorers[name](testy,pred_test_y))
        visualize(x,y,"Anger","Fear",rname)
        mlflow.log_artifact(f"data/visualizations/{rname}_Anger_Fear.png")

def visualize(x,y,emotion1,emotion2,testname):
    data = (pd.concat([x[emotion1],x[emotion2],y],axis=1))
    print(data.head())
    #sns.heatmap(data,annot=True)
   #plt.savefig(f"data/visualizations/{testname}_{emotion1}_{emotion2}.png")
    


#run_experiment(data,RandomForestRegressor(n_estimators=1000, max_depth=10,n_jobs=3, random_state=0,verbose=True),normalized_predictor_popularity_process,{"mse":mean_squared_error},"test5")