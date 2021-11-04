from popularity_predictor import *

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


def plot_emotion_distribution(x,y,emotion):
    dat = x[emotion]
    graph(dat,y)

def graph(x,y):
    sns.regplot(x=x,y=y)
    plt.savefig('data/visualizations/emotion_distribution.png')

x,y = unnormalized_predictor_popularity_process(data)
plot_emotion_distribution(x,y,'Fear')