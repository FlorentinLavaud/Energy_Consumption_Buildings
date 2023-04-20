"""
lib/model.py : Model functions
"""
import pandas as pd
import rich 
from rich.pretty import pprint
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GridSearchCV

def features_importance(df):
    """
    find the importance of the features
    """
    return df

def features_selection(df, threshold):
    """
    select the features
    """
    return df

def train_models(df, target, models):
    """
    train the models
    """
    return df

def optimize_models(df, target, models):
    """
    optimize the models
    """
    return df

