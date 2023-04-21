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

def features_importance(df1, df2):
  """
  Compute mutual information between the features (df1) and the target (df2)
  Returns the value of Mutual Information and a plot
  """
  
  mutual_info = mutual_info_regression(df1, df2)
  mutual_info = pd.Series(mutual_info)
  mutual_info.index = df1.columns
  value = mutual_info.sort_values(ascending=False)
  return value

#mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))

def features_selection(df1,df2,threshold):
    """
    select the features
    """
    mutual_info = mutual_info_regression(df1, df2)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X_train.columns
    mutual_info.sort_values(ascending=False)
    
    mutual_info_selected = mutual_info[mutual_info > threshold]

    df1 = df1[[col for col in df1.columns if col in mutual_info_selected]]

    return df1


def train_models(df1, df2, models):
    """
    train the models
    """
    
    results = {
    "train": {}, 
    "cross_validation_mean": {},
    "cross_validation_variance": {}
    }

    for estimator in tqdm(estimators):
        model = estimator
        model_name = model.__class__.__name__
        
        model.fit(X_train, y_train)
        
        cv_scores = cross_val_score(estimator, X_train, y_train, cv=5)
        results["cross_validation_mean"][model_name] = cv_scores.mean()
        results["cross_validation_variance"][model_name] = cv_scores.var()
        train_score = model.score(X_train, y_train)
        results["train"][model_name] = train_score
    
    return pprint(results)

