"""
lib/preprocessing.py : Preprocessing functions for the data
"""
import pandas as pd 
import numpy as np
import os 
# import matplotlib as plt
# import seaborn as sns
import rich 
from rich.pretty import pprint
from tqdm import tqdm
from pandas import DataFrame

def calculate_outlier_percentage(df, column_name):
    """
    Calculates the percentage of outliers in a given column of a Pandas DataFrame,
    where an outlier is defined as a point value that falls in the top or bottom 5% of the distribution.
    """
    column_values = df[column_name].values # extract the column values as a NumPy array
    cutoff = np.percentile(column_values, [1, 99]) # calculate the 5th and 95th percentiles
    outliers = [x for x in column_values if x < cutoff[0] or x > cutoff[1]] # identify outliers
    outlier_percentage = len(outliers) / len(column_values) * 100 # calculate percentage
    return outlier_percentage

def remove_outlier_columns(df, cols):
    """
    Removes any columns from a Pandas DataFrame that have a percentage of outliers
    greater than 10%
    """
    for column_name in cols:
        outlier_percentage = calculate_outlier_percentage(df, column_name)
        if outlier_percentage > 10:
            df = df.drop(column_name, axis=1)
    return df

def na_per_column(df):
    na_counts = df.isna().sum()
    na_counts_sorted = na_counts.sort_values(ascending=False)
    na_percentages = na_counts_sorted / len(df) * 100
    result = pd.concat([na_counts_sorted, na_percentages], axis=1, keys=['Count', 'Percentage'])
    return result

def fill_missing_values(df, column_name):
    """
    Fills missing values in a Pandas DataFrame column with the median for numeric
    columns and the mode (most frequent category) for non-numeric columns.
    """
    column_values = df[column_name]
    if column_values.dtype == "object":
        df[column_name] = column_values.fillna(column_values.mode().iloc[0]) # fill missing values with mode
    else:
        df[column_name] = column_values.fillna(column_values.median()) # fill missing values with median
    return df


def str2dummy(df: DataFrame, column_name, string: str) -> DataFrame:
    """
    create dummy variables if a substring is in a string in a column
    """
    df['dummy_'+string] = [1 if string in x else 0 for x in df[column_name]]
    return df

def scale_features(df, feature):
    """
    normalize feature
    """

    df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
    return df

def unscale_features(df, feature):
    """
    remove normalization from feature
    """
    df[feature] = df[feature] * df[feature].std() + df[feature].mean()
    return df

def sklearn_preprocessing(df, target):
    """
    apply sklearn preprocessing
    to call the function : y_train, X_train = sklearn_preprocessing(X_train, 'SiteEnergyUse_kBtu')
    """
    y_train = df[target]
    X_train = df.drop([target], axis = 1)

    return y_train, X_train
