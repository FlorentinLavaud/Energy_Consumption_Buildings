"""
lib/preprocessing.py : Preprocessing functions for the data
"""
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import rich 
from rich.pretty import pprint
from tqdm import tqdm
from pandas import DataFrame

def import_selected_data(name_file, df): 
    """
    import data from drive and delete the unwanted data
    """
    return df

def outliers(df):
    """
    find the outliers in the data
    """
    return df


def outlier_removal(df):
    """
    remove the outliers from the data
    """
    return df

def na_per_column(df):
    na_counts = df.isna().sum()
    na_counts_sorted = na_counts.sort_values(ascending=False)
    na_percentages = na_counts_sorted / len(df) * 100
    result = pd.concat([na_counts_sorted, na_percentages], axis=1, keys=['Count', 'Percentage'])
    return result

def fill_na(df):
    """
    fill the na values with the mean of the column
    """
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