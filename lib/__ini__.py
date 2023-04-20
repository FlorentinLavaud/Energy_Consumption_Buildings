""" 
lib/__ini__.py: initilization file for the lib directory
"""
import pandas as pd 
pd.set_option('display.max_columns', None)

import os 
path = os.getcwd()
path_data = path+'/Building_Energy_TRAIN.xlsx'
pd.read_excel(path+'/Building_Energy_TRAIN.xlsx')
