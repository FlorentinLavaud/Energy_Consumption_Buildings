from lib.preprocessing import calculate_outlier_percentage, remove_outlier_columns, na_per_column
from lib.preprocessing import fill_missing_values, str2dummy, scale_features, sklearn_preprocessing
from lib.model import features_importance, features_selection, train_models

import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_columns', None)

path = r'C:\Users\flore\OneDrive\Bureau\2023\Drive\_Projects'
df_train = pd.read_excel(path + '/Building_Energy_TRAIN.xlsx')

import os
path = r'C:\Users\flore\OneDrive\Bureau\2023\Drive\_Projects\Energy_Consumption_Buildings'
os.getcwd()

# Dealing with outliers
cols = ['NumberofFloors', 'PropertyGFATotal', 'PropertyGFABuilding', 'ENERGYSTARScore']

for col in cols: 
  outlier_percentage = calculate_outlier_percentage(df_train, col)
  print(f"The percentage of outliers in {col} is {outlier_percentage:.2f}%")

tmp = remove_outlier_columns(df_train, cols)
nb_cols_removed = tmp.shape[1] - df_train.shape[1]
print(f"We removed {nb_cols_removed} columns")

# Dealing with NaN 

na_per_column(df_train)

for col in tqdm(cols): 
   df_train = fill_missing_values(df_train,'ENERGYSTARScore')
   df_train = fill_missing_values(df_train,'LargestPropertyUseType')

na_per_column(df_train)

# Creating dummies

column_to_transform = ['BuildingType', 'PrimaryPropertyType', 'CouncilDistrictCode']
X_train = pd.get_dummies(df_train, columns=column_to_transform)
print(f"on vient d'ajouter {X_train.shape[1] - df_train.shape[1]} colonnes")

types = ['Multifamily Housing', 'Office', 'Parking', 'Retail Store', 
         'Services', 'Warehouse', 'Wholesale Club/Supercenter', 'Other',
         'Swimming Pool', 'Fitness Center/Health Club/Gym', 'Food Service',
            'Data Center', 'Restaurant', 'Worship Facility', 'Laboratory',
            'Hotel', 'College/University', 'Senior Care Community', 'Distribution Center']

for type in tqdm(types):
    X_train = str2dummy(X_train, 'ListOfAllPropertyUseTypes', type)

X_train['Anciennete_bat'] = 2023 - X_train['YearBuilt'] 
X_train.drop(['ListOfAllPropertyUseTypes', 'LargestPropertyUseType', 'YearBuilt'], axis=1, inplace=True)

# Scaling numerical features

numerical_vars =  ['NumberofFloors', 'PropertyGFATotal', 'PropertyGFAParking', 
                'PropertyGFABuilding', 'ENERGYSTARScore', 'SiteEnergyUse_kBtu']

for var in tqdm(numerical_vars):
    X_train = scale_features(X_train, var)

# Preparing data for sklearn

y_train, X_train = sklearn_preprocessing(X_train, 'SiteEnergyUse_kBtu')

# Feature selection

features_importance(X_train,y_train)
features_selection(X_train, y_train, 0.05)

# Training models

y_train = X_train.values
X_train = X_train.values

estimators = [
    DummyRegressor(), 
    LinearRegression(),
    Ridge(),
    KNeighborsRegressor(n_neighbors=10),
    DecisionTreeRegressor(), 
    RandomForestRegressor(),
    XGBRegressor(), 
]

train_models(X_train, y_train, estimators)

# GridSearhCV with the best model: 

xgb = XGBRegressor() 

parameters =  {'max_depth': [3, 5, 7], 
               'n_estimators': [200, 300, 400],
               'learning_rate': [0.1, 0.01, 0.05]}


model_optimized = GridSearchCV(xgb, parameters)
model_optimized.fit(X_train, y_train)