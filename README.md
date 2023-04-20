# Energy_Consumption_Buildings

## Objective
Prediction of the energy consumption of buildings with an sklearn pipeline

## Presentation of the data
The train [database]((data/description.txt)) shape is 1804 x 19. It gathers the following variables for buildings in Boston, USA. An exhaustive description of the database is avaiable [here](data/description.txt). The target variable is the energy used by a building, called 'SiteEnergyUse_kBtu'.

## Modeling process 
#### Preprocessing 
It consists in four steps:  

- Dealing with outliers :  

- Processing NaN values : we first compute the percentage of NaN per columns. When the % of missing values is below 10%, we delete rows with NaN. When the % of missing values is between 10% and 50%, we impute missing data by the median for numeric feature or the most frequent category for categorical features. If the % of missing values is greater than 50%, we delete the feature. 

- Get dummies from string variables : we use get_dummy function from pandas

- Normalizing numerical features : we normlize data by applying the following formula for a given numerical variable x : x - x.mean()) / x.std().  

#### Feature selection
In order to avoid overfitting, we compute the Mutual Information between features and the target variable. Mutual information (MI) between two random variables is a non-negative value, which measures the dependency between the variables. It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency. More information are available [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html). We keep only the variables bringing more than 5% of information. Our database of X_train data reaches 1798 rows for 9 columns.
  
#### Cross validation
Since we only have ~1800 rows from the training set, there is a need to train through [cross validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation). To avoid overfitting, it is common practice to hold out a portion of the data as a test set and to use cross-validation. Cross-validation involves partitioning the training data into k smaller sets and training the model on k-1 sets, while using the remaining set as a validation set to compute a performance measure (cf. graph below). The average of the performance measures is reported as the final performance measure.

![](img/grid_search_cross_validation.png)

#### Models 
We then use various ML models from Sklearn packages, namely: DummyRegressor, LinearRegression, Ridge, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor, and XGBRegressor.

#### Metrics: 
For each model, we compute the cross_val_score (5 splitting) and the score on train data. Hence one could see the overfitting by comparing both scores. We also compute the variance of cross validation, to ensure the consistency of the scores. 

## Results: 
We find the XGBoost to be the best estimator with a average cross validation score of 96%. We then optimized the hyperparameters of the XGBoost with GridSearchCV to predict the target variable. 

## The repo :
  
main.py  
  
Data/  
├── links.txt  
  
lib/  
├── __init__.py  
├── preprocessing.py   
├── model.py  


## Contact: 
If you have any suggestions/remarks, feel free to contact: florentin.lavo@gmail.com
