# Energy_Consumption_Buildings

## Objective
Prediction of the energy consumption of buildings with an sklearn pipeline

## Presentation of the data
The train [database]((data/description.txt)) shape is 1804 x 19. It gathers the following variables for buildings in Boston, USA. An exhaustive description of the database is avaiable [here](data/description.txt).

## Modeling process 
#### Preprocessing :  
It consists in four steps:  

- Dealing with outliers   

- Processing NaN values

- Get dummies from string variables 

- Normalizing numerical features   

#### Feature selection
In order to avoid overfitting, we compute the Mutual Information between features and the target variable. We keep only the variables bringing more than 5% of information. Our database of X_train data reaches 1798 rows for 9 columns.
  
#### Cross validation
Since we only have ~1800 rows, there is a need to train through [cross validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation).

#### Scores

## Results 

## The repo :
  
main.py  
  
Data/  
├── links.txt  
  
lib/  
├── __init__.py  
├── preprocessing.py   
├── model.py  


## Contact: 
florentin.lavo@gmail.com
