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
Since we only have ~1800 rows, there is a need to train through [cross validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation). To avoid overfitting, it is common practice to hold out a portion of the data as a test set and to use cross-validation. Cross-validation involves partitioning the training data into k smaller sets and training the model on k-1 sets, while using the remaining set as a validation set to compute a performance measure. The average of the performance measures is reported as the final performance measure.

![](imgs/grid_search_cross_validation.png)

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
