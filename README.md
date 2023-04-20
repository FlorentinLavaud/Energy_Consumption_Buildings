# Energy_Consumption_Buildings
Prediction of the energy consumption of buildings with sklearn

## Objective
Prediction of the energy consumption of buildings with an sklearn pipeline

## Presentation of the data
Database shape is 1804 x 19. It gathers the following variables for buildings in Boston, USA. A exhaustive description of the database is avaiable in data/description.txt

## Modeling process 
#### Preprocessing :  
It consists in four steps:  

- Dealing with outliers   

- Processing NaN values

- Get dummies from string variables 

- Normalizing numerical features   

#### Feature selection
In order to avoid overfitting, we calculate the mutual information between our features and the variable of interest, namely Site Energy Use. We keep only the variables bringing more than 5% of information. Our database of X-train data reaches 1798 rows for 9 columns.
  
 
#### Cross validation
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
