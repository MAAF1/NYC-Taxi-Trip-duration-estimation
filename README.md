# `NYC-Taxi-Trip-Duration-Estimation`
## Project Overview: 
### The project aims to create a model that predicts the trip duration of taxis in New York City using data provided in the dataset provided.
## Dataset used in this project :
### https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data
## Project implementaion:
### EDA 
#### Usage of EDA to investigate data, check records, and analyze data to get insights that help to improve model performance
### Data Cleaning
#### The data cleaning performed in this project only on features such as removal of useless features like id year ...etc.but the outlier handling on this data either using z-score or iqr techniques is not useful in fact it made bad model performance.
### Feature Engineering
#### Creation of new features like distance and bearing(direction) from and performing and trying to map features or perform encoding like "one-hot"
### Modeling 
#### The modeling life cycle starts with creation of multiple model baselines on a sample of data to get an indicator which model will perform well "might be a little biased since we choosed a random sample" then we select the best baselines based on their score. We selected Xgboost and random forest 
### Hyperparameter Tuning 
#### Usage of Grid search with cross-validation to search for best parameters of select baselines.
### Model Training 
#### We selected the best estimators we got from cross-validation and Grid search then we trained on the whole dataset the two models Xgboost and random forest 
![rf vs xgb.png](https://github.com/MAAF1/NYC-Taxi-Trip-duration-estimation/blob/main/Screenshot%20from%202024-09-21%2006-42-54.png)
#### We selected XGBoost cause it satisfies the generalization principle in ML and it's way lighter and faster than random forest.
### Model Evaluation 
#### Saved a model to use it on inference.
