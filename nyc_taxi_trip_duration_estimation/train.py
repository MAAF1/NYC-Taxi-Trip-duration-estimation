from data_helper import get_data
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
import numpy as np
from saving_json import save_json_with_numpy
import pickle
import matplotlib.pyplot as plt
import os
kfolds = 5
random_seed = 33

def rf_hyper(X_train,y_train,X_test,y_test):
    model_name = "Random_forest"
    rf_grid = {
    'n_estimators': [100, 130],
    'max_depth': [ 10, 20],
    'max_features': [0.5, 1.0]}
    kf = KFold(n_splits=kfolds,random_state = random_seed,shuffle=True)
    search_rf = GridSearchCV(RandomForestRegressor(), rf_grid, verbose = 2, scoring = 'r2', cv = kf)
    search_rf.fit(X_train,y_train)
    rf_best_params = search_rf.best_params_
    pred_rf_train = search_rf.predict(X_train)
    rf_train_score = r2_score(y_train, pred_rf_train)
    pred_rf_test = search_rf.best_estimator_.predict(X_test)
    rf_test_score = r2_score(y_test,pred_rf_test)
    print(f"for best parameters are {rf_best_params}\ntrain score is : {rf_train_score}\ntest score is {rf_test_score}")
    return model_name, rf_best_params, rf_train_score, rf_test_score
    pass

def xgb_hyper(X_train,y_train,X_test,y_test):
    model_name = "Xgboost"
    xgb_grid = {
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200,  250],
        'gamma': [0, 0.25, 1.0],  
    }
    kf = KFold(n_splits=kfolds, random_state=random_seed, shuffle=True)
    search_xgb = GridSearchCV(xgb.XGBRegressor(), xgb_grid, verbose=2, scoring='r2', cv=kf)
    search_xgb.fit(X_train, y_train)
    xgb_best_params = search_xgb.best_params_
    pred_train_xgb = search_xgb.predict(X_train)
    xgb_train_score = r2_score(y_train,pred_train_xgb)
    pred_test_xgb = search_xgb.best_estimator_.predict(X_test)
    xgb_test_score = r2_score(y_test, pred_test_xgb)

    print(f"Best parameters are: {xgb_best_params}\nTrain score is: {xgb_train_score}\nTest score is: {xgb_test_score}")
    
    return model_name, xgb_best_params, xgb_train_score, xgb_test_score

    

def visualize(xgb_train_score, xgb_test_score, rf_train_score, rf_test_score):
   
    
    fig, ax = plt.subplots(figsize=(10, 6))

    
    width = 0.35
    x = np.arange(2)

    bars_xgb = ax.bar(x - width/2, [xgb_train_score, xgb_test_score], width, label='XGBoost')
    bars_rf = ax.bar(x + width/2, [rf_train_score, rf_test_score], width, label='Random Forest')

    ax.set_title('Comparison of Model Performances')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(['Train', 'Test'])
    

    ax.legend()


    def add_value_on_top(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')

    add_value_on_top(bars_xgb)
    add_value_on_top(bars_rf)

    # Show the plot
    plt.tight_layout()
    plt.show()


def save_to_json(model_name, train_score, test_score, best_params):
    filename = 'tuning_selection.json'
    data = {'model': model_name,
            'train_score': train_score,
            
            'test_score' : test_score,
            'best_parameters': best_params}
            

    save_json_with_numpy(data, filename)

def fitting_final_model(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(gamma = 0, n_estimators = 300, max_depth = 10, learning_rate = 0.1 )
    model.fit(X_train, y_train, verbose = 2)
    pred_train = model.predict(X_train)
    train_score_r2 = r2_score(y_train, pred_train)
    train_score_mse = mean_squared_error(y_train, pred_train)
    pred_test= model.predict(X_test)
    test_score_r2 = r2_score(y_test, pred_test)
    test_score_mse = mean_squared_error(y_test, pred_test)
    print(f"train scores  \n1 - r2 : {train_score_r2}\n2 - MSE : {train_score_mse}\ntest scores \n1- r2 : {test_score_r2}\n2- MSE : {test_score_mse}")
    with open('xgb_train_084_test_076_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully!")
if __name__ == "__main__":

    X_t, y_t = get_data(r'nyc_taxi_trip_duration_estimation/datasets/train.csv')
    X_v, y_v = get_data(r"nyc_taxi_trip_duration_estimation/datasets/val.csv")
    X , y  = np.r_[X_t,X_v], np.r_[y_t, y_v]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)
    fitting_final_model(X_train, y_train, X_test, y_test)
    
    
    
    
    
    #model, best_params, train_score, test_score = rf_hyper(X_train, y_train, X_test, y_test)
    #rf_tr, rf_tst = train_score, test_score
    #save_to_json(model, train_score, test_score, best_params)
    #model, best_params, train_score, test_score = xgb_hyper(X_train, y_train, X_test, y_test)
    #xgb_tr = train_score
    #xgb_ts = test_score
    #save_to_json(model, train_score, test_score, best_params)
    #visualize(xgb_tr, xgb_ts, rf_tr, rf_tst)


