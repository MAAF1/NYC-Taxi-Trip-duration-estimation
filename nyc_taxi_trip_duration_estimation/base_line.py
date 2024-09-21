from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from data_helper import preprocessing,get_data
from saving_json import save_json_with_numpy


def ridge(X_train,y_train,X_test,y_test):
    model_name = 'ridge'
    model = Ridge(alpha=0.01,max_iter=5000)
    model.fit(X_train,y_train)
    train_pred = model.predict(X_train)
    train_score_r2 = r2_score(y_train, train_pred)
    train_score_mse = mean_squared_error(y_train, train_pred)
    pred_test = model.predict(X_test)
    test_score_r2 = r2_score(y_test,pred_test)
    test_score_mse = mean_squared_error(y_test,pred_test)
    print(f"ridge train mse : {train_score_mse} **** r2 : {train_score_r2}\nridge test mse : {test_score_mse} **** r2 : {test_score_r2}")
    return model_name, train_score_mse, train_score_r2, test_score_mse, test_score_r2    

def decision_tree(X_train,y_train,X_test,y_test):
    model_name = 'decision_tree'
    model = DecisionTreeRegressor(max_depth = 10, min_samples_leaf = 16, min_samples_split = 30)
    model.fit(X_train,y_train)
    train_pred = model.predict(X_train)
    train_score_r2 = r2_score(y_train, train_pred)
    train_score_mse = mean_squared_error(y_train, train_pred)
    pred_test = model.predict(X_test)
    test_score_r2 = r2_score(y_test,pred_test)
    test_score_mse = mean_squared_error(y_test,pred_test)
    print(f"decision_tree train mse : {train_score_mse} **** r2 : {train_score_r2}\ndecision_tree test mse : {test_score_mse} **** r2 : {test_score_r2}")

    return model_name, train_score_mse, train_score_r2, test_score_mse, test_score_r2 

def random_forest(X_train,y_train,X_test,y_test):
    model_name = 'random_forest'
    model = RandomForestRegressor(n_estimators=130,max_depth = 20, max_features = 1)
    model.fit(X_train,y_train)
    train_pred = model.predict(X_train)
    train_score_r2 = r2_score(y_train, train_pred)
    train_score_mse = mean_squared_error(y_train, train_pred)
    pred_test = model.predict(X_test)
    test_score_r2 = r2_score(y_test,pred_test)
    test_score_mse = mean_squared_error(y_test,pred_test)
    print(f"random_forest train mse : {train_score_mse} **** r2 : {train_score_r2}\nrandom_forest test mse : {test_score_mse} **** r2 : {test_score_r2}")

    return model_name, train_score_mse, train_score_r2, test_score_mse, test_score_r2 
def xgboost(X_train,y_train,X_test,y_test):
    model_name = 'Xgboost'
    model = xgb.XGBRegressor(n_estimators=200,max_depth = 10, max_features = 8, learning_rate = 0.01, gamma = 1)
    model.fit(X_train,y_train)
    train_pred = model.predict(X_train)
    train_score_r2 = r2_score(y_train, train_pred)
    train_score_mse = mean_squared_error(y_train, train_pred)
    pred_test = model.predict(X_test)
    test_score_r2 = r2_score(y_test,pred_test)
    test_score_mse = mean_squared_error(y_test,pred_test)
    print(f"XGBOOST train mse : {train_score_mse} **** r2 : {train_score_r2}\nXGBOOST test mse : {test_score_mse} **** r2 : {test_score_r2}")

    return model_name, train_score_mse, train_score_r2, test_score_mse, test_score_r2 
if __name__ == "__main__":


    filename = 'model_baseline_metadata.json'
    encoder = 'one_hot'
    X, y, features = get_data(r'nyc_taxi_trip_duration_estimation/datasets/split_sample/train.csv',encoder)
    X_train,X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2,shuffle=False)
    degree = 1
    scaler = 'none'
    features = np.array(features)
    
    #X_train = preprocessing(X_train,1,degree,scaler)
    #X_test = preprocessing(X_test,1,degree,scaler)
    
    model_name, train_mse, train_r2_score, test_mse, test_r2_score = xgboost(X_train,y_train,X_test,y_test)

    data = {'model': model_name,
            'train_r2_score': train_r2_score,
            'train_mse': train_mse,
            'test_r2_score' : test_r2_score,
            'test_mse' : test_mse,
            'features': features,
            'polynomial_degree' : degree,
            'encoding_option':encoder,
            'scaler_option' : scaler        
    }
    save_json_with_numpy(data, filename)