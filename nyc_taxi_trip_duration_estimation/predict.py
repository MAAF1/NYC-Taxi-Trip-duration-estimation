from data_helper import get_data
from sklearn.metrics import r2_score, mean_squared_error
import pickle

def load_model():
    with open('xgb_train_084_test_076_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def inference(model, X,y_truth):
    pred = model.predict(X)
    score = r2_score(y_truth, pred)
    error =  mean_squared_error(y_truth, pred)
    print(f"r2_score : {score}\nMSE : {error}")
if __name__ == "__main__":
    X,y = get_data(r'nyc_taxi_trip_duration_estimation/datasets/test.csv')  
    model = load_model()
    inference(model, X, y)