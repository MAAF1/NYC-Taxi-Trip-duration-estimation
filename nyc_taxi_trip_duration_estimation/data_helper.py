import numpy as np
import pandas as pd
from geopy.distance import geodesic
import math
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Difference in longitude
    delta_lon = lon2_rad - lon1_rad
    
    # Apply the Haversine formula
    y = math.sin(delta_lon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
    
    # Calculate the initial bearing
    theta = math.atan2(y, x)
    
    # Convert to degrees and ensure it's between 0° and 360°
    bearing = math.degrees(theta)
    if bearing < 0:
        bearing += 360
    
    return bearing
def prepare_data(name):
    data_frame = pd.read_csv(name)
    data_frame['pickup_datetime'] = pd.to_datetime(data_frame['pickup_datetime'])
    data_frame['year'] = data_frame['pickup_datetime'].dt.year
    data_frame['month'] = data_frame['pickup_datetime'].dt.month
    data_frame['day'] = data_frame['pickup_datetime'].dt.day_name()
    data_frame['hour'] = data_frame['pickup_datetime'].dt.hour
    distance = []
    for index in data_frame['pickup_latitude'].index :
      distance.append(geodesic((data_frame['pickup_latitude'].iloc[index], data_frame['pickup_longitude'].iloc[index]),(data_frame['dropoff_latitude'].iloc[index], data_frame['dropoff_longitude'].iloc[index])).miles)
    data_frame['distance'] = distance
    bearing = []
    for index in data_frame['pickup_latitude'].index :
        bearing.append(calculate_bearing(data_frame['pickup_latitude'].iloc[index],data_frame['pickup_longitude'].iloc[index],data_frame['dropoff_latitude'].iloc[index],data_frame['dropoff_longitude'].iloc[index]))
    data_frame['bearing'] = bearing
    data_frame.drop(['id','pickup_datetime','year','store_and_fwd_flag'],axis=1, inplace=True)
    data  =  data_frame.drop(['trip_duration'],axis = 1)
    target = data_frame['trip_duration']
    return data, target

def get_data(data_name, encoding_option = 'one_hot'):
    data,target = prepare_data(data_name)
  
    if encoding_option == 'mapping':
        d = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
        data['day']=data['day'].map(d) 
    elif encoding_option == 'one_hot' :
        encoded_df = pd.get_dummies(data, columns=['day'], drop_first=True, sparse=True, dtype=int)
        data = encoded_df.copy()
        del encoded_df
    features = data.columns
    X = data.to_numpy()
    y = target.to_numpy()
    
    y = np.log1p(y)
    return X,y#,features
def preprocessing(X, poly_option = 0,  degree = 1, scaler_option = 'min_max'):
    if poly_option == 1:
        poly = PolynomialFeatures(degree = degree)
        X = poly.fit_transform(X)
    if scaler_option == 'min_max':
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    elif scaler_option == 'standard':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X




