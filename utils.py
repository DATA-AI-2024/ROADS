import shap
import pandas as pd
import requests
import time
import os
import pickle
import holidays
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
import xgboost as xgb
import folium

# API key for weather data
API = "gGryFchORUuq8hXITjVLWQ"

def get_weather(year, month, day, hour, minute):
    time_str = f"{year}{month:02d}{day:02d}{hour:02d}{minute:02d}"
    url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php?tm1={time_str}&stn=133&help=1&authKey={API}"
    response = requests.get(url)
    raw_data = [i for i in response.text.split("\n")[-3].split(' ') if i != '']
    WS, TEMP, HUMI, RN = raw_data[3], raw_data[11], raw_data[13], raw_data[15]
    return WS, TEMP, HUMI, RN

def load_and_preprocess_data():
    train = pd.read_csv('train_taxi_tims.csv', encoding='utf-8-sig', engine='python')
    test = pd.read_csv('test_taxi_tims.csv', encoding='utf-8-sig', engine='python')

    for df in [train, test]:
        df = df[(df['x_axis'] != 0) & (df['y_axis'] != 0)]
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['minute'] = df['datetime'].dt.minute // 10 * 10
        df['hour'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['year'] = df['datetime'].dt.year
        df.drop(['datetime'], axis=1, inplace=True)

    return train, test

def get_weather_data():
    if 'weather_data.csv' not in os.listdir():
        # Fetch weather data and save to CSV
        # (Code for fetching weather data)
        pass
    else:
        total_data = pd.read_csv('weather_data.csv', encoding='utf-8-sig', engine='python')
    
    total_data['datetime'] = pd.to_datetime(total_data['datetime'], format='%Y%m%d%H%M')
    total_data['hour'] = total_data['datetime'].dt.hour
    total_data['month'] = total_data['datetime'].dt.month
    total_data['day'] = total_data['datetime'].dt.day
    total_data['year'] = total_data['datetime'].dt.year
    total_data['holiday'] = total_data['datetime'].apply(lambda x: 1 if x in holidays.KR() else 0)
    total_data.drop(['datetime'], axis=1, inplace=True)
    
    return total_data

def perform_clustering(train, test):
    if 'kmeans_model.pkl' not in os.listdir():
        cluster_num = 50
        kmeans = KMeans(n_clusters=cluster_num, random_state=5).fit(train[['x_axis', 'y_axis']])
        with open('kmeans_model.pkl', 'wb') as f:
            pickle.dump(kmeans, f)
    else:
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
    
    train['cluster'] = kmeans.predict(train[['x_axis', 'y_axis']])
    test['cluster'] = kmeans.predict(test[['x_axis', 'y_axis']])
    
    return train, test, kmeans

def train_model(train):
    train_columns = ['hour', 'weekday', 'month', 'day', 'WS', 'TEMP', 'HUMI', 'RN', 'cluster', 'holiday']
    model = xgb.XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.1, random_state=5)
    model.fit(train[train_columns], train['count'])
    
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

def create_shap_explainer(model, train_columns, test):
    explainer = shap.Explainer(model, test[train_columns])
    with open('explainer.pkl', 'wb') as f:
        pickle.dump(explainer, f)
    return explainer

def create_map(kmeans, cluster_num):
    data = {
        'name': ['cluster_' + str(i) for i in range(cluster_num)],
        'latitude': [i[1] for i in kmeans.cluster_centers_],
        'longitude': [i[0] for i in kmeans.cluster_centers_]
    }
    df = pd.DataFrame(data)
    
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    for idx, row in df.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=row['name'],
            tooltip=row['name']
        ).add_to(m)
    
    m.save("map_with_markers.html")

def main():
    train, test = load_and_preprocess_data()
    weather_data = get_weather_data()
    train = pd.merge(train, weather_data, on=['year', 'month', 'day', 'hour'], how='left')
    test = pd.merge(test, weather_data, on=['year', 'month', 'day', 'hour'], how='left')
    
    train, test, kmeans = perform_clustering(train, test)
    
    train['count'] = 1
    train_count = train.groupby(['cluster', 'minute', 'hour', 'day', 'month', 'year']).count().reset_index()
    train_count = train_count[['cluster', 'minute', 'hour', 'day', 'month', 'year','count']]
    train = pd.merge(train, train_count, on=['cluster', 'minute', 'hour', 'day', 'month', 'year'], how='left')
    train['count'] = train['count_y']
    train.drop(['count_x', 'count_y'], axis=1, inplace=True)
    
    model = train_model(train)
    explainer = create_shap_explainer(model, test)
    create_map(kmeans, 50)
    
    # Prediction loop
    train_columns = ['hour', 'weekday', 'month', 'day', 'WS', 'TEMP', 'HUMI', 'RN', 'cluster', 'holiday']
    cluster_centers = kmeans.cluster_centers_
    while True:
        time.sleep(60)  # Update every minute
        temp_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        year, month, day, hour, minute = int(temp_time[:4]), int(temp_time[4:6]), int(temp_time[6:8]), int(temp_time[8:10]), int(temp_time[10:12])
        weekday = pd.to_datetime(f'{year}-{month}-{day}').weekday()
        minute = minute // 10 * 10
        WS, TEMP, HUMI, RN = get_weather(year, month, day, hour, minute)
        holiday = 1 if f'{year}{str(month).zfill(2)}{day}' in holidays.KR() else 0
        
        for i in range(len(cluster_centers)):
            x_axis, y_axis = cluster_centers[i]
            new_data = pd.DataFrame([[x_axis, y_axis, minute, hour, day, weekday, month, year, WS, TEMP, HUMI, RN, i, holiday]], 
                                    columns=['x_axis', 'y_axis', 'minute', 'hour', 'day', 'weekday', 'month', 'year', 'WS', 'TEMP', 'HUMI', 'RN', 'cluster', 'holiday'])
            new_data[['WS', 'TEMP', 'HUMI', 'RN']] = new_data[['WS', 'TEMP', 'HUMI', 'RN']].astype(float)
            pred = model.predict(new_data[train_columns])
            print(f'cluster_{i} : {pred[0]}')