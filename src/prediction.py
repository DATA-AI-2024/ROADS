import xgboost as xgb
import shap
from sklearn.cluster import KMeans
import pickle
import requests
import time
import pandas as pd
import numpy as np
import holidays
import configparser
import os
import json

config = configparser.ConfigParser()
config.read('../scripts/config_train.ini')

SEED = int(config["GENERAL"]["seed"])
CLUSTER_NUM = int(config["TRAIN"]["cluster_num"])
HOUR_INTERVAL = int(config["TRAIN"]["hour_interval"])
FLOAT_COLUMNS = eval(config["TRAIN"]["float_columns"])
CATEGORICAL_COLUMNS = eval(config["TRAIN"]["categorical_columns"])
TARGET_COLUMN = eval(config["TRAIN"]["target_column"])
TRAIN_COLUMNS = FLOAT_COLUMNS + CATEGORICAL_COLUMNS
TRAIN_FILE = config["TRAIN"]["train_file"]
TEST_FILE = config["TRAIN"]["test_file"]
WEATHER_FILE = config["TRAIN"]["weather_file"]
WEATHER_API = config["TRAIN"]["weather_api"]
SAVE_PATH = config["TRAIN"]["save_path"]
CLUSTERTING_MODEL_PATH = config["TRAIN"]["clustering_model_path"]
CLUSTER_FEATURES_PATH = config["TRAIN"]["cluster_features_path"]
REMAINING_CLUSTERS_PATH = config["TRAIN"]["remaining_clusters_path"]
XGB_MODEL_PATH = config["TRAIN"]["xgb_model_path"]
EXPLAINER_PATH = config["TRAIN"]["explainer_path"]
DATA_PATH = config["TRAIN"]["data_path"]


def __init__():
    global kmeans, remaining_clusters, model, explainer, weather_API, cluster_num, cluster_centers, train_columns, test_data, remaining_cluster, cluster_features

    with open(f"{SAVE_PATH}/{CLUSTERTING_MODEL_PATH}", 'rb') as f:
        kmeans = pickle.load(f)

    with open(f"{SAVE_PATH}/{CLUSTER_FEATURES_PATH}", "r") as f:
        cluster_features = json.load(f)

    with open(f"{SAVE_PATH}/{REMAINING_CLUSTERS_PATH}", 'rb') as f:
        remaining_clusters = pickle.load(f)

    with open(f"{SAVE_PATH}/{XGB_MODEL_PATH}", 'rb') as f:
        model = pickle.load(f)

    with open(f"{SAVE_PATH}/{EXPLAINER_PATH}", 'rb') as f:
        explainer = pickle.load(f)
    
    test_data = pd.read_csv(WEATHER_FILE)


def get_weather(year, month, day, hour):
    time_str = f'{year}{month}{day}{hour}{str(0).zfill(2)}'
    url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php?tm1={time_str}&stn=133&help=1&authKey={weather_API}"

    response = requests.get(url)
    raw_data = [i for i in response.text.split("\n")[-3].split(' ') if i!='']
    WS, TEMP, HUMI, RN = raw_data[3], raw_data[11], raw_data[13], raw_data[15]
    return float(WS), float(TEMP), float(HUMI), float(RN)


def predict(temp_time = None, test = False):
    if 'kmeans' not in globals():
        __init__()

    result = []
    if not test:
        temp_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        year, month, day, hour = int(temp_time[:4]), int(temp_time[4:6]), int(temp_time[6:8]), int(temp_time[8:10])
        weekday = pd.to_datetime(f'{year}-{month}-{day}').weekday()
        WS, TEMP, HUMI, RN = get_weather(year, month, day, hour)
        holiday = 1 if f'{year}{str(month).zfill(2)}{day}' in holidays.KR() else 0
    else:
        year, month, day, hour = int(temp_time[:4]), int(temp_time[4:6]), int(temp_time[6:8]), int(temp_time[8:10])
        weekday = pd.to_datetime(f'{year}-{month}-{day}').weekday()
        WS, TEMP, HUMI, RN = test_data[(test_data['datetime'] == int(temp_time))][['WS', 'TEMP', 'HUMI', 'RN']].values[0]
        holiday = 1 if f'{year}{str(month).zfill(2)}{day}' in holidays.KR() else 0

    # Use remaining_clusters for creating input data
    input_data = pd.DataFrame({
        'x_axis': [kmeans.cluster_centers_[i][0] for i in remaining_clusters],
        'y_axis': [kmeans.cluster_centers_[i][1] for i in remaining_clusters],
        'hour': hour,
        'day': day,
        'weekday': weekday,
        'month': month,
        'year': year,
        'WS': WS,
        'TEMP': TEMP,
        'HUMI': HUMI,
        'RN': RN,
        'cluster': remaining_clusters,
        'holiday': holiday
    })

    # Add the cluster features
    for feature in ["sum_drinks", "sum_hospitals", "sum_hotels", "sum_drinks_area", "sum_hospital_rooms"]:
        input_data[feature] = input_data['cluster'].map(lambda c: cluster_features[str(c)][feature])

    # Ensure all required columns are present
    for col in TRAIN_COLUMNS:
        if col not in input_data.columns:
            input_data[col] = 0  # Initialize with 0 or appropriate default value

    # Make predictions using the trained model
    predictions = model.predict(input_data[TRAIN_COLUMNS])

    # Calculate SHAP values
    # shap_values = explainer(input_data[TRAIN_COLUMNS])

    # Prepare the result
    for i, pred in enumerate(predictions):
        cluster = remaining_clusters[i]
        prediction = pred.item()
        
        # Get the absolute SHAP values for this prediction
        # abs_shap_values = np.abs(shap_values.values[i])
        
        # Find the index of the feature with the highest absolute SHAP value
        # top_feature_index = np.argmax(abs_shap_values)
        
        # Get the name of the top feature
        # top_feature = TRAIN_COLUMNS[top_feature_index]
        top_feature = "엘렐레"
        
        result.append((cluster, prediction, top_feature))

    return result


if __name__ == '__main__':
    values = predict(None, False)