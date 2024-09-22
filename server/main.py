import xgboost as xgb # 2.1.1
import shap # 0.46.0
from sklearn.cluster import KMeans
import pickle
import requests
import time
import pandas as pd
import holidays # 0.56
import configparser


def __init__():
    global kmeans, model, explainer, weather_API, cluster_num, cluster_centers, train_columns

    with open('../models/kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)

    # load the xgb model
    with open('../models/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # load the explainer
    with open('../models/explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    
    config = configparser.ConfigParser()
    config.read('config.ini')

    cluster_num = int(config['DEFAULT']['cluster_num'])
    train_columns = eval(config['DEFAULT']['train_columns'])
    weather_API = config['DEFAULT']['weather_API']
    cluster_centers = kmeans.cluster_centers_


def get_cluster(x,y):
    return kmeans.predict([[x,y]])[0]
    


def get_weather(year, month, day, hour, minute):
    time_str = f'{year}{month}{day}{hour}{minute}'
    url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php?tm1={time_str}&stn=133&help=1&authKey={weather_API}"

    response = requests.get(url)
    raw_data = [i for i in response.text.split("\n")[-3].split(' ')if i!='']
    WS, TEMP, HUMI, RN = raw_data[3], raw_data[11], raw_data[13], raw_data[15]
    try:
        return float(WS), float(TEMP), float(HUMI), float(RN)
    except Exception as e:
        print(e)
        print(raw_data)
        raise e


def predict():
    # check if init has been called
    result = []
    if 'kmeans' not in globals():
        # print error message and init
        print("Initializing...")
        __init__()
    temp_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    year, month, day, hour, minute = int(temp_time[:4]), int(temp_time[4:6]), int(temp_time[6:8]), int(temp_time[8:10]), int(temp_time[10:12])
    weekday = pd.to_datetime(f'{year}-{month}-{day}').weekday()
    minute = minute // 10 * 10
    WS, TEMP, HUMI, RN = get_weather(year, month, day, hour, minute)
    holiday = 1 if f'{year}{str(month).zfill(2)}{day}' in holidays.KR() else 0
    for i in range(cluster_num):
        x_axis, y_axis = cluster_centers[i]
        new_data = pd.DataFrame([[x_axis, y_axis, minute, hour, day, weekday, month, year, WS, TEMP, HUMI, RN, i, holiday]], columns=['x_axis', 'y_axis', 'minute', 'hour', 'day', 'weekday', 'month', 'year', 'WS', 'TEMP', 'HUMI', 'RN', 'cluster', 'holiday'])
        pred = model.predict(new_data[train_columns])
        shap_values = explainer(new_data[train_columns])
        result.append((i, x_axis, y_axis, pred[0].item(), train_columns[shap_values[0].values.argmax()]))

    return result

if __name__ == '__main__':
    values = predict()