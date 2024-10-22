import configparser
import pickle
import json
from typing import Callable, List, Dict, Optional
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import requests
import time
import holidays
import math


clusters = []


class Matrix:
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
    
        
class Taxi:
    def __init__(self, name: str, x_axis: float, y_axis: float, status: str):
        self.name = name
        self.x_axis, self.y_axis = x_axis, y_axis
        self.driving_time = 0
        self.passengerless_time = 0
        self.to_passenger_time = 0
        self.to_destination_time = 0
        self.status = status  # "waiting", "to_passenger", "to_destination", "to_cluster", "resting", "disconnected"
        self.earnings = 0


    def calculate_distance(self, cluster: 'Cluster') -> float:
        return math.sqrt(
            (cluster.x_axis - self.x_axis) ** 2 + (cluster.y_axis - self.y_axis) ** 2
        )

    def safe_division(n, d, default=0):
        return n / d if d != 0 else default

    def choose_cluster_matrix(self, assignments):
        global temp_time, global_last_updated_time
        updated = False

        if temp_time.minute != global_last_updated_time.minute:
            for cluster in clusters:
                cluster.update_nearby_taxis(
                    [self] + observer.moving_taxis + observer.waiting_taxis, 0.1
                )
                cluster.update_competition()
                updated = True

        if (global_last_updated_time is None or global_last_updated_time.hour != temp_time.hour):
            temp_time_str = temp_time.strftime("%Y%m%d%H%M")
            temp_time_str = temp_time_str[:-2] + "00"
            for cluster in clusters:
                cluster.update_prediction(temp_time_str)
            
            cluster.last_updated_time = temp_time
            updated = True
        
        if updated:
            global_last_updated_time = temp_time

        return assignments[self.name]


class Cluster:
    global temp_time, global_last_updated_time

    def __init__(self, x_axis: float, y_axis: float, name: str):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.name = name
        self.predicted_demand = 0.0
        self.predicted_reason = ""
        self.nearby_taxis: List[Taxi] = []
        self.competition = 0.0
        self.features = cluster_features.get(str(name), {})  
        self.last_updated_time = None


    def update_prediction(self, time_str: str):
        predictions = predict(time_str)
        for pred in predictions:
            if pred[0] == int(self.name):
                self.predicted_demand = pred[1]
        self.predicted_reason = pred[2]


    def update_nearby_taxis(self, all_taxis: List[Taxi], max_distance: float):
        self.nearby_taxis = [
            taxi
            for taxi in all_taxis
            if self.calculate_distance(taxi) <= max_distance
            and taxi.status not in ["to_passenger", "to_destination"]
        ]

    def update_competition(self):
        if self.predicted_demand > 0 and len(self.nearby_taxis) > 0:
            self.competition = len(self.nearby_taxis) / self.predicted_demand
        else:
            self.competition = 0.0

    def calculate_distance(self, taxi: "Taxi") -> float:
        return math.sqrt(
            (self.x_axis - taxi.x_axis) ** 2 + (self.y_axis - taxi.y_axis) ** 2
        )
    
    def return_info(self):
        return {'id': self.name, 'lon': float(self.x_axis), 'lat': float(self.y_axis), 'predicted_demand': self.predicted_demand, 'predicted_reason': self.predicted_reason, 'competition': self.competition}

class Observer:
    def __init__(self):
        self.moving_taxis: Dict[Taxi] = {}
        self.waiting_taxis: Dict[Taxi] = {}
        self.resting_taxis: Dict[Taxi] = {}
        self.available_taxis: Dict[Taxi] = {}
        self.distance_matrix = None
        self.competition_matrix = None
        self.demand_matrix = None
        self.assignments = {}

    # Initialize matrices for optimal assignment
    def create_assignment_matrices(self, taxis, clusters):
        n_taxis = len(taxis)
        n_clusters = len(clusters)

        distance_matrix = np.zeros((n_taxis, n_clusters))
        competition_matrix = np.zeros((n_taxis, n_clusters))
        demand_matrix = np.zeros((n_taxis, n_clusters))

        for i, taxi in enumerate(taxis):
            for j, cluster in enumerate(clusters):
                distance_matrix[i, j] = taxi.calculate_distance(cluster)
                competition_matrix[i, j] = max(
                    cluster.competition, 0.1
                )  # Avoid division by zero
                demand_matrix[i, j] = cluster.predicted_demand

        return (
            Matrix(distance_matrix),
            Matrix(competition_matrix),
            Matrix(demand_matrix),
        )

    def optimal_cluster_assignment(
        self, taxi, taxis, clusters, distance_matrix, competition_matrix, demand_matrix
    ):
        # global distance_rate, competition_rate, demand_rate
        n_taxis = len(taxis)
        n_clusters = len(clusters)

        # Normalize matrices
        norm_distance = distance_matrix.matrix / np.max(distance_matrix.matrix)
        norm_competition = competition_matrix.matrix / np.max(competition_matrix.matrix)
        norm_demand = demand_matrix.matrix / np.max(demand_matrix.matrix)

        # Create cost matrix (you can adjust weights here)
        cost_matrix = (
            distance_rate * norm_distance
            + competition_rate * norm_competition
            + demand_rate * (1 - norm_demand)
        )  # Invert demand because higher demand is better

        # If there are more taxis than clusters, we need to create dummy clusters
        if n_taxis > n_clusters:
            dummy_clusters = n_taxis - n_clusters
            dummy_cost = np.mean(
                cost_matrix
            )  # Use mean cost as the cost for dummy clusters
            cost_matrix = np.pad(
                cost_matrix,
                ((0, 0), (0, dummy_clusters)),
                mode="constant",
                constant_values=dummy_cost,
            )

        # Use linear_sum_assignment to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create a dictionary mapping taxis to their assigned clusters
        assignments = {}
        for i, j in zip(row_ind, col_ind):
            if j < n_clusters:  # It's a real cluster
                assignments[taxis[i].name] = clusters[j]
            else:  # It's a dummy cluster, assign to the best real cluster
                best_cluster_index = np.argmin(cost_matrix[i, :n_clusters])
                assignments[taxis[i].name] = clusters[best_cluster_index]
        
        self.assignments[taxi.name] = assignments[taxi.name]
        return assignments

    def set_taxi(self, taxi: Taxi):
        global updated
        start_hour = time.localtime(time.time()).tm_hour
        if taxi.name not in self.available_taxis:
            self.available_taxis[taxi.name] = taxi.status
            taxis[taxi.name] = taxi

        else:
            if taxi.status == "disconnected":
                if self.available_taxis[taxi.name] == "waiting":
                    del self.waiting_taxis[taxi.name]
                elif self.available_taxis[taxi.name] == "resting":
                    del self.resting_taxis[taxi.name]
                else:
                    del self.moving_taxis[taxi.name]
                del self.available_taxis[taxi.name]
                del taxis[taxi.name]
                return
            
            if self.available_taxis[taxi.name] == taxi.status:
                return
            
            if self.available_taxis[taxi.name] == "waiting":
                del self.waiting_taxis[taxi.name]
            elif self.available_taxis[taxi.name] == "resting":
                del self.resting_taxis[taxi.name]
                del taxis[taxi.name]
            else:
                del self.moving_taxis[taxi.name]

            if taxi.status == "waiting":
                self.waiting_taxis[taxi.name] = taxi
                self.available_taxis[taxi.name] = taxi.status
            elif taxi.status == "resting":
                self.resting_taxis[taxi.name] = taxi
                self.available_taxis[taxi.name] = taxi.status
            else:
                self.moving_taxis[taxi.name] = taxi
                self.available_taxis[taxi.name] = taxi.status
        
        taxi_list = list(taxis.values())
        if start_hour != time.localtime(time.time()).tm_hour:
            update_prediction_matrix()
        update_distance_matrix()

        observer.distance_matrix, observer.competition_matrix, observer.demand_matrix = observer.create_assignment_matrices(taxi_list, clusters)
        observer.optimal_cluster_assignment(taxi_list, taxis, clusters, observer.distance_matrix, observer.competition_matrix, observer.demand_matrix)
        del self.available_taxis[taxi.name]
        if assign_callback is not None:
            assign_callback()

t = ['월', '화', '수', '목', '금', '토', '일']
temp_time = time.localtime(time.time()).tm_wday
weekday = t[temp_time]
two_reasons= {('hour', 'weekday'): f"{weekday}요일 이 시간대에는 수요가 많아요.",
              ('hour', 'holiday'): f"휴일 이 시간대에는 수요가 많아요.",
            ('hour', 'sum_drinks'): f"이 시간대에 주점이 많아 수요가 많아요.",
            ('hour', 'sum_hospitals'): f"이 시간대에 병원이 많아 수요가 많아요.",
            ('hour', 'sum_hotels'): f"이 시간대에 숙박업소가 많아 수요가 많아요."
            }
two_reasons_order = [('hour', 'sum_drinks'), ('hour', 'sum_hospitals'), ('hour', 'sum_hotels'), ('hour', 'holiday'),('hour', 'weekday')]
one_reasons = {('hour'): "이 시간대에 수요가 많아요.",
               ('weekday'): f"{weekday}요일에 수요가 많아요.",
                ('holiday'): "휴일이어서 수요가 많아요.",
                ('sum_drinks'): "주점이 많아 수요가 많아요.",
                ('sum_hospitals'): "병원이 많아 수요가 많아요.",
                ('sum_hotels'): "숙박업소가 많아 수요가 많아요.",
                ('day'): "이 날짜에 수요가 많아요.",
                ('TEMP'): "이 날씨에 수요가 많아요.",
                ('HUMI'): "습해서 수요가 많아요.",
                ('WS'): "이 날씨에 수요가 많아요.",
                ('RN'): "비가 와서 수요가 많아요."
               }
one_reasons_order = ['sum_drinks', 'sum_hospitals', 'sum_hotels', 'RN', 'TEMP', 'HUMI', 'WS', 'holiday', 'weekday', 'hour', 'day']

def get_weather(year, month, day, hour):
    time_str = f'{year}{month}{day}{hour}{str(0).zfill(2)}'
    url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php?tm1={time_str}&stn=133&help=1&authKey={weather_API}"

    response = requests.get(url)
    raw_data = [i for i in response.text.split("\n")[-3].split(' ') if i!='']
    WS, TEMP, HUMI, RN = raw_data[3], raw_data[11], raw_data[13], raw_data[15]
    return float(WS), float(TEMP), float(HUMI), float(RN)


def predict(temp_time = None):
    result = []
    temp_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    year, month, day, hour = int(temp_time[:4]), int(temp_time[4:6]), int(temp_time[6:8]), int(temp_time[8:10])
    weekday = pd.to_datetime(f'{year}-{month}-{day}').weekday()
    WS, TEMP, HUMI, RN = get_weather(year, month, day, hour)
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
    for col in train_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Initialize with 0 or appropriate default value

    # Make predictions using the trained model
    predictions = model.predict(input_data[train_columns])

    # Calculate SHAP values
    # shap_values = explainer(input_data[train_columns])

    # Prepare the result
        # Calculate SHAP values
    shap_values = explainer(input_data[train_columns])
    for i, pred in enumerate(predictions):
        cluster = remaining_clusters[i]
        prediction = pred.item()
        
        # Get the absolute SHAP values for this prediction
        abs_shap_values = np.abs(shap_values.values[i])
        
        # Find the index of the feature with the highest absolute SHAP value
        top_feature_index = np.argmax(abs_shap_values)
        
        # Get the name of the top feature
        top_feature = train_columns[top_feature_index]
        # Get the name of the second top feature
        second_top_feature = train_columns[np.argsort(abs_shap_values)[-2]]
        third_top_feature = train_columns[np.argsort(abs_shap_values)[-3]]
        reason = [top_feature, second_top_feature, third_top_feature]

        reason_found = False

        for key in two_reasons_order:
            if not set(key) - set(reason):
                reason = two_reasons[key]
                reason_found = True
                break

        if not reason_found:
            for key in one_reasons_order:
                if key in set(reason):
                    reason = one_reasons[key]
                    break
        result.append((cluster, prediction, reason))

    return result

def update_prediction_matrix():
    temp_time_str = time.strftime('%Y-%m-%d%H:%M:%S', time.localtime(time.time()))
    temp_time_str = temp_time_str[:-2] + "00"
    predictions = predict(temp_time_str)

    for cluster in clusters:
        for pred in predictions:
            if pred[0] == int(cluster.name):
                cluster.predicted_demand = pred[1]
                cluster.predicted_reason = pred[2]
    
    if predict_callback is not None:
        predict_callback()


def update_distance_matrix():
    for cluster in clusters:
        cluster.update_nearby_taxis(
            taxis, 0.1
        )
        cluster.update_competition()


initialize_callback: Optional[Callable] = None
is_initialized = False

def set_initialize_callback(callback: Callable):
    global initialize_callback
    if is_initialized:
        callback()
    else:
        initialize_callback = callback

def initialize(): 
    global kmeans, clusters, is_initialized, cluster_features, remaining_clusters, model, explainer, distance_rate, competition_rate, demand_rate, weather_API, train_columns, observer

    config = configparser.ConfigParser()
    config.read('server.ini')
    weather_API= config["TRAIN"]["weather_api"]
    save_path = config["TRAIN"]["save_path"]
    float_columns = eval(config["TRAIN"]["float_columns"])
    categorical_columns = eval(config["TRAIN"]["categorical_columns"])
    train_columns = float_columns + categorical_columns
    clustering_model_path = config["TRAIN"]["clustering_model_path"]
    cluster_features_path = config["TRAIN"]["cluster_features_path"]
    remaining_clusters_path = config["TRAIN"]["remaining_clusters_path"]
    xgb_model_path = config["TRAIN"]["xgb_model_path"]
    explainer_path = config["TRAIN"]["explainer_path"]
    distance_rate = float(config["SIMULATION"]["distance_rate"])
    competition_rate = float(config["SIMULATION"]["competition_rate"])
    demand_rate = float(config["SIMULATION"]["demand_rate"])

    with open(f"{save_path}/{clustering_model_path}", "rb") as f:
        kmeans = pickle.load(f)

    with open(f"{save_path}/{cluster_features_path}", "r") as f:    
        cluster_features = json.load(f)

    with open(f"{save_path}/{remaining_clusters_path}", "rb") as f:
        remaining_clusters = pickle.load(f)

    with open(f"{save_path}/{xgb_model_path}", "rb") as f:
        model = pickle.load(f)

    with open(f"{save_path}/{explainer_path}", "rb") as f:
        explainer = pickle.load(f)

    observer = Observer()

    is_initialized = True
    if initialize_callback: initialize_callback()

    clusters = [
        Cluster(kmeans.cluster_centers_[i][0], kmeans.cluster_centers_[i][1], i)
        for i in remaining_clusters
    ]

    update_prediction_matrix()

    return kmeans, clusters, cluster_features, remaining_clusters, model, explainer, distance_rate, competition_rate, demand_rate, weather_API, train_columns, observer

def return_model_values():
    return kmeans, clusters, cluster_features, remaining_clusters, model, explainer, distance_rate, competition_rate, demand_rate, weather_API, train_columns, observer
    

assign_callback: Optional[Callable] = None
predict_callback: Optional[Callable] = None


def set_assign_callback(callback):
    global assign_callback
    assign_callback = callback


def set_predict_callback(callback):
    global predict_callback
    predict_callback = callback
    # 이미 predict가 이전에 실행되었으면 콜백을 바로 실행
    if clusters[0].predicted_demand != 0.0:
        callback()

kmeans, clusters, cluster_features, remaining_clusters, model, explainer, distance_rate, competition_rate, demand_rate, weather_API, train_columns, observer = initialize()


taxis = {}


if predict_callback is not None:
    predict_callback()
