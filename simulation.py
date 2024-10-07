import configparser
import json
import logging
import math
import pickle
import random
import subprocess
import time
import webbrowser
from collections import deque
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from main import predict

config = configparser.ConfigParser()
config.read('config.ini')

taxis = int(config['SIMULATION']['taxis'])
steps = int(config['SIMULATION']['steps'])
test_file = config['SIMULATION']['test_file']
rest_file = config['SIMULATION']['rest_file']
visualize = bool(config['SIMULATION']['visualize'])
alg_name = config['SIMULATION']['alg_name']
save_path = config['SIMULATION']['save_path']
name = f'{alg_name}_{taxis}_{steps}'
seed = int(config['SIMULATION']['seed'])

random.seed(seed)

logging.basicConfig(filename=f"{save_path}/{name}.log", level=logging.INFO)

with open(f'{save_path}/{name}.csv', 'w') as f:
    f.write("id,time,lon,lat,status\n")

with open('models/kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# load the xgb model
with open('models/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# load the explainer
with open('models/explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)

webhookURL = 'https://sb1031.tw3.quickconnect.to/direct/webapi/entry.cgi?api=SYNO.Chat.External&method=incoming&version=2&token=%22nX5xYyMkltc8qFEJ65OLgISHBSvxrGSLRzCdZusuB1zKy0PvbFkmCgyCuv36JE5q%22'


def send_chat(webhook, message):
    params = {
        "payload":json.dumps({"text":message})
    }
    response = requests.post(webhook, data=params, verify=False)


class Cluster:
    global temp_time, global_last_updated_time
    def __init__(self, x_axis: float, y_axis: float, name: str):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.name = name
        self.predicted_demand = 0.0
        self.predicted_reason = ""
        self.nearby_taxis: List['Taxi'] = []
        self.competition = 0.0
        self.last_updated_time = None

    def update_prediction(self, time_str: str):
        global temp_time, global_last_updated_time
        predictions = predict(time_str, True)
        for pred in predictions:
            if pred[0] == int(self.name):
                self.predicted_demand = pred[1]
                break
        self.predicted_reason = pred[2]
        self.last_updated_time = temp_time
        global_last_updated_time = temp_time

    def update_nearby_taxis(self, all_taxis: List['Taxi'], max_distance: float):
        self.nearby_taxis = [
            taxi for taxi in all_taxis
            if self.calculate_distance(taxi) <= max_distance
            and taxi.status not in ["to_passenger", "to_destination"]
        ]

    def calculate_distance(self, taxi: 'Taxi') -> float:
        return math.sqrt((self.x_axis - taxi.x_axis)**2 + (self.y_axis - taxi.y_axis)**2)

    def update_competition(self):
        if self.predicted_demand > 0 and len(self.nearby_taxis) > 0:
            self.competition = len(self.nearby_taxis) / self.predicted_demand
        else:
            self.competition = 0.0
        

class Taxi:
    def __init__(self, name: str, x_axis: float, y_axis: float, rest_times: pd.DataFrame):
        self.name = name
        self.x_axis, self.y_axis = x_axis, y_axis
        self.to_x_axis = None
        self.to_y_axis = None
        self.velocity = 0.00004475
        self.x_velocity = 0
        self.y_velocity = 0
        self.passengerless_time = 0
        self.to_passenger_time = 0
        self.to_destination_time = 0
        self.status = "waiting"  # "waiting", "to_passenger", "to_destination", "to_cluster", "resting"
        self.passenger = None
        self.earnings = 0
        self.rest_times = rest_times
        self._rest_time_index = 0

    def choose_cluster(self, alg_name: str) -> Tuple[float, float]:
        global temp_time, global_last_updated_time
        if global_last_updated_time is None or (temp_time - global_last_updated_time).total_seconds() > 60:
            temp_time_str = temp_time.strftime('%Y%m%d%H%M')
            temp_time_str = temp_time_str[:-2] + '00'
            clusters[0].update_prediction(temp_time_str)
            for cluster in clusters:
                cluster.update_nearby_taxis([self] + observer.moving_taxis + observer.waiting_taxis, 0.1)
                cluster.update_competition()
            
        global_last_updated_time = temp_time

        def safe_division(n, d, default=0):
            return n / d if d != 0 else default

        match alg_name:
            case "Cluster_Probability":
                best_cluster = max(clusters, key=lambda c: c.predicted_demand)
            case "Cluster_Probability+Distance":                
                scores = []
                for cluster in clusters:
                    distance_score = self.calculate_distance(cluster)
                    demand_score = cluster.predicted_demand
                    competition_score = max(cluster.competition, 0.1)  # Avoid division by zero
                    total_score = safe_division(demand_score, distance_score * competition_score, default=0)
                    scores.append((cluster, total_score))
                
                best_cluster = max(scores, key=lambda x: x[1])[0]
            case "Inverse_Competition":
                best_cluster = max(clusters, key=lambda c: safe_division(c.predicted_demand, max(c.competition, 0.1)))
            case "Demand_Competition_Ratio":
                best_cluster = max(clusters, key=lambda c: safe_division(c.predicted_demand, max(c.competition, 0.1)))
            case "Weighted_Score":
                w_demand, w_distance, w_competition = 0.5, 0.3, 0.2  # Adjust weights as needed
                scores = []
                max_demand = max(c.predicted_demand for c in clusters)
                max_distance = max(self.calculate_distance(c) for c in clusters)
                max_competition = max(max(c.competition, 0.1) for c in clusters)
                for cluster in clusters:
                    demand_score = safe_division(cluster.predicted_demand, max_demand)
                    distance_score = 1 - safe_division(self.calculate_distance(cluster), max_distance)
                    competition_score = 1 - safe_division(max(cluster.competition, 0.1), max_competition)
                    total_score = w_demand * demand_score + w_distance * distance_score + w_competition * competition_score
                    scores.append((cluster, total_score))
                best_cluster = max(scores, key=lambda x: x[1])[0]
            case "Time_Aware":
                current_hour = temp_time.hour
                if 6 <= current_hour < 10 or 16 <= current_hour < 20:  # Rush hours
                    best_cluster = max(clusters, key=lambda c: safe_division(c.predicted_demand, max(c.competition, 0.1) ** 0.5))
                else:  # Non-rush hours
                    best_cluster = max(clusters, key=lambda c: safe_division(c.predicted_demand, self.calculate_distance(c) * max(c.competition, 0.1)))
            case "Adaptive":
                if self.passengerless_time > 300:  # If waiting for more than 5 minutes
                    best_cluster = max(clusters, key=lambda c: safe_division(c.predicted_demand, max(c.competition, 0.1)))
                else:
                    best_cluster = max(clusters, key=lambda c: safe_division(c.predicted_demand, self.calculate_distance(c) * max(c.competition, 0.1)))

        return best_cluster.x_axis, best_cluster.y_axis

    def calculate_distance(self, cluster: Cluster) -> float:
        return math.sqrt((cluster.x_axis - self.x_axis)**2 + (cluster.y_axis - self.y_axis)**2)

    def start_move(self, to_x_axis, to_y_axis, status: str):
        self.status = status
        self.to_x_axis = to_x_axis
        self.to_y_axis = to_y_axis
        distance = math.sqrt((self.to_x_axis - self.x_axis)**2 + (self.to_y_axis - self.y_axis)**2)
        self.x_velocity = self.velocity * (self.to_x_axis - self.x_axis) / distance
        self.y_velocity = self.velocity * (self.to_y_axis - self.y_axis) / distance

    def move(self):
        self.x_axis += self.x_velocity
        self.y_axis += self.y_velocity
        match self.status:
            case "to_passenger":
                self.to_passenger_time += 1
            case "to_destination":
                self.to_destination_time += 1

    def is_at_destination(self) -> bool:
        if self.to_x_axis is None or self.to_y_axis is None:
            return False
        distance = math.sqrt((self.to_x_axis - self.x_axis)**2 + (self.to_y_axis - self.y_axis)**2)
        return distance < self.velocity

    def is_resting_at(self, timestamp: pd.Timestamp) -> bool:
        """주어진 시각에 택시가 휴식해야 하는지 반환합니다."""
        while self._rest_time_index < len(self.rest_times):
            rest_time = self.rest_times.iloc[self._rest_time_index]
            if timestamp <= rest_time["end"]:
                return rest_time["start"] <= timestamp <= rest_time["end"]
            self._rest_time_index += 1
        return False

class Passenger:
    def __init__(self, data: dict):
        self.name = data['name']
        self.x_axis = data['x_axis']
        self.y_axis = data['y_axis']
        self.to_x_axis = data['to_x_axis']
        self.to_y_axis = data['to_y_axis']
        self.departure_time = None
        self.arrival_time = None
        self.x_velocity = 0
        self.y_velocity = 0
        self.velocity = 0.00004475
        self.waiting_time = 0
        self.fee = data['fee']

    def start_move(self, x_velocity: float, y_velocity: float):
        self.departure_time = temp_time
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity

    def move(self):
        self.x_axis += self.x_velocity
        self.y_axis += self.y_velocity

    def stop_move(self):
        self.arrival_time = temp_time
        self.waiting_time = self.arrival_time - self.departure_time
        return self

    def is_at_destination(self) -> bool:
        if self.to_x_axis is None or self.to_y_axis is None:
            return False
        distance = math.sqrt((self.to_x_axis - self.x_axis)**2 + (self.to_y_axis - self.y_axis)**2)
        return distance < self.velocity


class Matrix:
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix


class Observer:
    def __init__(self):
        self.moving_taxis: List[Taxi] = []
        self.waiting_taxis: List[Taxi] = []
        self.resting_taxis: List[Taxi] = []
        self.moving_passengers: List[Passenger] = []
        self.waiting_passengers: List[Passenger] = []

    def add_passenger(self, passenger: Passenger):
        self.waiting_passengers.append(passenger)

    def add_taxi(self, taxi: Taxi):
        self.waiting_taxis.append(taxi)

    def delete_passenger(self, passenger: Passenger):
        if passenger in self.moving_passengers:
            self.moving_passengers.remove(passenger)

    def delete_taxi(self, taxi: Taxi):
        if taxi in self.moving_taxis:
            self.moving_taxis.remove(taxi)

    def time_pass(self):
        global temp_time
        temp_time += pd.Timedelta(seconds=1)

    def assign_calls(self):
        for passenger in self.waiting_passengers[:]:
            available_taxis = [taxi for taxi in self.waiting_taxis + self.moving_taxis 
                            if taxi.status in ["waiting", "to_cluster"]]
            
            if available_taxis:
                nearest_taxi = min(available_taxis, key=lambda t: self.calculate_distance(t, passenger))
                
                if nearest_taxi in self.waiting_taxis:
                    self.waiting_taxis.remove(nearest_taxi)
                    self.moving_taxis.append(nearest_taxi)
                elif nearest_taxi in self.moving_taxis:
                    # 이미 moving_taxis에 있으므로 이동만 필요
                    pass

                nearest_taxi.start_move(passenger.x_axis, passenger.y_axis, "to_passenger")
                nearest_taxi.passenger = passenger
                self.waiting_passengers.remove(passenger)
                
                logging.info(f"{temp_time}: Assigned taxi {nearest_taxi.name} to passenger {nearest_taxi.passenger.name} (status: {nearest_taxi.status}) ")
    
    @staticmethod
    def calculate_distance(taxi: Taxi, passenger: Passenger) -> float:
        return math.sqrt((taxi.x_axis - passenger.x_axis)**2 + (taxi.y_axis - passenger.y_axis)**2)

    def update(self):
        global passenger_list, temp_time, global_last_updated_time
        if temp_time.second == 0:
            with open(f'{save_path}/{name}.csv', 'a') as f:
                for taxi in self.moving_taxis + self.waiting_taxis + self.resting_taxis:
                    f.write(f"{taxi.name},{temp_time},{taxi.x_axis},{taxi.y_axis},{taxi.status}\n")

        if global_last_updated_time == None:
            temp_time_str = temp_time.strftime('%Y%m%d%H%M')
            temp_time_str = temp_time_str[:-2] + '00'
            for cluster in clusters:
                cluster.update_prediction(temp_time_str)
                cluster.update_nearby_taxis(self.moving_taxis + self.waiting_taxis, 0.1)
                cluster.update_competition()
                cluster.last_updated_time = temp_time
            global_last_updated_time = temp_time

        for taxi in self.resting_taxis[:]:
            if not taxi.is_resting_at(temp_time):
                taxi.status = "waiting"
                self.resting_taxis.remove(taxi)
                self.waiting_taxis.append(taxi)
                logging.info(f"{temp_time}: Taxi {taxi.name} finished resting and is now waiting")

        for taxi in self.moving_taxis[:]:
            taxi.move()
            if taxi.status != "to_destination":
                taxi.passengerless_time+=1
                
            if taxi.is_at_destination():
                
                if taxi.status == "to_passenger":
                    taxi.start_move(taxi.passenger.to_x_axis, taxi.passenger.to_y_axis,"to_destination")
                    taxi.passenger.start_move(taxi.x_velocity, taxi.y_velocity)
                    self.moving_passengers.append(taxi.passenger)
                    logging.info(f"{temp_time}: Taxi {taxi.name} picked up passenger {taxi.passenger.name}")

                elif taxi.status == "to_destination":
                    taxi.x_velocity = taxi.y_velocity = 0
                    taxi.earnings += taxi.passenger.fee
                    taxi.passenger = None
                    logging.info(f"{temp_time}: Taxi {taxi.name} completed a trip and is now waiting, passengerless time: {taxi.passengerless_time}")

                    self.moving_taxis.remove(taxi)
                    if taxi.is_resting_at(temp_time):
                        taxi.status = "resting"
                        self.resting_taxis.append(taxi)
                    else:
                        taxi.status = "waiting"
                        self.waiting_taxis.append(taxi)
 
                elif taxi.status == "to_cluster":
                    taxi.x_velocity = taxi.y_velocity = 0
                    logging.info(f"{temp_time}: Taxi {taxi.name} arrived at cluster ({taxi.to_x_axis}, {taxi.to_y_axis}) and is ready for new passengers")

                    if taxi.is_resting_at(temp_time):
                        taxi.status = "resting"
                        self.moving_taxis.remove(taxi)
                        self.resting_taxis.append(taxi)
                    else:
                        taxi.status = "waiting"
                        # 클러스터에 도착해도 moving_taxis에 유지, 바로 다른 승객을 받을 수 있도록

        for taxi in self.waiting_taxis[:]:
            taxi.passengerless_time += 1
            cluster_x, cluster_y = taxi.choose_cluster(alg_name)
            taxi.start_move(cluster_x, cluster_y, "to_cluster")
            self.waiting_taxis.remove(taxi)
            self.moving_taxis.append(taxi)
            logging.info(f"{temp_time}: Taxi {taxi.name} is heading to cluster")

        for passnger in self.waiting_passengers[:]:
            passnger.waiting_time += 1
        for passenger in self.moving_passengers[:]:
            passenger.move()
            if passenger.is_at_destination():
                passenger.stop_move()
                self.moving_passengers.remove(passenger)

        while passenger_list and temp_time == passenger_list[0]['datetime']:
            self.add_passenger(Passenger(passenger_list.popleft()))

        self.assign_calls()
        self.time_pass()
        # print passenger number of who are waiting
        if temp_time.second == 0 and temp_time.minute % 5 == 0:
            logging.info(f"{temp_time}: Number of waiting passengers: {len(self.waiting_passengers)}")


# 시뮬레이션 실행 함수
def run_simulation(observer: Observer, steps: int):
    global temp_time
    for _ in tqdm(range(steps), desc="Simulation Progress"):
        observer.update()

    all_passengerless_time = np.array(
        [taxi.passengerless_time for taxi in observer.moving_taxis]
        + [taxi.passengerless_time for taxi in observer.waiting_taxis]
    )
    mean_passengerless_time = round(all_passengerless_time.mean(), 3)
    std_passengerless_time = round(all_passengerless_time.std(), 3)

    all_waiting_time = np.array(
        [passenger.waiting_time for passenger in observer.moving_passengers]
        + [passenger.waiting_time for passenger in observer.waiting_passengers]
    )
    mean_waiting_time = round(all_waiting_time.mean(), 3)
    std_waiting_time = round(all_waiting_time.std(), 3)

    passengerless_rate = sum([taxi.passengerless_time / steps for taxi in observer.moving_taxis]) / taxis
    passengerless_rate += sum([taxi.passengerless_time / steps for taxi in observer.waiting_taxis]) / taxis

    all_todest_time = np.array(
        [taxi.to_destination_time for taxi in observer.moving_taxis]
        + [taxi.to_destination_time for taxi in observer.waiting_taxis]
    )
    mean_todest_time = round(all_todest_time.mean(), 3)
    std_todest_time = round(all_todest_time.std(), 3)

    all_earnings = np.array(
        [taxi.earnings for taxi in observer.moving_taxis]
        + [taxi.earnings for taxi in observer.waiting_taxis]
    )
    mean_earnings = round(all_earnings.mean(), 3)
    std_earnings = round(all_earnings.std(), 3)

    logging.info(f"Mean passengerless time: {mean_passengerless_time} (± {std_passengerless_time})")
    logging.info(f"Mean waiting time: {mean_waiting_time} (± {std_waiting_time})")
    logging.info(f"Mean time heading to passenger: {mean_todest_time} (± {std_todest_time})")
    logging.info(f"Mean earnings: {mean_earnings} (± {std_earnings})")
    logging.info(f"Passengerless rate: {round(passengerless_rate*100, 3)}%")

    print(f"Mean passengerless time: {mean_passengerless_time} (±{std_passengerless_time})")
    print(f"Mean waiting time: {mean_waiting_time} (±{std_waiting_time})")
    print(f"Mean time heading to passenger: {mean_todest_time} (±{std_todest_time})")
    print(f"Mean earnings: {mean_earnings} (±{std_earnings})")
    print(f"Passengerless rate: {round(passengerless_rate*100, 3)}%")

    # message is the same as the print message
    
    message =   f"Seed = {seed}\n" + \
                f"Mean passengerless time: {mean_passengerless_time} (±{std_passengerless_time})\n" + \
                f"Mean waiting time: {mean_waiting_time} (±{std_waiting_time})\n" + \
                f"Mean time heading to passenger: {mean_todest_time} (±{std_todest_time})\n" + \
                f"Mean earnings: {mean_earnings} (±{std_earnings})\n" + \
                f"Passengerless rate: {round(passengerless_rate*100)}%"
    send_chat(webhookURL, message)

if __name__ == "__main__":
    test = pd.read_csv(test_file)
    test['datetime'] = pd.to_datetime(test['datetime'])

    passenger_list = test.to_dict('records')
    for i in range(len(passenger_list)):
        passenger_list[i]['name'] = f'P{i}'
    passenger_list = deque(passenger_list)
    temp_time = test['datetime'].iloc[0]
    global_last_updated_time = None
    clusters = [Cluster(center[0], center[1], str(i)) for i, center in enumerate(kmeans.cluster_centers_)]
    observer = Observer()

    all_rest_times = pd.read_csv(rest_file)
    all_rest_times["start"] = pd.to_datetime(all_rest_times["start"])
    all_rest_times["end"] = pd.to_datetime(all_rest_times["end"])
    all_rest_times["duration"] = pd.to_timedelta(all_rest_times["duration"])

    # 랜덤하게 택시 생성 - x_axis는 127.3~127.4, y_axis는 36.3~36.4 사이의 값으로 생성
    taxi_names = all_rest_times['name'].unique()
    for i, taxi_name in enumerate(taxi_names):
        observer.add_taxi(
            Taxi(
                name=f"T{i}",
                x_axis=(127.3 + 0.1 * random.random()),
                y_axis=(36.3 + 0.1 * random.random()),
                rest_times=all_rest_times[all_rest_times["name"] == taxi_name],
            ),
        )

    # Run simulation with steps from command line argument
    run_simulation(observer, steps)
    if visualize:
        subprocess.run(["streamlit", "run", "visualization.py"])
        time.sleep(3)
        webbrowser.open('http://localhost:8501')
