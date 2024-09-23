import configparser
from collections import abc
import pandas as pd
from main import predict
import pickle
import math
from typing import List, Tuple, Optional
import random
from collections import deque
import numpy as np
import logging
from tqdm import tqdm, trange
import subprocess
import webbrowser
import json
import requests
import time

config = configparser.ConfigParser()
config.read('config.ini')

taxis = int(config['SIMULATION']['taxis'])
steps = int(config['SIMULATION']['steps'])
test_file = config['SIMULATION']['test_file']
visualize = bool(config['SIMULATION']['visualize'])
alg_name = config['SIMULATION']['alg_name']
save_path = config['SIMULATION']['save_path']
name = f'{alg_name}_{taxis}_{steps}'

logging.basicConfig(filename=f"{save_path}/{name}.log", level=logging.INFO)

with open(f'{save_path}/{name}.csv', 'w') as f:
    f.write("id,time,lon,lat,status\n")

data = pd.read_csv('../test_taxi.csv')
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
    def __init__(self, data: dict):
        self.name = data['name']
        self.x_axis = data['x_axis']
        self.y_axis = data['y_axis']
        self.destination = None
        self.velocity = 0.00004475
        self.x_velocity = 0
        self.y_velocity = 0
        self.passengerless_time = 0
        self.to_passenger_time = 0
        self.status = "waiting"  # "waiting", "to_passenger", "to_destination", "to_cluster"
        self.passenger = None

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

        match alg_name:
            case "Cluster_Probability":
                best_cluster = max(clusters, key=lambda c: c.predicted_demand)
            case "Cluster_Probability+Distance":                
                scores = []
                for cluster in clusters:
                    distance_score = self.calculate_distance(cluster)
                    demand_score = cluster.predicted_demand
                    competition_score = cluster.competition
                    total_score = demand_score / (distance_score * competition_score) if competition_score > 0 else demand_score / distance_score
                    scores.append((cluster, total_score))
                
                best_cluster = max(scores, key=lambda x: x[1])[0]
            case "Inverse_Competition":
                best_cluster = min(clusters, key=lambda c: c.competition if c.competition > 0 else float('inf'))
            case "Demand_Competition_Ratio":
                best_cluster = max(clusters, key=lambda c: c.predicted_demand / c.competition if c.competition > 0 else c.predicted_demand)
            case "Weighted_Score":
                w_demand, w_distance, w_competition = 0.5, 0.3, 0.2  # Adjust weights as needed
                scores = []
                for cluster in clusters:
                    demand_score = cluster.predicted_demand / max(c.predicted_demand for c in clusters)
                    distance_score = 1 - (self.calculate_distance(cluster) / max(self.calculate_distance(c) for c in clusters))
                    competition_score = 1 - (cluster.competition / max(c.competition for c in clusters) if cluster.competition > 0 else 0)
                    total_score = w_demand * demand_score + w_distance * distance_score + w_competition * competition_score
                    scores.append((cluster, total_score))
                best_cluster = max(scores, key=lambda x: x[1])[0]
            case "Time_Aware":
                current_hour = temp_time.hour
                if 6 <= current_hour < 10 or 16 <= current_hour < 20:  # Rush hours
                    best_cluster = max(clusters, key=lambda c: c.predicted_demand / (c.competition if c.competition>0.0 else float('inf') ** 0.5))
                else:  # Non-rush hours
                    best_cluster = max(clusters, key=lambda c: c.predicted_demand / (self.calculate_distance(c) * c.competition if c.competition>0.0 else float('inf')))
            case "Adaptive":
                if self.passengerless_time > 300:  # If waiting for more than 5 minutes
                    best_cluster = min(clusters, key=lambda c: c.competition)
                else:
                    best_cluster = max(clusters, key=lambda c: c.predicted_demand / (self.calculate_distance(c) * c.competition if c.competition>0.0 else float('inf')))

        return best_cluster.x_axis, best_cluster.y_axis

    def calculate_distance(self, cluster: Cluster) -> float:
        return math.sqrt((cluster.x_axis - self.x_axis)**2 + (cluster.y_axis - self.y_axis)**2)

    def start_move(self, destination: Tuple[float, float], status: str):
        self.status = status
        self.destination = destination
        distance = math.sqrt((destination[0] - self.x_axis)**2 + (destination[1] - self.y_axis)**2)
        self.x_velocity = self.velocity * (destination[0] - self.x_axis) / distance
        self.y_velocity = self.velocity * (destination[1] - self.y_axis) / distance

    def move(self):
        self.x_axis += self.x_velocity
        self.y_axis += self.y_velocity
        if self.status == "to_passenger":
            self.to_passenger_time += 1

    def is_at_destination(self) -> bool:
        if self.destination is None:
            return False
        distance = math.sqrt((self.destination[0] - self.x_axis)**2 + (self.destination[1] - self.y_axis)**2)
        return distance < self.velocity

class Passenger:
    def __init__(self, data: dict):
        self.name = data['name']
        self.x_axis = data['x_axis']
        self.y_axis = data['y_axis']
        self.destination = data['destination']
        self.departure_time = None
        self.arrival_time = None
        self.x_velocity = 0
        self.y_velocity = 0
        self.velocity = 0.00004475
        self.waiting_time = 0

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
        if self.destination is None:
            return False
        distance = math.sqrt((self.destination[0] - self.x_axis)**2 + (self.destination[1] - self.y_axis)**2)
        return distance < self.velocity

class Observer:
    def __init__(self):
        self.moving_taxis: List[Taxi] = []
        self.waiting_taxis: List[Taxi] = []
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

                nearest_taxi.start_move((passenger.x_axis, passenger.y_axis), "to_passenger")
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
                for taxi in self.moving_taxis:
                    f.write(f"{taxi.name},{temp_time},{taxi.x_axis},{taxi.y_axis},{taxi.status}\n")
                for taxi in self.waiting_taxis:
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
            

        for taxi in self.moving_taxis[:]:
            taxi.move()
            if taxi.status != "to_destination":
                taxi.passengerless_time+=1
                
            if taxi.is_at_destination():
                
                if taxi.status == "to_passenger":
                    taxi.start_move(taxi.passenger.destination, "to_destination")
                    taxi.passenger.start_move(taxi.x_velocity, taxi.y_velocity)
                    self.moving_passengers.append(taxi.passenger)
                    logging.info(f"{temp_time}: Taxi {taxi.name} picked up passenger {taxi.passenger.name}")

                elif taxi.status == "to_destination":
                    taxi.status = "waiting"
                    taxi.passenger = None
                    self.moving_taxis.remove(taxi)
                    self.waiting_taxis.append(taxi)
                    logging.info(f"{temp_time}: Taxi {taxi.name} completed a trip and is now waiting, passengerless time: {taxi.passengerless_time}")

                elif taxi.status == "to_cluster":
                    taxi.status = "waiting"
                    taxi.x_velocity = 0
                    taxi.y_velocity = 0
                    # 클러스터에 도착해도 moving_taxis에 유지, 바로 다른 승객을 받을 수 있도록
                    logging.info(f"{temp_time}: Taxi {taxi.name} arrived at cluster ({taxi.destination[0]}, {taxi.destination[1]}) and is ready for new passengers")

        for taxi in self.waiting_taxis[:]:
            taxi.passengerless_time += 1
            taxi.destination = taxi.choose_cluster(alg_name)
            taxi.start_move(taxi.destination, "to_cluster")
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

    sum_passengerless_time = sum([taxi.passengerless_time for taxi in observer.moving_taxis])
    sum_passengerless_time += sum([taxi.passengerless_time for taxi in observer.waiting_taxis])

    sum_waiting_time = sum([passenger.waiting_time for passenger in observer.moving_passengers])
    sum_waiting_time += sum([passenger.waiting_time for passenger in observer.waiting_passengers])

    passengerless_rate = sum([taxi.passengerless_time / steps for taxi in observer.moving_taxis]) / taxis
    passengerless_rate += sum([taxi.passengerless_time / steps for taxi in observer.waiting_taxis]) / taxis

    sum_to_passenger_time = sum([taxi.to_passenger_time for taxi in observer.moving_taxis])
    sum_to_passenger_time += sum([taxi.to_passenger_time for taxi in observer.waiting_taxis])  

    logging.info(f"Sum of passengerless time: {sum_passengerless_time}")
    logging.info(f"Sum of waiting time: {sum_waiting_time}")
    logging.info(f"Sum of time heading to passenger: {sum_to_passenger_time}")
    logging.info(f"Passengerless rate: {passengerless_rate*100}%")

    print(f"Sum of passengerless time: {sum_passengerless_time}")
    print(f"Sum of waiting time: {sum_waiting_time}")
    print(f"Sum of time heading to passenger: {sum_to_passenger_time}")
    print(f"Passengerless rate: {passengerless_rate*100}%")

    message = f'Algorithm name : {alg_name}\nTaxi numbers: {taxis}\nStep numbers: {steps}\nSum of passengerless time: {sum_passengerless_time}\nSum of waiting time: {sum_waiting_time}\nSum of time heading to passenger: {sum_to_passenger_time}\nPassengerless rate: {passengerless_rate*100}%'

    send_chat(webhookURL, message)

if __name__ == "__main__":
    test = pd.read_csv(test_file)
    test['datetime']= pd.to_datetime(test['datetime'])
    test['destination'] = test['destination'].apply(lambda x: eval(x))

    passenger_list = test.to_dict('records')
    for i in range(len(passenger_list)):
        passenger_list[i]['name'] = f'P{i}'
    passenger_list = deque(passenger_list)
    temp_time = test['datetime'].iloc[0]
    global_last_updated_time = None
    clusters = [Cluster(center[0], center[1], str(i)) for i, center in enumerate(kmeans.cluster_centers_)]
    observer = Observer()
    
    # 랜덤하게 택시 생성 - x_axis는 127.3~127.4, y_axis는 36.3~36.4 사이의 값으로 생성
    for i in range(taxis):
        observer.add_taxi(Taxi({'name': f'T{i}', 'x_axis': 127.3 + 0.1 * random.random(), 'y_axis': 36.3 + 0.1 * random.random()}))

    # Run simulation with steps from command line argument
    run_simulation(observer, steps)
    if visualize:
        subprocess.run(["streamlit", "run", "visualization.py"])
        time.sleep(3)
        webbrowser.open('http://localhost:8501')