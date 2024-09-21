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

config = configparser.ConfigParser()
config.read('config.ini')

output = config['SIMULATION']['output']
taxis = int(config['SIMULATION']['taxis'])
steps = int(config['SIMULATION']['steps'])
test_file = config['SIMULATION']['test_file']
visualize = bool(config['SIMULATION']['visualize'])

logging.basicConfig(filename="record.log", level=logging.INFO)

with open(output, 'w') as f:
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

clusters = kmeans.cluster_centers_

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
        self.status = "waiting"  # "waiting", "to_passenger", "to_destination", "to_cluster"
        self.passenger = None

    def choose_cluster(self) -> Tuple[float, float]:
        temp_time_str = temp_time.strftime('%Y%m%d%H%M')
        temp_time_str = temp_time_str[:-2] + '00'
        predictions = predict(temp_time_str, True)
        predictions.sort(key=lambda x: x[1], reverse=True)
        return clusters[predictions[0][0]][0], clusters[predictions[0][0]][1]

    def start_move(self, destination: Tuple[float, float], status: str):
        self.status = status
        self.destination = destination
        distance = math.sqrt((destination[0] - self.x_axis)**2 + (destination[1] - self.y_axis)**2)
        self.x_velocity = self.velocity * (destination[0] - self.x_axis) / distance
        self.y_velocity = self.velocity * (destination[1] - self.y_axis) / distance

    def move(self):
        self.x_axis += self.x_velocity
        self.y_axis += self.y_velocity

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
        global passenger_list, temp_time
        if temp_time.second == 0:
            with open(output, 'a') as f:
                for taxi in self.moving_taxis:
                    f.write(f"{taxi.name},{temp_time},{taxi.x_axis},{taxi.y_axis},{taxi.status}\n")
                for taxi in self.waiting_taxis:
                    f.write(f"{taxi.name},{temp_time},{taxi.x_axis},{taxi.y_axis},{taxi.status}\n")

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
                    # 클러스터에 도착해도 moving_taxis에 유지, 바로 다른 승객을 받을 수 있도록
                    logging.info(f"{temp_time}: Taxi {taxi.name} arrived at cluster ({taxi.destination[0]}, {taxi.destination[1]}) and is ready for new passengers")

        for taxi in self.waiting_taxis[:]:
            taxi.passengerless_time += 1
            taxi.destination = taxi.choose_cluster()
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

    logging.info(f"Sum of passengerless time: {sum_passengerless_time}")
    logging.info(f"Sum of waiting time: {sum_waiting_time}")

    print(f"Sum of passengerless time: {sum_passengerless_time}")
    print(f"Sum of waiting time: {sum_waiting_time}")


test = pd.read_csv(test_file)
test['datetime']= pd.to_datetime(test['datetime'])
test['destination'] = test['destination'].apply(lambda x: eval(x))


passenger_list = test.to_dict('records')
for i in range(len(passenger_list)):
    passenger_list[i]['name'] = f'P{i}'
passenger_list = deque(passenger_list)

temp_time = test['datetime'].iloc[0]

if __name__ == "__main__":
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
        