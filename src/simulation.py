import configparser
import json
import logging
import math
import pickle
import random
import subprocess
import time
import webbrowser
from argparse import ArgumentParser
from collections import deque
from typing import List, Tuple

import numpy as np
import optuna
import pandas as pd
import requests
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from prediction import predict


class Cluster:
    global temp_time, global_last_updated_time

    def __init__(self, x_axis: float, y_axis: float, name: str):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.name = name
        self.predicted_demand = 0.0
        self.predicted_reason = ""
        self.nearby_taxis: List["Taxi"] = []
        self.competition = 0.0
        self.features = cluster_features.get(str(name), {})  # Use str(name) as the key
        self.last_updated_time = None

    def update_prediction(self, time_str: str):
        global temp_time, global_last_updated_time
        if global_last_updated_time and global_last_updated_time.hour == temp_time.hour:
            return
        predictions = predict(time_str, True)
        for pred in predictions:
            if pred[0] == int(self.name):
                self.predicted_demand = pred[1]
        self.predicted_reason = pred[2]
        self.last_updated_time = temp_time

    def update_nearby_taxis(self, all_taxis: List["Taxi"], max_distance: float):
        self.nearby_taxis = [
            taxi
            for taxi in all_taxis
            if self.calculate_distance(taxi) <= max_distance
            and taxi.status not in ["to_passenger", "to_destination"]
        ]

    def calculate_distance(self, taxi: "Taxi") -> float:
        return math.sqrt(
            (self.x_axis - taxi.x_axis) ** 2 + (self.y_axis - taxi.y_axis) ** 2
        )

    def update_competition(self):
        if self.predicted_demand > 0 and len(self.nearby_taxis) > 0:
            self.competition = len(self.nearby_taxis) / self.predicted_demand
        else:
            self.competition = 0.0


class Taxi:
    def __init__(
        self, name: str, x_axis: float, y_axis: float, rest_times: pd.DataFrame
    ):
        self.name = name
        self.x_axis, self.y_axis = x_axis, y_axis
        self.to_x_axis = None
        self.to_y_axis = None
        self.velocity = 0.00004475
        self.x_velocity = 0
        self.y_velocity = 0
        self.driving_time = 0
        self.passengerless_time = 0
        self.to_passenger_time = 0
        self.to_destination_time = 0
        self.status = "waiting"  # "waiting", "to_passenger", "to_destination", "to_cluster", "resting"
        self.passenger = None
        self.earnings = 0
        rest_times = rest_times.sort_values("start")
        # 휴식 시간을 (start, end) 튜플의 리스트로 변환
        self.rest_intervals = list(zip(rest_times["start"], rest_times["end"]))
        self._rest_time_index = 0

    def safe_division(n, d, default=0):
        return n / d if d != 0 else default

    def choose_cluster_matrix(self, assignments):
        # This method now simply returns the assigned cluster from the optimal assignment
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

    def calculate_distance(self, cluster: Cluster) -> float:
        return math.sqrt(
            (cluster.x_axis - self.x_axis) ** 2 + (cluster.y_axis - self.y_axis) ** 2
        )

    def start_move(self, to_x_axis, to_y_axis, status: str):
        self.status = status
        self.to_x_axis = to_x_axis
        self.to_y_axis = to_y_axis
        distance = math.sqrt(   
            (self.to_x_axis - self.x_axis) ** 2 + (self.to_y_axis - self.y_axis) ** 2
        )
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
        distance = math.sqrt(
            (self.to_x_axis - self.x_axis) ** 2 + (self.to_y_axis - self.y_axis) ** 2
        )
        return distance < self.velocity

    def is_resting_at(self, timestamp: pd.Timestamp) -> bool:
        """주어진 시각에 택시가 휴식해야 하는지 반환합니다."""
        while self._rest_time_index < len(self.rest_intervals):
            start, end = self.rest_intervals[self._rest_time_index]
            if timestamp < start:
                return False
            if start <= timestamp <= end:
                return True
            self._rest_time_index += 1
        return False


class Passenger:
    def __init__(self, data: dict):
        self.name = data["name"]
        self.x_axis = data["x_axis"]
        self.y_axis = data["y_axis"]
        self.to_x_axis = data["to_x_axis"]
        self.to_y_axis = data["to_y_axis"]
        self.departure_time = None
        self.arrival_time = None
        self.x_velocity = 0
        self.y_velocity = 0
        self.velocity = 0.00004475
        self.waiting_time = 0
        self.fee = data["fee"]

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
        distance = math.sqrt(
            (self.to_x_axis - self.x_axis) ** 2 + (self.to_y_axis - self.y_axis) ** 2
        )
        return distance < self.velocity


class Matrix:
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix


class Observer:
    def __init__(self):
        self.moving_taxis: List[Taxi] = []
        self.waiting_taxis: List[Taxi] = []
        self.resting_taxis: List[Taxi] = []
        self.available_taxis: List[Taxi] = []
        self.moving_passengers: List[Passenger] = []
        self.waiting_passengers: List[Passenger] = []
        self.distance_matrix = None
        self.competition_matrix = None
        self.demand_matrix = None

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
        self, taxis, clusters, distance_matrix, competition_matrix, demand_matrix
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

        return assignments

    def naive_cluster_assignment(
        self, taxis, clusters, distance_matrix, competition_matrix, demand_matrix
    ):
                # global distance_rate, competition_rate, demand_rate
        n_taxis = len(taxis)
        n_clusters = len(clusters)

        # Normalize matrices
        norm_demand = demand_matrix.matrix / np.max(demand_matrix.matrix)

        # return the cluster with the highest demand
        assignments = {}
        for i in range(n_taxis):
            best_cluster_index = np.argmax(norm_demand[i, :])
            assignments[taxis[i].name] = clusters[best_cluster_index]
        
        return assignments

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
            available_taxis = [
                taxi
                for taxi in self.waiting_taxis + self.moving_taxis
                if taxi.status in ["waiting", "to_cluster"]
            ]

            if available_taxis:
                nearest_taxi = min(
                    available_taxis, key=lambda t: self.calculate_distance(t, passenger)
                )

                if nearest_taxi in self.waiting_taxis:
                    self.waiting_taxis.remove(nearest_taxi)
                    self.moving_taxis.append(nearest_taxi)
                elif nearest_taxi in self.moving_taxis:
                    # 이미 moving_taxis에 있으므로 이동만 필요
                    pass

                nearest_taxi.start_move(
                    passenger.x_axis, passenger.y_axis, "to_passenger"
                )
                nearest_taxi.passenger = passenger
                self.waiting_passengers.remove(passenger)

                if write_log:
                    logging.info(
                        f"{temp_time}: Assigned taxi {nearest_taxi.name} to passenger {nearest_taxi.passenger.name} (status: {nearest_taxi.status}) "
                    )

    @staticmethod
    def calculate_distance(taxi: Taxi, passenger: Passenger) -> float:
        return math.sqrt(
            (taxi.x_axis - passenger.x_axis) ** 2
            + (taxi.y_axis - passenger.y_axis) ** 2
        )

    def update(self):
        global passenger_list, temp_time, global_last_updated_time, date
        if write_csv and temp_time.second == 0:
            with open(f"{result_path}/{date}_{name}.csv", "a") as f:
                for taxi in self.moving_taxis + self.waiting_taxis + self.resting_taxis:
                    f.write(
                        f"{taxi.name},{temp_time},{taxi.x_axis},{taxi.y_axis},{taxi.status}\n"
                    )

        if global_last_updated_time == None:
            temp_time_str = temp_time.strftime("%Y%m%d%H%M")
            temp_time_str = temp_time_str[:-2] + "00"
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
                if write_log:
                    logging.info(
                        f"{temp_time}: Taxi {taxi.name} finished resting and is now waiting"
                    )

        for taxi in self.moving_taxis[:]:
            taxi.driving_time += 1
            taxi.move()
            if taxi.status != "to_destination":
                taxi.passengerless_time += 1

            if taxi.is_at_destination():

                if taxi.status == "to_passenger":
                    taxi.start_move(
                        taxi.passenger.to_x_axis,
                        taxi.passenger.to_y_axis,
                        "to_destination",
                    )
                    taxi.passenger.start_move(taxi.x_velocity, taxi.y_velocity)
                    self.moving_passengers.append(taxi.passenger)
                    if write_log:
                        logging.info(
                            f"{temp_time}: Taxi {taxi.name} picked up passenger {taxi.passenger.name}"
                        )

                elif taxi.status == "to_destination":
                    taxi.x_velocity = taxi.y_velocity = 0
                    taxi.earnings += taxi.passenger.fee
                    taxi.passenger = None
                    if write_log:
                        logging.info(
                            f"{temp_time}: Taxi {taxi.name} completed a trip and is now waiting, passengerless time: {taxi.passengerless_time}"
                        )

                    self.moving_taxis.remove(taxi)
                    if taxi.is_resting_at(temp_time):
                        taxi.status = "resting"
                        self.resting_taxis.append(taxi)
                    else:
                        taxi.status = "waiting"
                        self.waiting_taxis.append(taxi)

                elif taxi.status == "to_cluster":
                    taxi.x_velocity = taxi.y_velocity = 0
                    if write_log:
                        logging.info(
                            f"{temp_time}: Taxi {taxi.name} arrived at cluster ({taxi.to_x_axis}, {taxi.to_y_axis}) and is ready for new passengers"
                        )

                    if taxi.is_resting_at(temp_time):
                        taxi.status = "resting"
                        self.moving_taxis.remove(taxi)
                        self.resting_taxis.append(taxi)
                    else:
                        taxi.status = "waiting"
                        # 클러스터에 도착해도 moving_taxis에 유지, 바로 다른 승객을 받을 수 있도록

        self.available_taxis = self.moving_taxis + self.waiting_taxis

        if (
            self.distance_matrix is None
            or len(self.available_taxis) != self.distance_matrix.matrix.shape[0]
        ):
            self.distance_matrix, self.competition_matrix, self.demand_matrix = (
                self.create_assignment_matrices(self.available_taxis, clusters)
            )

        if self.waiting_taxis:
            if args.naive == "True":
                assignments = self.naive_cluster_assignment(
                    self.available_taxis,
                    clusters,
                    self.distance_matrix,
                    self.competition_matrix,
                    self.demand_matrix,
                )
            else:
                assignments = self.optimal_cluster_assignment(
                    self.available_taxis,
                    clusters,
                    self.distance_matrix,
                    self.competition_matrix,
                    self.demand_matrix,
                )

        for taxi in self.waiting_taxis[:]:
            taxi.driving_time += 1
            taxi.passengerless_time += 1

            assigned_cluster = taxi.choose_cluster_matrix(assignments)
            cluster_x, cluster_y = assigned_cluster.x_axis, assigned_cluster.y_axis

            taxi.start_move(cluster_x, cluster_y, "to_cluster")
            self.waiting_taxis.remove(taxi)
            self.moving_taxis.append(taxi)
            if write_log:
                logging.info(f"{temp_time}: Taxi {taxi.name} is heading to cluster")

        for passnger in self.waiting_passengers[:]:
            passnger.waiting_time += 1
        for passenger in self.moving_passengers[:]:
            passenger.move()
            if passenger.is_at_destination():
                passenger.stop_move()
                self.moving_passengers.remove(passenger)

        while passenger_list and temp_time == passenger_list[0]["datetime"]:
            self.add_passenger(Passenger(passenger_list.popleft()))

        self.assign_calls()
        self.time_pass()
        # print passenger number of who are waiting
        if temp_time.second == 0 and temp_time.minute % 5 == 0:
            if write_log:
                logging.info(
                    f"{temp_time}: Number of waiting passengers: {len(self.waiting_passengers)}"
                )


# 시뮬레이션 실행 함수
def run_simulation():
    global temp_time, global_last_updated_time, clusters, passenger_list, observer

    temp_time = test["datetime"].iloc[0]
    global_last_updated_time = None

    passenger_list = test.to_dict("records")
    for i in range(len(passenger_list)):
        passenger_list[i]["name"] = f"P{i}"
    passenger_list = deque(passenger_list)

    clusters = [
        Cluster(kmeans.cluster_centers_[i][0], kmeans.cluster_centers_[i][1], i)
        for i in remaining_clusters
    ]

    observer = Observer()

    taxi_names = all_rest_times["name"].unique()

    for i, taxi_name in enumerate(taxi_names):
        observer.add_taxi(
            Taxi(
                name=f"T{i}",
                x_axis=(127.3 + 0.1 * random.random()),
                y_axis=(36.3 + 0.1 * random.random()),
                rest_times=all_rest_times[all_rest_times["name"] == taxi_name],
            ),
        )

    for _ in tqdm(range(steps), desc="Simulation Progress"):
        observer.update()

    # exclude taxis which has resting time of more than 75% of the total time
    final_taxis = [
        taxi
        for taxi in observer.moving_taxis
        if taxi.passengerless_time / taxi.driving_time < 0.75
    ]
    final_taxis += [
        taxi
        for taxi in observer.waiting_taxis
        if taxi.passengerless_time / taxi.driving_time < 0.75
    ]
    final_taxis += [
        taxi
        for taxi in observer.resting_taxis
        if taxi.passengerless_time / taxi.driving_time < 0.75
    ]

    all_passengerless_time = np.array([taxi.passengerless_time for taxi in final_taxis])
    all_waiting_time = np.array(
        [passenger.waiting_time for passenger in observer.moving_passengers]
        + [passenger.waiting_time for passenger in observer.waiting_passengers]
    )
    all_todest_time = np.array([taxi.to_destination_time for taxi in final_taxis])
    all_earnings = np.array([taxi.earnings for taxi in final_taxis])

    results = {
        "mean_passengerless_time": all_passengerless_time.mean(),
        "std_passengerless_time": all_passengerless_time.std(),
        "mean_waiting_time": all_waiting_time.mean(),
        "std_waiting_time": all_waiting_time.std(),
        "mean_todest_time": all_todest_time.mean(),
        "std_todest_time": all_todest_time.std(),
        "mean_earnings": all_earnings.mean(),
        "std_earnings": all_earnings.std(),
        "passengerless_rate": sum(
            [taxi.passengerless_time / taxi.driving_time for taxi in final_taxis]
        ) / len(final_taxis),
        "todest_time_rate": sum(
            [taxi.to_destination_time / taxi.driving_time for taxi in final_taxis]
        ) / len(final_taxis),
        "earning_per_time": sum(
            [taxi.earnings / taxi.driving_time for taxi in final_taxis]
        ) / len(final_taxis),
    }
    
    return results
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="config.ini",
        help="Path to the configuration file (default: config.ini)",
    )
    parser.add_argument(
        "-d",
        "--distance_rate",
        type=float,
        help="Distance rate for the simulation",
    ),
    parser.add_argument(
        "-cr",
        "--competition_rate",
        type=float,
        help="Competition rate for the simulation",
    ),
    parser.add_argument(
        "-dr",
        "--demand_rate",
        type=float,
        help="Demand rate for the simulation",
    ),
    parser.add_argument(
        "-w",
        "--weekend",
        type=int,
        help="Weekend flag",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["simulation", "optimization"],
        help="Execution mode",
    )
    parser.add_argument(
        "-n",
        "--naive",
        help="Naive mode",
    )

    args = parser.parse_args()

    distance_rate = args.distance_rate
    competition_rate = args.competition_rate
    demand_rate = args.demand_rate
    weekend = args.weekend
    date = args.config.split("_")[1].split(".")[0]

    config = configparser.ConfigParser()
    config.read(args.config)
    taxis = int(config["SIMULATION"]["taxis"])
    steps = int(config["SIMULATION"]["steps"])
    save_path = config["TRAIN"]["save_path"]
    clustering_model_path = config["TRAIN"]["clustering_model_path"]
    cluster_features_path = config["TRAIN"]["cluster_features_path"]
    remaining_clusters_path = config["TRAIN"]["remaining_clusters_path"]
    xgb_model_path = config["TRAIN"]["xgb_model_path"]
    explainer_path = config["TRAIN"]["explainer_path"]
    test_file = config["SIMULATION"]["test_file"]
    rest_file = config["SIMULATION"]["rest_file"]
    write_csv = True if config["SIMULATION"]["write_csv"] == "True" else False
    write_log = True if config["SIMULATION"]["write_log"] == "True" else False
    visualize = True if config["SIMULATION"]["visualize"] == "True" else False
    result_path = config["SIMULATION"]["save_path"]
    name = f"{taxis}_{steps}"
    seed = int(config["GENERAL"]["seed"])

    random.seed(seed)

    if write_log:
        logging.basicConfig(filename=f"{result_path}/{date}_{name}.log", level=logging.INFO)

    if write_csv:
        with open(f"{result_path}/{date}_{name}.csv", "w") as f:
            f.write("id,time,lon,lat,status\n")

    with open(f"{save_path}/{clustering_model_path}", "rb") as f:
        kmeans = pickle.load(f)

    with open(f"{save_path}/{cluster_features_path}", "r") as f:    
        cluster_features = json.load(f)

    with open(f"{save_path}/{remaining_clusters_path}", "rb") as f:
        remaining_clusters = pickle.load(f)

    with open(f"{save_path}/{xgb_model_path}", "rb") as f:
        model = pickle.load(f)

    # load the explainer
    with open(f"{save_path}/{explainer_path}", "rb") as f:
        explainer = pickle.load(f)

    test = pd.read_csv(test_file)
    test["datetime"] = pd.to_datetime(test["datetime"])

    all_rest_times = pd.read_csv(rest_file)
    all_rest_times["start"] = pd.to_datetime(all_rest_times["start"])
    all_rest_times["end"] = pd.to_datetime(all_rest_times["end"])
    all_rest_times["duration"] = pd.to_timedelta(all_rest_times["duration"])

    if args.mode == "simulation":

        # Run simulation with steps from command line argument
        results = run_simulation()

        mean_passengerless_time = round(results['mean_passengerless_time'], 3)
        std_passengerless_time = round(results['std_passengerless_time'], 3)
        mean_waiting_time = round(results['mean_waiting_time'], 3)
        std_waiting_time = round(results['std_waiting_time'], 3)
        mean_todest_time = round(results['mean_todest_time'], 3)
        std_todest_time = round(results['std_todest_time'], 3)
        mean_earnings = round(results['mean_earnings'], 3)
        std_earnings = round(results['std_earnings'], 3)
        passengerless_rate = round(100 * results['passengerless_rate'], 3)
        todest_time_rate = round(100 * results['todest_time_rate'], 3)
        earning_per_time = round(results['earning_per_time'], 3)

        if write_log:
            logging.info(
                f"Mean passengerless time: {[mean_passengerless_time]} (± {std_passengerless_time})"
            )
            logging.info(f"Mean waiting time: {mean_waiting_time} (± {std_waiting_time})")
            logging.info(
                f"Mean time heading to passenger: {mean_todest_time} (± {std_todest_time})"
            )
            logging.info(f"Mean earnings: {mean_earnings} (± {std_earnings})")
            logging.info(f"Passengerless rate: {passengerless_rate}%")
            logging.info(f"Time heading to destination rate: {todest_time_rate}%")
            logging.info(f"Earnings per time: {earning_per_time} ₩")

        print(
            f"Mean passengerless time: {mean_passengerless_time} (±{std_passengerless_time})"
        )
        print(f"Mean waiting time: {mean_waiting_time} (±{std_waiting_time})")
        print(f"Mean time heading to passenger: {mean_todest_time} (±{std_todest_time})")
        print(f"Mean earnings: {mean_earnings} (±{std_earnings})")
        print(f"Passengerless rate: {passengerless_rate}%")
        print(f"Time heading to destination rate: {todest_time_rate}%")
        print(f"Earnings per time: {earning_per_time} ₩")

        # message is the same as the print message
        message =   f"Date: {date}\n" + \
                    f"Distance rate: {distance_rate}\n" + \
                    f"Competition rate: {competition_rate}\n" + \
                    f"Demand rate: {demand_rate}\n" + \
                    f"Mean passengerless time: {mean_passengerless_time} (±{std_passengerless_time})\n" + \
                    f"Mean waiting time: {mean_waiting_time} (±{std_waiting_time})\n" + \
                    f"Mean time heading to passenger: {mean_todest_time} (±{std_todest_time})\n" + \
                    f"Mean earnings: {mean_earnings} (±{std_earnings})\n" + \
                    f"Passengerless rate: {passengerless_rate}%\n" + \
                    f"Time heading to destination rate: {todest_time_rate}%\n" + \
                    f"Earnings per time: {earning_per_time} ₩"

        with open(f"{result_path}/result.csv", "a") as f:
            f.write(f"{distance_rate},{competition_rate},{demand_rate},{mean_waiting_time},{mean_todest_time},{mean_earnings},{passengerless_rate},{todest_time_rate},{earning_per_time},{weekend},{date}\n")

        if visualize:
            subprocess.run(["streamlit", "run", "visualization.py"])
            time.sleep(3)
            webbrowser.open("http://localhost:8501")

    elif args.mode == "optimization":

        def objective(trial):
            global distance_rate, competition_rate, demand_rate

            distance_rate = trial.suggest_float("distance_rate", 0.1, 0.9)
            competition_rate = trial.suggest_float("competition_rate", 0.1, 0.9)
            demand_rate = trial.suggest_float("demand_rate", 0.1, 0.9)

            # Normalize rates to sum to 1
            total = distance_rate + competition_rate + demand_rate
            distance_rate /= total
            competition_rate /= total
            demand_rate /= total

            results = run_simulation()

            return results["mean_earnings"]  # Negative because we want to maximize

        study = optuna.create_study(direction="maximize")
        if args.naive == "True":
            print("Naive mode")
            study.optimize(objective, n_trials=1)
        else:
            study.optimize(objective, n_trials=10)

        print("Best trial:")
        trial = study.best_trial

        print("Value: ", -trial.value)
        if args.naive != "True":
            print("Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

            # Normalize the best parameters
            total = sum(trial.params.values())
            normalized_params = {k: v / total for k, v in trial.params.items()}
            print("Normalized Params:")
            for key, value in normalized_params.items():
                print(f"    {key}: {value}")
