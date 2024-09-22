import numpy as np
from scipy.optimize import linear_sum_assignment
from taxi import Taxi

def baecha(taxis, passengers):
    result = allocate_taxis(taxis, passengers)
    return result

def allocate_taxis(taxis, passenger_coords) -> list[tuple[Taxi, tuple[np.float64, np.float64], np.float64]]:
    if len(taxis) == 0 or len(passenger_coords) == 0:
        return []
    
    n_taxis = len(taxis)
    n_passengers = len(passenger_coords)
    
    # Extract coordinates from Taxi objects
    taxi_coords = np.array([(taxi.lat, taxi.lng) for taxi in taxis])
    
    # Calculate the distance matrix
    distance_matrix = np.sqrt(((taxi_coords[:, np.newaxis, :] - passenger_coords) ** 2).sum(axis=2))
    
    # Pad the matrix if necessary
    if n_taxis > n_passengers:
        # Add dummy passengers
        padding = np.full((n_taxis, n_taxis - n_passengers), -100) # np.inf or -100?
        distance_matrix = np.hstack((distance_matrix, padding))
    elif n_passengers > n_taxis:
        # Add dummy taxis
        padding = np.full((n_passengers - n_taxis, n_passengers), -100) # np.inf or -100?
        distance_matrix = np.vstack((distance_matrix, padding))
    
    # Solve the assignment problem
    taxi_indices, passenger_indices = linear_sum_assignment(distance_matrix)
    
    # Filter out dummy assignments and create result
    valid_assignments = []
    for taxi_idx, passenger_idx in zip(taxi_indices, passenger_indices):
        if taxi_idx < n_taxis and passenger_idx < n_passengers:
            taxi = taxis[taxi_idx]
            passenger = tuple(passenger_coords[passenger_idx])
            distance = distance_matrix[taxi_idx, passenger_idx]
            valid_assignments.append((taxi, passenger, distance))
    
    return valid_assignments

# assignments = allocate_taxis(taxis, passenger_coords)
# 
# print("Optimal Taxi-Passenger Assignments:")
# total_distance = 0
# for taxi, passenger, distance in assignments:
    # total_distance += distance
    # print(f"Taxi {taxi.id} at ({taxi.lat}, {taxi.lng}) assigned to passenger at {passenger} (Distance: {distance:.2f})")
# 
# print(f"\nTotal Distance: {total_distance:.2f}")
# print(f"Unassigned taxis: {len(taxis) - len(assignments)}")
# print(f"Unserved passengers: {len(passenger_coords) - len(assignments)}")