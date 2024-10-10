import configparser
import os
import pickle
import warnings
from typing import List, Tuple, Dict
import json

import holidays
import numpy as np
import pandas as pd
import requests
import shap
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import optuna

warnings.filterwarnings("ignore")

# Read configuration
config = configparser.ConfigParser()
config.read("configs/config_train.ini")

# Get configuration values
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

# Set random seed
np.random.seed(SEED)
random_state = SEED


def load_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess data from CSV file."""
    df = pd.read_csv(file_path, encoding="utf-8-sig", engine="python")
    df = df[(df["x_axis"] != 0) & (df["y_axis"] != 0)]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["minute"] = df["datetime"].dt.minute // HOUR_INTERVAL
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["year"] = df["datetime"].dt.year
    df.drop(["datetime"], axis=1, inplace=True)
    return df


def get_weather_data() -> pd.DataFrame:
    """Fetch or load weather data."""
    if WEATHER_FILE != "None":
        data = []
        for month in range(3, 12):
            url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php?tm1=2023{str(month).zfill(2)}010000&tm2=2023{str(month+1).zfill(2)}010000&stn=133&help=0&authKey={WEATHER_API}"
            response = requests.get(url)
            raw_data = [
                [i for i in response.text.split("\n")[j].split(" ") if i != ""]
                for j in range(4, len(response.text.split("\n")))
            ]
            data.extend(raw_data)

        for month in range(1, 5):
            url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php?tm1=2024{str(month).zfill(2)}010000&tm2=2024{str(month+1).zfill(2)}010000&stn=133&help=0&authKey={WEATHER_API}"
            response = requests.get(url)
            raw_data = [
                [i for i in response.text.split("\n")[j].split(" ") if i != ""]
                for j in range(4, len(response.text.split("\n")))
            ]
            data.extend(raw_data)

        total_data = []
        for row in data:
            try:
                new_data = [row[0], row[3], row[11], row[13], row[15]]
                total_data.append(new_data)
            except:
                continue

        weather_df = pd.DataFrame(
            total_data, columns=["datetime", "WS", "TEMP", "HUMI", "RN"]
        )
        weather_df.to_csv("data/weather.csv", index=False)
    else:
        weather_df = pd.read_csv(WEATHER_FILE, encoding="utf-8-sig", engine="python")

    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"], format="%Y%m%d%H%M")
    weather_df["hour"] = weather_df["datetime"].dt.hour
    weather_df["month"] = weather_df["datetime"].dt.month
    weather_df["day"] = weather_df["datetime"].dt.day
    weather_df["year"] = weather_df["datetime"].dt.year
    weather_df["holiday"] = weather_df["datetime"].apply(
        lambda x: 1 if x in holidays.KR() else 0
    )
    weather_df.drop(["datetime"], axis=1, inplace=True)

    return weather_df


def perform_clustering(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> Tuple[KMeans, pd.DataFrame, pd.DataFrame]:
    """Perform KMeans clustering on the data."""

    kmeans = KMeans(n_clusters=CLUSTER_NUM, random_state=random_state).fit(
            train_data[["x_axis", "y_axis"]]
        )

    train_data["cluster"] = kmeans.predict(train_data[["x_axis", "y_axis"]])
    test_data["cluster"] = kmeans.predict(test_data[["x_axis", "y_axis"]])

    # Save clustering model
    with open(f"{SAVE_PATH}/{CLUSTERTING_MODEL_PATH}", "wb") as f:
        pickle.dump(kmeans, f)

    return kmeans, train_data, test_data


def filter_clusters(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    """Filter out clusters with less than 900 data points."""
    cluster = train_data.groupby("cluster").size().reset_index()
    cluster.columns = ["cluster", "count"]
    excluded_cluster = cluster[cluster["count"] < 900]["cluster"].tolist()
    remaining_cluster = cluster[cluster["count"] >= 900]["cluster"].tolist()

    train_data = train_data[train_data["cluster"].isin(remaining_cluster)]
    test_data = test_data[test_data["cluster"].isin(remaining_cluster)]

    return train_data, test_data, remaining_cluster


def add_count_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add count feature to the dataframe."""
    df["count"] = 1
    count_df = (
        df.groupby(["cluster", "hour", "day", "month", "year"]).count().reset_index()
    )
    count_df = count_df[["cluster", "hour", "day", "month", "year", "count"]]

    df = pd.merge(
        df, count_df, on=["cluster", "hour", "day", "month", "year"], how="left"
    )
    df["count"] = df["count_y"]
    df.drop(["count_x", "count_y"], axis=1, inplace=True)

    return df


def load_additional_data() -> Dict[str, pd.DataFrame]:
    """Load additional data sources."""
    data_sources = {
        "ktv": "dj_ktv.csv",
        "karaoke": "dj_karaoke.csv",
        "hospital": "dj_hospital.csv",
        "small_hospital": "dj_small_hospital.csv",
        "hotel": "dj_hotel.csv",
    }

    additional_data = {}
    for key, filename in data_sources.items():
        df = pd.read_csv(
            os.path.join(DATA_PATH, filename), encoding="utf-8-sig", engine="python"
        )
        df.rename(columns={"경도": "x_axis", "위도": "y_axis"}, inplace=True)
        additional_data[key] = df

    return additional_data


def add_new_columns(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    kmeans: KMeans,
    additional_data: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add new columns based on additional data sources."""
    for df in additional_data.values():
        df["cluster"] = kmeans.predict(df[["x_axis", "y_axis"]])

    hospital = additional_data["hospital"]
    small_hospital = additional_data["small_hospital"]

    hospital["업태구분명"] = pd.Categorical(hospital["업태구분명"]).codes
    number_of_hospital_type = len(hospital["업태구분명"].unique())

    small_hospital["업태구분명"] = pd.Categorical(small_hospital["업태구분명"]).codes
    number_of_small_hospital_type = len(small_hospital["업태구분명"].unique())

    new_columns = (
        [
            "sum_drinks",
            "sum_hospitals",
            "sum_hotels",
            "sum_drinks_area",
            "sum_hospital_rooms",
        ]
        + [f"sum_hospital_type_{i}" for i in range(number_of_hospital_type)]
        + [f"sum_small_hospital_type_{i}" for i in range(number_of_small_hospital_type)]
    )

    for df in [train_data, test_data]:
        df[new_columns] = 0

    drink_dfs = [additional_data["ktv"], additional_data["karaoke"]]
    hospital_dfs = [hospital, small_hospital]
    hotel_dfs = [additional_data["hotel"]]

    def sum_for_cluster(df, cluster, column=None):
        mask = df["cluster"] == cluster
        return len(df[mask]) if column is None else df.loc[mask, column].sum()

    for i in range(CLUSTER_NUM):
        for df in [train_data, test_data]:
            cluster_mask = df["cluster"] == i

            df.loc[cluster_mask, "sum_drinks"] = sum(
                sum_for_cluster(df, i) for df in drink_dfs
            )
            df.loc[cluster_mask, "sum_drinks_area"] = sum(
                sum_for_cluster(df, i, "시설총규모") for df in drink_dfs
            )

            df.loc[cluster_mask, "sum_hospitals"] = sum(
                sum_for_cluster(df, i) for df in hospital_dfs
            )
            df.loc[cluster_mask, "sum_hospital_rooms"] = sum(
                sum_for_cluster(df, i, "병상수") for df in hospital_dfs
            )

            for j in range(number_of_hospital_type):
                df.loc[cluster_mask, f"sum_hospital_type_{j}"] = sum(
                    sum_for_cluster(df[df["업태구분명"] == j], i) for df in hospital_dfs
                )

            for k in range(number_of_small_hospital_type):
                df.loc[cluster_mask, f"sum_small_hospital_type_{k}"] = sum(
                    sum_for_cluster(df[df["업태구분명"] == k], i) for df in hospital_dfs
                )

            df.loc[cluster_mask, "sum_hotels"] = sum(
                sum_for_cluster(df, i) for df in hotel_dfs
            )

    return train_data, test_data


def calculate_and_save_cluster_features(kmeans, additional_data):
    cluster_features = {}
    
    for i in range(CLUSTER_NUM):
        cluster_features[i] = {
            "sum_drinks": 0,
            "sum_hospitals": 0,
            "sum_hotels": 0,
            "sum_drinks_area": 0,
            "sum_hospital_rooms": 0
        }
        
        drink_dfs = [additional_data["ktv"], additional_data["karaoke"]]
        hospital_dfs = [additional_data["hospital"], additional_data["small_hospital"]]
        hotel_dfs = [additional_data["hotel"]]
        
        for df in drink_dfs:
            df["cluster"] = kmeans.predict(df[["x_axis", "y_axis"]])
            cluster_mask = df["cluster"] == i
            cluster_features[i]["sum_drinks"] += len(df[cluster_mask])
            cluster_features[i]["sum_drinks_area"] += df.loc[cluster_mask, "시설총규모"].sum()
        
        for df in hospital_dfs:
            df["cluster"] = kmeans.predict(df[["x_axis", "y_axis"]])
            cluster_mask = df["cluster"] == i
            cluster_features[i]["sum_hospitals"] += len(df[cluster_mask])
            cluster_features[i]["sum_hospital_rooms"] += df.loc[cluster_mask, "병상수"].sum()
        
        for df in hotel_dfs:
            df["cluster"] = kmeans.predict(df[["x_axis", "y_axis"]])
            cluster_mask = df["cluster"] == i
            cluster_features[i]["sum_hotels"] += len(df[cluster_mask])
    
    #change all values in cluster_features to int
    for i in range(CLUSTER_NUM):
        for key in cluster_features[i].keys():
            cluster_features[i][key] = int(cluster_features[i][key])
    
    # Save the cluster features to a JSON file
    
    with open(f"{SAVE_PATH}/{CLUSTER_FEATURES_PATH}", "w") as f:
        json.dump(cluster_features, f)


def objective(trial, X, y):
    """Optuna objective function for XGBoost optimization."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1.0),
        "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": random_state,
        "enable_categorical": True,
    }

    model = xgb.XGBRegressor(**params)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=random_state
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    mae = np.mean(np.abs(predictions - y_val.values.ravel()))

    return mae


def train_model(train_data: pd.DataFrame) -> xgb.XGBRegressor:
    """Train XGBoost model using Optuna for hyperparameter optimization."""
    X = train_data[TRAIN_COLUMNS]
    y = train_data[
        TARGET_COLUMN[0]
    ]  # Assuming TARGET_COLUMN is a list with one element

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

    best_params = study.best_params
    best_params["random_state"] = random_state
    best_params["enable_categorical"] = True

    model = xgb.XGBRegressor(**best_params)
    model.fit(X, y)

    with open(f"{SAVE_PATH}/{XGB_MODEL_PATH}", "wb") as f:
        pickle.dump(model, f)

    print(f"Best hyperparameters: {best_params}")
    print(f"Best MAE: {study.best_value}")

    return model


def create_explainer(model: xgb.XGBRegressor, test_data: pd.DataFrame):
    """Create and save SHAP explainer."""
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')

    with open(f"{SAVE_PATH}/{EXPLAINER_PATH}", "wb") as f:
        pickle.dump(explainer, f)


def main():
    # Load and preprocess data
    train_data = load_data(TRAIN_FILE)
    test_data = load_data(TEST_FILE)

    # Get weather data
    weather_data = get_weather_data()

    # Merge weather data
    train_data = pd.merge(
        train_data, weather_data, on=["year", "month", "day", "hour"], how="left"
    )
    test_data = pd.merge(
        test_data, weather_data, on=["year", "month", "day", "hour"], how="left"
    )

    # Perform clustering
    kmeans, train_data, test_data = perform_clustering(train_data, test_data)
    print("Clustering completed.")

    # Filter clusters
    train_data, test_data, remaining_cluster = filter_clusters(train_data, test_data)

    # Save remaining clusters
    with open(f"{SAVE_PATH}/{REMAINING_CLUSTERS_PATH}", "wb") as f:
        pickle.dump(remaining_cluster, f)

    # Load additional data
    additional_data = load_additional_data()
    
    # Calculate and save cluster features
    calculate_and_save_cluster_features(kmeans, additional_data)

    # Add new columns
    train_data, test_data = add_new_columns(
        train_data, test_data, kmeans, additional_data
    )

    # Add count feature
    train_data = add_count_feature(train_data)
    test_data = add_count_feature(test_data)

    for col in FLOAT_COLUMNS:
        train_data[col] = train_data[col].astype(float)
        test_data[col] = test_data[col].astype(float)

    for col in CATEGORICAL_COLUMNS:
        train_data[col] = train_data[col].astype("category")
        test_data[col] = test_data[col].astype("category")

    for col in TARGET_COLUMN:
        train_data[col] = train_data[col].astype(float)
        test_data[col] = test_data[col].astype(float)

    # Train model
    model = train_model(train_data)
    print("XGBoost Model training with Optuna optimization completed.")

    # Create explainer
    create_explainer(model, test_data)

    # Test accuracy
    pred = model.predict(test_data[TRAIN_COLUMNS])
    mae = np.mean(np.abs(pred - test_data["count"]))
    print(f"Mean Absolute Error: {mae}")

    print("Clustering Model, XGB Model and explainer saved.")

if __name__ == "__main__":
    main()