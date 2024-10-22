from typing import Optional, TypedDict
from flask import Flask, request
from flask_socketio import Namespace, SocketIO, send, disconnect, emit
from threading import Lock, Thread
import time
import json
import numpy as np

from server import Taxi as ModelTaxi, Cluster as ModelCluster, taxis as modelTaxis, initialize, return_model_values, set_assign_callback, set_initialize_callback, set_predict_callback

# Model 관련 코드 시작

model = {}


def on_model_initialize():
    model['kmeans'], model['clusters'], model['cluster_features'], model['remaining_clusters'], model['model'], model['explainer'], \
        model['distance_rate'], model['competition_rate'], model['demand_rate'], model['weather_API'], model['train_columns'], \
        model['observer'] = return_model_values()
    print('initialize complete!!!')


set_initialize_callback(on_model_initialize)


# Model 관련 코드 끝

class Taxi:
    def __init__(self, id: str, lat: Optional[float] = None, lng: Optional[float] = None, state: str = 'IDLE'):
        self.id = id
        self.lat = lat
        self.lng = lng
        self.state = state

    def __repr__(self):
        return f"Taxi({self.id}, {self.lat}, {self.lng}, {self.state})"


# Type aliases
RawCoords = tuple[np.float64, np.float64]
ClusterCoords = list[RawCoords]
# BaechaResult = tuple[list[tuple[Taxi, RawCoords, np.float64]], ClusterCoords]
BaechaResult = dict[str, ModelCluster]  # key: Taxi ID

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app,
                    ping_interval=5,
                    ping_timeout=10,
                    cors_allowed_origins=[],
                    )
taxis: dict[str, Taxi] = {}
taxiLock = Lock()
dashboards = {}
dashboardLock = Lock()

last_predict_result: Optional[list[ModelCluster]] = None
last_baecha_result: Optional[BaechaResult] = None


class ClientNamespace(Namespace):
    def on_connect(self):
        sid: str = request.sid  # type: ignore
        print(f'Client {sid} connected.')
        with taxiLock:
            taxis[sid] = Taxi(id=sid)
        model['observer'].set_taxi(
            ModelTaxi(name=sid, x_axis=0, y_axis=0, status='connected')
        )

    def on_disconnect(self):
        sid: str = request.sid  # type: ignore
        print(f'Client {sid} disconnected.')
        with taxiLock:
            taxis.pop(sid, None)
        print('now calling set_taxi')
        model['observer'].set_taxi(
            ModelTaxi(name=sid, x_axis=0, y_axis=0, status='disconnected')
        )

    def on_update(self, data):
        sid: str = request.sid  # type: ignore
        # print(f'Client {sid} updated location: {data}')
        location = json.loads(data)
        lat, lng = location['lat'], location['lng']
        taxis[sid].lat = lat
        taxis[sid].lng = lng
        modelTaxis[sid].set_location(lng, lat)

    def on_request_baecha(self):
        sid: str = request.sid  # type: ignore
        taxis[sid].state = 'WAITING'
        if not taxis[sid].lat or not taxis[sid].lng:
            print(f'Client {sid} not ready yet but requested baecha')
        model['observer'].set_taxi(
            ModelTaxi(
                name=sid,
                x_axis=taxis[sid].lng,  # type: ignore
                y_axis=taxis[sid].lat,  # type: ignore
                status='to_cluster',
            )
        )

    def on_message(self, msg):
        sid: str = request.sid  # type: ignore
        print(f'Received message from {sid}: {msg}')

    def on_state(self, state):
        sid: str = request.sid  # type: ignore
        with taxiLock:
            if state not in ['IDLE', 'WAITING', 'RUNNING']:
                print(f'{sid} wrong state: {state}')
                return
            taxis[sid].state = state

    def on_pong(self):
        sid: str = request.sid  # type: ignore
        print(f'Pong received from {sid}')

    def on_ping(self):
        sid: str = request.sid  # type: ignore
        print(f'Ping received from {sid}')


class DashboardNamespace(Namespace):
    def on_connect(self):
        sid: str = request.sid  # type: ignore
        print(f'dashboard connected: {sid}')
        # with dashboardLock:
        dashboards[sid] = {}

        if last_predict_result:
            notify_dashboard_predict_result(dashboard_id=sid)

        # This was originally in ClientNamespace. Why?
        if last_baecha_result:
            notify_dashboard_baecha_result(
                last_baecha_result, dashboard_id=sid)

    def on_disconnect(self):
        sid: str = request.sid  # type: ignore
        print(f'dashboard disconnecte: {sid}')
        # with dashboardLock:
        dashboards.pop(sid, None)

    # Not required anymore, model will continuously run in server.py
    def on_request_baecha(self):
        sid: str = request.sid  # type: ignore
        print(f'received baecha request from {sid}')
        # run_baecha()


# Deprecated, DO NOT USE
# def run_baecha():
#     global last_baecha_result

#     def map_clusters(e):
#         lat, lng = e[2], e[1]
#         return (lat, lng)

#     predicted = predict()
#     with taxiLock:
#         waiting_taxis = list(
#             filter(lambda x: x.state == 'WAITING', taxis.values()))
#         for taxi in waiting_taxis:
#             taxi.state = 'RUNNING'
#     clusters = list(map(map_clusters, predicted))

#     results = allocate_taxis(waiting_taxis, clusters)

#     print('baecha results:')
#     print(results)
#     on_baecha(results, clusters)

AssignmentResult = dict[str, ModelCluster]  # taxi id -> 배차된 클러스터 정보
PredictResult = dict[str, ModelCluster]  # cluster id -> 해당 클러스터 정보


def on_baecha_complete():
    result: AssignmentResult = model['observer'].assignments

    notify_dashboard_baecha_result(result)

    last_baecha_result = result

    for taxi_id, cluster in result.items():
        taxi_id: str
        cluster: ModelCluster

        # taxi, cluster, distance = result
        notify_taxi_baecha_result(taxis[taxi_id], (np.float64(
            cluster.y_axis), np.float64(cluster.x_axis)))


def notify_dashboard_baecha_result(result: AssignmentResult,
                                   dashboard_id=None):

    # def map_results(e):
    #     return {
    #         'taxi': {
    #             'id': e[0].id,
    #             'lat': e[0].lat,
    #             'lng': e[0].lng,
    #         },
    #         'target': [e.cluster],
    #         # 'distance': e[2],
    #     }
    data = {
        'results': [],
        'clusters': {}
    }

    for taxi_id, cluster in result.items():
        taxi_id: str
        cluster: ModelCluster

        # TODO: 연결 해제한 택시에 대한 배차가 됐을 때도 고려하여
        # taxi_id가 taxis에 존재하는지에 대한 체크에 전반적으로 필요함.

        data['results'].append({
            'taxi': {
                'id': taxi_id,
                'lat': taxis[taxi_id].lat,
                'lng': taxis[taxi_id].lng,
            },
            'target': str(cluster.name),
            'reason': cluster.predicted_reason
        })

    data['clusters'] = {str(cluster.name): [cluster.y_axis, cluster.x_axis]
                        for cluster in model['clusters']}

    if dashboard_id:
        socketio.emit('baecha', data, to=dashboard_id, namespace='/dashboard')
    else:
        # with dashboardLock:
        for dashboard in dashboards.keys():
            socketio.emit('baecha', data, to=dashboard,
                          namespace='/dashboard')


# 서버에서 predict가 실행되어 클러스터 정보가 바뀌었을 때 대시보드에 알림
# TODO: 추후 클러스터 경쟁률 정보 등도 알릴 필요 있으면 여기서 하기
def notify_dashboard_predict_result(dashboard_id=None):
    data = {
        'clusters': {}
    }

    for cluster in model['clusters']:
        cluster: ModelCluster

        # TODO: 연결 해제한 택시에 대한 배차가 됐을 때도 고려하여
        # taxi_id가 taxis에 존재하는지에 대한 체크에 전반적으로 필요함.

        data['clusters'][cluster.name] = {
            'coords': [cluster.y_axis, cluster.x_axis],
            'demand': cluster.predicted_demand,
            'reason': cluster.predicted_reason
        }

    if dashboard_id:
        socketio.emit('predict', data, to=dashboard_id, namespace='/dashboard')
    else:
        # with dashboardLock:
        for dashboard in dashboards.keys():
            socketio.emit('predict', data, to=dashboard,
                          namespace='/dashboard')


def notify_taxi_baecha_result(taxi: Taxi, cluster: tuple[np.float64, np.float64]):
    lat, lng = cluster
    # baecha_callbacks[taxi.id](cluster[0], cluster[1])
    with taxiLock:
        if taxi.id in taxis:
            socketio.emit('baecha', {"lat": lat, "lng": lng},
                          to=taxi.id, namespace='/client')

# 대시보드에 변경된 택시 위치 정보를 업데이트


def notify_taxi_locations():
    if not dashboards:
        return

    data = [{'id': taxi.id, 'lat': taxi.lat, 'lng': taxi.lng}
            for taxi in taxis.values()]
    # with dashboardLock:
    for dashboard in dashboards.keys():
        socketio.emit('update', data, to=dashboard, namespace='/dashboard')


def notify_taxi_locations_loop():
    while True:
        notify_taxi_locations()
        time.sleep(0.5)


# Set server callbacks


def on_assign():
    # print(model)
    print('assign done!!!')
    on_baecha_complete()


set_assign_callback(on_assign)


def on_predict():
    global last_predict_result
    print('predict done!!!')
    last_predict_result = model['clusters']
    notify_dashboard_predict_result()


set_predict_callback(on_predict)


socketio.on_namespace(ClientNamespace('/client'))
socketio.on_namespace(DashboardNamespace('/dashboard'))


if __name__ == '__main__':
    # print(predict())

    Thread(name='notify_taxi_locations',
           target=notify_taxi_locations_loop).start()

    socketio.run(app, debug=False, host='0.0.0.0', port=5980)
