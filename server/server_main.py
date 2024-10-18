from typing import Optional
from flask import Flask, request
from flask_socketio import Namespace, SocketIO, send, disconnect, emit
from threading import Lock, Thread
import time

from main import predict
import json
from baecha import allocate_taxis
from taxi import Taxi
import numpy as np

# Type aliases
RawCoords = tuple[np.float64, np.float64]
ClusterCoords = list[RawCoords]
BaechaResult = tuple[list[tuple[Taxi, RawCoords, np.float64]], ClusterCoords]

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

last_baecha_result: Optional[BaechaResult] = None


# def get_taxi(id) -> Taxi:
#     lat = taxis[id]['location']['lat']
#     lng = taxis[id]['location']['lng']
#     return Taxi(id=id, lat=lat, lng=lng)


# def get_taxis() -> list[Taxi]:
#     with taxiLock:
#         val = [get_taxi(id) for id, client in taxis.items()]
#    return val


class ClientNamespace(Namespace):
    def on_connect(self):
        sid: str = request.sid  # type: ignore
        print(f'Client {sid} connected.')
        with taxiLock:
            taxis[sid] = Taxi(id=sid)

        if last_baecha_result:
            notify_dashboard_baecha_result(
                *last_baecha_result, dashboard_id=sid)

    def on_disconnect(self):
        sid: str = request.sid  # type: ignore
        print(f'Client {sid} disconnected.')
        with taxiLock:
            taxis.pop(sid, None)

    def on_update(self, data):
        sid: str = request.sid  # type: ignore
        print(f'Client {sid} updated location: {data}')
        location = json.loads(data)
        taxis[sid].lat = location['lat']
        taxis[sid].lng = location['lng']

    def on_request_baecha(self):
        sid: str = request.sid  # type: ignore
        taxis[sid].state = 'WAITING'

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
        with dashboardLock:
            dashboards[sid] = {}

    def on_disconnect(self):
        sid: str = request.sid  # type: ignore
        print(f'dashboard disconnecte: {sid}')
        with dashboardLock:
            dashboards.pop(sid, None)

    def on_request_baecha(self):
        sid: str = request.sid  # type: ignore
        print(f'received baecha request from {sid}')
        run_baecha()


def run_baecha():
    global last_baecha_result

    def map_clusters(e):
        lat, lng = e[2], e[1]
        return (lat, lng)

    predicted = predict()
    with taxiLock:
        waiting_taxis = list(
            filter(lambda x: x.state == 'WAITING', taxis.values()))
        for taxi in waiting_taxis:
            taxi.state = 'RUNNING'
    clusters = list(map(map_clusters, predicted))

    results = allocate_taxis(waiting_taxis, clusters)

    print('baecha results:')
    print(results)

    notify_dashboard_baecha_result(results, clusters)

    last_baecha_result = (results, clusters)

    for result in results:
        taxi, cluster, distance = result
        notify_taxi_baecha_result(taxi, cluster)


def notify_dashboard_baecha_result(results: list[tuple[Taxi, RawCoords, np.float64]],
                                   clusters: ClusterCoords,
                                   dashboard_id=None):
    def map_results(e):
        return {
            'taxi': {
                'id': e[0].id,
                'lat': e[0].lat,
                'lng': e[0].lng,
            },
            'target': e[1],
            'distance': e[2],
        }
    data = {
        'results': list(map(map_results, results)),
        'clusters': clusters
    }

    if dashboard_id:
        socketio.emit('baecha', data, to=dashboard_id, namespace='/dashboard')
    else:
        with dashboardLock:
            for dashboard in dashboards.keys():
                socketio.emit('baecha', data, to=dashboard,
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
    with dashboardLock:
        for dashboard in dashboards.keys():
            socketio.emit('update', data, to=dashboard, namespace='/dashboard')


def notify_taxi_locations_loop():
    while True:
        notify_taxi_locations()
        time.sleep(0.5)


socketio.on_namespace(ClientNamespace('/client'))
socketio.on_namespace(DashboardNamespace('/dashboard'))


if __name__ == '__main__':
    # print(predict())

    Thread(name='notify_taxi_locations',
           target=notify_taxi_locations_loop).start()

    socketio.run(app, debug=False, host='0.0.0.0', port=5980)
