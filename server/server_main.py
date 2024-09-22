from flask import Flask, request
from flask_socketio import Namespace, SocketIO, send, disconnect, emit
from threading import Lock, Thread
import time
from main import predict
import json
from baecha import allocate_taxis
from taxi import Taxi
import numpy as np

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app,
                    ping_interval=5,
                    ping_timeout=10,
                    cors_allowed_origins=[],
                    )
clients = {}
clientLock = Lock()
dashboards = {}
dashboardLock = Lock()
bg_thread_started = False

baecha_callbacks = {}


class ClientNamespace(Namespace):
    def on_connect(self):
        with clientLock:
            clients[request.sid] = {}
            clients[request.sid]['state'] = 'WAITING'
        print(f'Client {request.sid} connected.')

    def on_disconnect(self):
        with clientLock:
            clients.pop(request.sid, None)
            baecha_callbacks.pop(request.sid, None)
        print(f'Client {request.sid} disconnected.')

    def on_update(self, data):
        print(f'Client {request.sid} updated location: {data}')
        clients[request.sid]['location'] = json.loads(data)

    def on_message(self, msg):
        print(f'Received message from {request.sid}: {msg}')

    def on_state(self, state):
        with clientLock:
            if state not in ['WAITING', 'RUNNING']:
                print(f'{request.sid} wrong state: {state}')
                return
            clients[request.sid]['state'] = state

    def on_pong(self):
        print(f'Pong received from {request.sid}')

    def on_ping(self):
        print(f'Ping received from {request.sid}')


class DashboardNamespace(Namespace):
    def on_connect(self):
        # 현재 배차 정보 보내기
        with dashboardLock:
            dashboards[request.sid] = {}
        print(f'dashboard connected: {request.sid}')

    def on_disconnect(self):
        with dashboardLock:
            dashboards.pop(request.sid, None)
        print(f'dashboard disconnecte: {request.sid}')

    def on_request_baecha(self):
        print(f'received baecha request from {request.sid}')

        def map_taxis(e):
            lat = clients[e]['location']['lat']
            lng = clients[e]['location']['lng']
            return Taxi(id=e, lat=lat, lng=lng)

        def map_passengers(e):
            lat, lng = e[2], e[1]
            return (lat, lng)
        
        clusters = predict()
        taxis = list(map(map_taxis, clients))
        passengers = list(map(map_passengers, clusters))

        results = allocate_taxis(taxis, passengers)

        print('baecha results:')
        print(results)

        notify_dashboard_results(results, passengers)
        for result in results:
            taxi, passenger, distance = result
            baecha(taxi, passenger)
    

def notify_dashboard_results(results, passengers):
    def map_results(e):
        return {
            'taxi':{
                'id':e[0].id,
                'lat':e[0].lat,
                'lng':e[0].lng,
            },
            'target':e[1],
            'distance':e[2],
        }
    data = {
        'results': list(map(map_results, results)),
        'passengers': passengers
    }
    for dashboard in dashboards.keys():
        socketio.emit('baecha', data, to=dashboard, namespace='/dashboard')


def baecha(taxi: Taxi, passenger: tuple[np.float64, np.float64]):
    lat, lng = passenger
    # baecha_callbacks[taxi.id](passenger[0], passenger[1])
    if taxi.id in clients:
        socketio.emit('baecha', {"lat":lat, "lng":lng}, to=taxi.id, namespace='/client')



socketio.on_namespace(ClientNamespace('/client'))
socketio.on_namespace(DashboardNamespace('/dashboard'))


if __name__ == '__main__':
    # print(predict())

    # Thread(name='background', target=background_loop).start()

    socketio.run(app, debug=False, host='0.0.0.0', port=5980)
