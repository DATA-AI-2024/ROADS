from typing import Optional


class Taxi:
    def __init__(self, id: str, lat: Optional[float] = None, lng: Optional[float] = None, state: str = 'IDLE'):
        self.id = id
        self.lat = lat
        self.lng = lng
        self.state = state

    def __repr__(self):
        return f"Taxi({self.id}, {self.lat}, {self.lng}, {self.state})"
