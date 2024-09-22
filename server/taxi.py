class Taxi:
    def __init__(self, id: str, lat: float, lng: float):
        self.id = id
        self.lat = lat
        self.lng = lng

    def __repr__(self):
        return f"Taxi({self.id}, {self.lat}, {self.lng})"