import requests

payload = {
    "MedInc": 4.0,
    "HouseAge": 30.0,
    "AveRooms": 5.5,
    "AveBedrms": 1.0,
    "Population": 1000.0,
    "AveOccup": 3.0,
    "Latitude": 34.0,
    "Longitude": -118.0
}

r = requests.post("http://127.0.0.1:8000/predict", json=payload)
print("Status:", r.status_code)
print("Response:", r.json())
