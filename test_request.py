"""Send a sample request to the running service (local, Docker or Kubernetes port-forward)."""
import sys

import requests

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

trip = {
    "distance": 3.2,
    "passengers": 1,
    "color": "yellow",
    "pickup_hour": 18,
    "pickup_dayofweek": 2,
    "pickup_borough": "Manhattan",
    "dropoff_borough": "Brooklyn",
}

print("GET  /health  ->", requests.get(f"{url}/health", timeout=10).json())
r = requests.post(f"{url}/predict", json=trip, timeout=10)
print("POST /predict ->", r.status_code, r.json())
