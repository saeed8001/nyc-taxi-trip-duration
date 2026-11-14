"""FastAPI service that serves the trained taxi trip-duration model.

Run locally:  uvicorn predict:app --host 0.0.0.0 --port 8000
Docs (Swagger UI): http://localhost:8000/docs
"""
import os
import pickle
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "model.bin")
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(MODEL_PATH, "rb") as f:
        state["model"] = pickle.load(f)
    yield
    state.clear()


app = FastAPI(title="NYC Taxi Trip Duration", version="1.0", lifespan=lifespan)


class Trip(BaseModel):
    distance: float = Field(..., gt=0, le=100, description="trip distance in miles")
    passengers: int = Field(1, ge=0, le=6)
    color: str = Field("yellow", description="'yellow' or 'green'")
    pickup_hour: int = Field(..., ge=0, le=23)
    pickup_dayofweek: int = Field(..., ge=0, le=6, description="0 = Monday ... 6 = Sunday")
    pickup_borough: str = Field("Manhattan", description="Manhattan, Brooklyn, Queens, Bronx, Staten Island, Unknown")
    dropoff_borough: str = Field("Manhattan")

    model_config = {
        "json_schema_extra": {
            "example": {
                "distance": 3.2,
                "passengers": 1,
                "color": "yellow",
                "pickup_hour": 18,
                "pickup_dayofweek": 2,
                "pickup_borough": "Manhattan",
                "dropoff_borough": "Brooklyn",
            }
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(trip: Trip):
    row = trip.model_dump()
    row["log_distance"] = float(np.log1p(row["distance"]))
    minutes = float(np.expm1(state["model"].predict([row])[0]))
    return {"duration_minutes": round(minutes, 2)}
