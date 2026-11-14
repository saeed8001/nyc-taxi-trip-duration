"""Train the final NYC taxi trip-duration model and save it to model.bin.

Usage:  python train.py
The hyper-parameters below were selected in notebook.ipynb (best validation RMSE).
"""
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = "data/taxis.csv"
MODEL_PATH = "model.bin"

NUM = ["distance", "log_distance", "passengers", "pickup_hour", "pickup_dayofweek"]
CAT = ["color", "pickup_borough", "dropoff_borough"]
FEATURES = NUM + CAT
TARGET = "duration_min"

# Best configuration found in the notebook (XGBoost)
XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=1,
    n_jobs=-1,
)


def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Columns:", list(df.columns), "| shape:", df.shape)

    df["pickup"] = pd.to_datetime(df["pickup"])
    df["dropoff"] = pd.to_datetime(df["dropoff"])
    df[TARGET] = (df["dropoff"] - df["pickup"]).dt.total_seconds() / 60

    # remove impossible trips
    bad = (df["distance"] <= 0) | (df[TARGET] < 1) | (df[TARGET] > 120)
    print(f"Removed {bad.sum()} invalid rows")
    df = df[~bad].copy()

    for c in ["pickup_borough", "dropoff_borough"]:
        df[c] = df[c].fillna("Unknown")

    df["pickup_hour"] = df["pickup"].dt.hour
    df["pickup_dayofweek"] = df["pickup"].dt.dayofweek
    df["log_distance"] = np.log1p(df["distance"])
    return df


def main():
    df = load_and_prepare(DATA_PATH)

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    print(f"full_train = {len(df_full_train)} rows | test = {len(df_test)} rows")

    pipeline = Pipeline(
        [("dv", DictVectorizer(sparse=True)), ("model", xgb.XGBRegressor(**XGB_PARAMS))]
    )
    pipeline.fit(
        df_full_train[FEATURES].to_dict(orient="records"),
        np.log1p(df_full_train[TARGET].values),  # train on log(1 + minutes)
    )

    pred = np.expm1(pipeline.predict(df_test[FEATURES].to_dict(orient="records")))
    y = df_test[TARGET].values
    print(
        f"TEST  RMSE = {np.sqrt(mean_squared_error(y, pred)):.3f} min | "
        f"MAE = {mean_absolute_error(y, pred):.3f} min | R2 = {r2_score(y, pred):.3f}"
    )

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
