from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

FIG_DIR = os.path.join("reports", "figures")
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = "models"

def ensure_dirs():
    for d in [FIG_DIR, DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR]:
        os.makedirs(d, exist_ok=True)

def load_water_csv(path: str = os.path.join(RAW_DIR, "water.csv")) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}. Ejecuta primero: python -m src.build_dataset")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["zone", "date"]).reset_index(drop=True)
    return df

def make_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    return df

def add_lagged_features(df: pd.DataFrame, lags: Iterable[int], roll_windows: Iterable[int]) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    for L in lags:
        df[f"lag_{L}"] = df["volume"].shift(L)
    for w in roll_windows:
        df[f"rollmean_{w}"] = df["volume"].shift(1).rolling(w).mean()
        df[f"rollstd_{w}"]  = df["volume"].shift(1).rolling(w).std()
    return df

def build_supervised_frame(zone_df: pd.DataFrame, lags=(1,7,14,28), rolls=(7,28)) -> pd.DataFrame:
    x = zone_df.copy()
    x = make_calendar_features(x)
    x = add_lagged_features(x, lags, rolls)
    x = x.dropna().reset_index(drop=True)
    return x

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def evaluate_forecast(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }

def seasonal_naive_forecast(series: pd.Series, horizon: int, season_length: int = 7) -> List[float]:
    hist = series.values
    if len(hist) < season_length:
        return [float(hist[-1])] * horizon
    pattern = hist[-season_length:]
    out = []
    for i in range(horizon):
        out.append(float(pattern[i % season_length]))
    return out


def get_env_str(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)

def get_env_int(key: str, default: int | None = None) -> int | None:
    val = os.getenv(key, None)
    try:
        return int(val) if val is not None and val != "" else default
    except ValueError:
        return default

@dataclass(frozen=True)
class Paths:
    RAW_DIR: str = RAW_DIR
    INTERIM_DIR: str = INTERIM_DIR
    PROCESSED_DIR: str = PROCESSED_DIR
    FIG_DIR: str = FIG_DIR
    MODELS_DIR: str = MODELS_DIR
