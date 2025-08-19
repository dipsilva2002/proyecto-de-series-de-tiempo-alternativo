import os
import argparse
import json as jsonlib
import warnings
from typing import Tuple, Dict, List
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.utils import Paths, ensure_dirs, get_env_int, get_env_str


class MeanDummyRegressor(BaseEstimator):
    def __init__(self, mean_value: float):
        self.mean_value = float(mean_value)

    def predict(self, X):
        import numpy as np
        return np.repeat(self.mean_value, len(X))


def generate_synthetic_multizone(start="2020-01-01", periods=1000, freq="D", seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=periods, freq=freq)
    zones = ["A", "B", "C"]
    rows = []
    for z in zones:
        baseline = {"A": 120, "B": 200, "C": 80}[z]
        seasonal = 20 * np.sin(2 * np.pi * np.arange(periods) / 7.0)
        trend = np.linspace(0, 10, periods)
        noise = rng.normal(0, 8, periods)
        volume = baseline + seasonal + trend + noise
        rows.append(pd.DataFrame({"date": idx, "zone": z, "volume": volume}))
    df = pd.concat(rows, ignore_index=True)
    return df


def load_or_create_raw(p: Paths, seed: int = 42) -> pd.DataFrame:
    ensure_dirs(p)
    raw_csv = os.path.join(p.raw_dir, "water.csv")
    if os.path.exists(raw_csv):
        df = pd.read_csv(raw_csv, parse_dates=["date"])
    else:
        df = generate_synthetic_multizone(seed=seed)
        df.to_csv(raw_csv, index=False)
    df = df.dropna().copy()
    df["zone"] = df["zone"].astype(str)
    df = df.sort_values(["zone", "date"])
    return df


def plot_overview(df: pd.DataFrame, p: Paths):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for z, g in df.groupby("zone"):
        ax.plot(g["date"], g["volume"], label=f"Zone {z}", alpha=0.8)
    ax.set_title("Volumen de agua por zona (serie temporal)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Volumen")
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(p.reports_dir, "overview.png")
    plt.savefig(fig_path)
    plt.close(fig)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    return df


def make_lag_features(df: pd.DataFrame, lags: List[int] = [1, 7, 14]) -> pd.DataFrame:
    df = df.copy().sort_values(["zone", "date"])
    for L in lags:
        df[f"lag_{L}"] = df.groupby("zone")["volume"].shift(L)
    df["roll7"] = df.groupby("zone")["volume"].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
    df["roll14"] = df.groupby("zone")["volume"].rolling(14, min_periods=1).mean().reset_index(0, drop=True)
    return df


def prepare_processed(df_raw: pd.DataFrame, p: Paths) -> pd.DataFrame:
    df = add_time_features(df_raw)
    df = make_lag_features(df)
    df = df.dropna().reset_index(drop=True)
    out_csv = os.path.join(p.processed_dir, "processed.csv")
    df.to_csv(out_csv, index=False)
    return df


def seasonal_naive_forecast(zone_series: pd.Series, horizon: int, season: int = 7) -> np.ndarray:
    history = zone_series.values
    if len(history) < season:
        return np.repeat(history[-1], horizon)
    last_season = history[-season:]
    reps = int(np.ceil(horizon / season))
    fcst = np.tile(last_season, reps)[:horizon]
    return fcst


def train_sarimax_per_zone(df: pd.DataFrame, order=(1,0,1), seasonal_order=(1,0,1,7)) -> Dict[str, object]:
    models = {}
    for z, g in df.groupby("zone"):
        g = g.sort_values("date")
        endog = g["volume"].values
        model = SARIMAX(endog, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        models[z] = fit
    return models


def train_xgb_global(df_proc: pd.DataFrame):
    df = pd.get_dummies(df_proc.copy(), columns=["zone"], drop_first=False)
    feature_cols = [c for c in df.columns if c not in ["date", "volume"]]
    if not _HAS_XGB:
        print("XGBoost no disponible. Usando MeanDummyRegressor (media global).")
        dummy = MeanDummyRegressor(mean_value=df["volume"].mean())
        return dummy, feature_cols
    X = df[feature_cols]
    y = df["volume"]
    model = XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    return model, feature_cols


def metrics(y_true, y_pred) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))  # sin 'squared=' para compatibilidad
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def evaluate_models(df_raw: pd.DataFrame, df_proc: pd.DataFrame, saris: Dict[str, object], xgb: object, xgb_cols: List[str], horizon: int = 30, p: Paths = Paths()) -> Dict:
    results = {}
    for z, g in df_raw.groupby("zone"):
        g = g.sort_values("date")
        train = g.iloc[:-horizon]
        test = g.iloc[-horizon:]
        sn_pred = seasonal_naive_forecast(train["volume"], horizon=horizon, season=7)
        sn_metrics = metrics(test["volume"].values, sn_pred)
        sar = saris[z]
        sar_pred = sar.forecast(steps=horizon)
        sar_metrics = metrics(test["volume"].values, sar_pred)
        g_proc = df_proc[df_proc["zone"] == z].sort_values("date")
        X_test = g_proc.iloc[-horizon:].copy()
        X_test = pd.get_dummies(X_test, columns=["zone"], drop_first=False)
        for c in xgb_cols:
            if c not in X_test.columns:
                X_test[c] = 0
        X_test = X_test[xgb_cols]
        xgb_pred = xgb.predict(X_test)
        xgb_metrics = metrics(test["volume"].values, xgb_pred)
        results[z] = {
            "seasonal_naive": sn_metrics,
            "sarimax": sar_metrics,
            "xgb": xgb_metrics
        }
    return results


def forecast(df_raw: pd.DataFrame, df_proc: pd.DataFrame, saris: Dict[str, object], xgb: object, xgb_cols: List[str], horizon: int, p: Paths) -> pd.DataFrame:
    last_date = df_raw["date"].max()
    future_idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    out_rows = []
    for z in sorted(df_raw["zone"].unique()):
        g = df_raw[df_raw["zone"] == z].sort_values("date")
        sn_pred = seasonal_naive_forecast(g["volume"], horizon=horizon, season=7)
        sar = saris[z]
        sar_pred = sar.forecast(steps=horizon)
        g_proc = df_proc[df_proc["zone"] == z].sort_values("date")
        last = g_proc.iloc[-1:].copy()
        future_feats = []
        for dt in future_idx:
            row = last.copy()
            row["date"] = dt
            row["dayofweek"] = dt.dayofweek
            row["month"] = dt.month
            row["dayofyear"] = dt.timetuple().tm_yday
            future_feats.append(row)
        fut_df = pd.concat(future_feats, ignore_index=True)
        fut_df = pd.get_dummies(fut_df, columns=["zone"], drop_first=False)
        for c in xgb_cols:
            if c not in fut_df.columns:
                fut_df[c] = 0
        fut_df = fut_df[xgb_cols]
        xgb_pred = xgb.predict(fut_df)
        out_rows.append(pd.DataFrame({
            "date": future_idx,
            "zone": z,
            "pred_seasonal_naive": sn_pred,
            "pred_sarimax": sar_pred,
            "pred_xgb": xgb_pred
        }))
    fcst = pd.concat(out_rows, ignore_index=True)
    fcst.to_csv(os.path.join(p.processed_dir, "forecast.csv"), index=False)
    return fcst


def save_models(saris: Dict[str, object], xgb: object, p: Paths):
    import joblib
    for z, model in saris.items():
        joblib.dump(model, os.path.join(p.models_dir, f"sarimax_zone_{z}.joblib"))
    if isinstance(xgb, MeanDummyRegressor):
        print("XGB es dummy; no se guarda artefacto xgb_global.joblib.")
    else:
        joblib.dump(xgb, os.path.join(p.models_dir, "xgb_global.joblib"))


def save_metrics(metrics_dict: Dict, p: Paths):
    with open(os.path.join(p.models_dir, "metrics.json"), "w") as f:
        jsonlib.dump(metrics_dict, f, indent=2)


def plot_forecast(fcst: pd.DataFrame, df_raw: pd.DataFrame, p: Paths):
    for z in sorted(df_raw["zone"].unique()):
        hist = df_raw[df_raw["zone"] == z].sort_values("date")
        fut = fcst[fcst["zone"] == z].sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hist["date"], hist["volume"], label="Histórico")
        ax.plot(fut["date"], fut["pred_seasonal_naive"], "--", label="SN")
        ax.plot(fut["date"], fut["pred_sarimax"], "--", label="SARIMAX")
        ax.plot(fut["date"], fut["pred_xgb"], "--", label="XGB")
        ax.set_title(f"Zona {z} - Forecast")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(p.reports_dir, f"forecast_zone_{z}.png"))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Proyecto de Series Temporales: Volumen de agua por zonas")
    parser.add_argument("--prepare", action="store_true", help="Crea/limpia datos y features")
    parser.add_argument("--train", action="store_true", help="Entrena modelos")
    parser.add_argument("--evaluate", action="store_true", help="Evalúa modelos")
    parser.add_argument("--forecast", action="store_true", help="Genera pronóstico")
    parser.add_argument("--horizon", type=int, default=None, help="Horizonte de forecast")
    parser.add_argument("--all", action="store_true", help="Ejecuta prepare+train+evaluate+forecast")
    args = parser.parse_args()

    seed = get_env_int("SEED", 42)
    default_h = get_env_int("FORECAST_HORIZON", 30)
    horizon = args.horizon if args.horizon is not None else default_h

    p = Paths()
    ensure_dirs(p)

    if args.prepare or args.all:
        df_raw = load_or_create_raw(p, seed=seed)
        plot_overview(df_raw, p)
        df_proc = prepare_processed(df_raw, p)
        print("PREPARE listo: data/raw/water.csv, data/processed/processed.csv y reports/figures/overview.png")

    if not (args.prepare or args.all):
        df_raw = pd.read_csv(os.path.join(p.raw_dir, "water.csv"), parse_dates=["date"])
        df_proc = pd.read_csv(os.path.join(p.processed_dir, "processed.csv"), parse_dates=["date"])

    if args.train or args.all:
        saris = train_sarimax_per_zone(df_raw)
        xgb, xgb_cols = train_xgb_global(df_proc)
        save_models(saris, xgb, p)
        with open(os.path.join(p.models_dir, "xgb_columns.json"), "w") as f:
            jsonlib.dump(xgb_cols, f)
        print("modelos guardados en /models")

    if not (args.train or args.all):
        import joblib
        saris = {}
        for z in sorted(df_raw["zone"].unique()):
            saris[z] = joblib.load(os.path.join(p.models_dir, f"sarimax_zone_{z}.joblib"))
        with open(os.path.join(p.models_dir, "xgb_columns.json")) as f:
            xgb_cols = jsonlib.load(f)
        xgb = joblib.load(os.path.join(p.models_dir, "xgb_global.joblib"))

    if args.evaluate or args.all:
        res = evaluate_models(df_raw, df_proc, saris, xgb, xgb_cols, horizon=horizon, p=p)
        save_metrics(res, p)
        print("EVALUATE listo: métricas en models/metrics.json")

    if args.forecast or args.all:
        fcst = forecast(df_raw, df_proc, saris, xgb, xgb_cols, horizon=horizon, p=p)
        plot_forecast(fcst, df_raw, p)
        print(f"FORECAST listo: forecast.csv y figuras por zona en {p.reports_dir}")


if __name__ == "__main__":
    main()
