from __future__ import annotations
import os, json, argparse
from typing import Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv

from src.utils import (
    ensure_dirs, load_water_csv, build_supervised_frame, evaluate_forecast,
    seasonal_naive_forecast, PROCESSED_DIR, RAW_DIR, FIG_DIR
)

def list_raw_files():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    if not files:
        print(f"No hay CSVs en {RAW_DIR}. Ejecuta primero: python -m src.prepare_kaggle --dest data/raw")
    else:
        print("CSV encontrados en data/raw:\n - " + "\n - ".join(files))

def plot_zone(df_zone: pd.DataFrame, zone: str):
    plt.figure(figsize=(10,4))
    plt.plot(df_zone['date'], df_zone['volume'])
    plt.title(f"Zona: {zone}")
    plt.xlabel("Fecha"); plt.ylabel("Volumen")
    out = os.path.join(FIG_DIR, f"{zone}_series.png")
    plt.tight_layout(); plt.savefig(out); plt.close()
    print(f"Figura guardada: {out}")

def train_rf_for_zone(zone_df: pd.DataFrame, horizon: int, random_state: int = 42) -> Dict[str, float]:
    Xdf = build_supervised_frame(zone_df)
    n = len(Xdf)
    if n < 50:
        return {"error": "Muy pocas filas para entrenar"}
    split = int(n * 0.8)
    train, test = Xdf.iloc[:split], Xdf.iloc[split:]
    feats = [c for c in Xdf.columns if c not in ("date","zone","volume")]
    model = RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1)
    model.fit(train[feats], train["volume"])
    y_pred = model.predict(test[feats])
    metrics = evaluate_forecast(test["volume"].values, y_pred)

    last_known = Xdf.iloc[-1:].copy()
    future_rows = []
    current_date = last_known["date"].iloc[0]
    last_zone = last_known["zone"].iloc[0]
    history = Xdf[["date","volume"]].copy().reset_index(drop=True)

    for _ in range(horizon):
        current_date = current_date + pd.Timedelta(days=1)
        tmp = pd.DataFrame({"date": [current_date], "zone": [last_zone]})
        tmp_full = pd.concat([history.assign(zone=last_zone), tmp.assign(volume=np.nan)], ignore_index=True)
        tmp_full["date"] = pd.to_datetime(tmp_full["date"])
        tmp_feats = tmp_full.copy()
        tmp_feats["dow"] = tmp_feats["date"].dt.dayofweek
        tmp_feats["month"] = tmp_feats["date"].dt.month
        tmp_feats["day"] = tmp_feats["date"].dt.day
        for L in (1,7,14,28):
            tmp_feats[f"lag_{L}"] = tmp_feats["volume"].shift(L)
        for w in (7,28):
            tmp_feats[f"rollmean_{w}"] = tmp_feats["volume"].shift(1).rolling(w).mean()
            tmp_feats[f"rollstd_{w}"]  = tmp_feats["volume"].shift(1).rolling(w).std()
        row = tmp_feats.iloc[[-1]].drop(columns=["volume"]).copy()
        row = row.drop(columns=[c for c in ("date","zone") if c in row.columns], errors="ignore")
        if row.isna().any().any():
            yhat = seasonal_naive_forecast(history["volume"], 1, season_length=7)[0]
        else:
            yhat = float(model.predict(row)[0])
        history = pd.concat([history, pd.DataFrame({"date":[current_date], "volume":[yhat]})], ignore_index=True)
        future_rows.append({"date": current_date, "zone": last_zone, "prediction": yhat})
    return {**metrics, "model": "RandomForest", "horizon": horizon, "forecast": future_rows}

def train_baseline_for_zone(zone_df: pd.DataFrame, horizon: int, season_length: int = 7) -> Dict[str, float]:
    n = len(zone_df)
    split = int(n*0.8)
    train = zone_df.iloc[:split]
    test  = zone_df.iloc[split:]
    preds = []
    window = train["volume"].copy()
    i = 0
    while i < len(test):
        step = min(season_length, len(test)-i)
        yhat = seasonal_naive_forecast(window, step, season_length=season_length)
        preds.extend(yhat)
        window = pd.concat([window, test["volume"].iloc[i:i+step]], ignore_index=True)
        i += step
    metrics = evaluate_forecast(test["volume"].values, preds)
    future = seasonal_naive_forecast(zone_df["volume"], horizon, season_length=season_length)
    future_rows = []
    last_date = zone_df["date"].iloc[-1]
    last_zone = zone_df["zone"].iloc[-1]
    for h, yhat in enumerate(future, start=1):
        future_rows.append({"date": last_date + pd.Timedelta(days=h), "zone": last_zone, "prediction": float(yhat)})
    return {**metrics, "model": f"SeasonalNaive(s={season_length})", "horizon": horizon, "forecast": future_rows}

def cmd_build_dataset(args):
    import runpy
    runpy.run_path(os.path.join("src","build_dataset.py"))

def cmd_explore(args):
    df = load_water_csv()
    zones = sorted(df['zone'].unique())
    print(f"Zonas detectadas: {len(zones)}")
    for z in zones:
        zdf = df[df['zone']==z].copy()
        plot_zone(zdf, z)

def cmd_train(args):
    df = load_water_csv()
    zones_target = sorted(df['zone'].unique()) if args.zones == 'all' else [z.strip() for z in args.zones.split(',')]
    all_forecasts = []
    metrics_out = {}
    for z in tqdm(zones_target, desc="Entrenando por zona"):
        zdf = df[df['zone']==z].copy().sort_values('date')
        if len(zdf) < 60:
            print(f"Saltando zona {z} (muy pocos datos)")
            continue
        if args.model == 'baseline':
            res = train_baseline_for_zone(zdf, horizon=args.horizon, season_length=args.season)
        else:
            res = train_rf_for_zone(zdf, horizon=args.horizon)
        metrics_out[z] = {k:v for k,v in res.items() if k not in ("forecast",)}
        all_forecasts.extend(res.get("forecast", []))
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(PROCESSED_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Métricas guardadas en {os.path.join(PROCESSED_DIR, 'metrics.json')}")
    if all_forecasts:
        fdf = pd.DataFrame(all_forecasts)
        fdf = fdf.sort_values(["zone","date"]).reset_index(drop=True)
        out_csv = os.path.join(PROCESSED_DIR, "forecasts.csv")
        fdf.to_csv(out_csv, index=False)
        print(f"Pronósticos guardados en {out_csv}")

def main():
    load_dotenv()
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Proyecto Tutorial de Series Temporales - Acea Water Prediction")
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('list', help='Lista CSVs en data/raw').set_defaults(func=lambda args: list_raw_files())
    sub.add_parser('build-dataset', help='Combina CSVs en data/raw a data/raw/water.csv').set_defaults(func=cmd_build_dataset)
    sub.add_parser('explore', help='EDA básica (figuras por zona)').set_defaults(func=cmd_explore)

    tr = sub.add_parser('train', help='Entrena y pronostica por zona')
    tr.add_argument('--model', default='rf', choices=['rf','baseline'])
    tr.add_argument('--horizon', type=int, default=30)
    tr.add_argument('--season', type=int, default=7, help='Solo para baseline')
    tr.add_argument('--zones', default='all', help='"all" o lista separada por coma: Z1,Z2')
    tr.set_defaults(func=cmd_train)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        return
    args.func(args)

if __name__ == '__main__':
    main()
