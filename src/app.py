from __future__ import annotations
import os, json, argparse, warnings
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

DATA_URL_DEFAULT = "https://raw.githubusercontent.com/4GeeksAcademy/alternative-time-series-project/main/sales.csv"


def ensure_dirs():
    for d in ["data/raw", "data/interim", "data/processed", "models", "reports/figures"]:
        os.makedirs(d, exist_ok=True)


def load_sales_dataframe(source: str) -> pd.DataFrame:
    """
    Carga el CSV desde URL o ruta local. Busca columna de fecha y de ventas,
    normaliza nombres y asegura frecuencia regular (mensual por defecto si no se infiere).
    """
    df = pd.read_csv(source)


    df.columns = [c.lower() for c in df.columns]
    date_candidates = ["date", "ds", "fecha", "time", "month"]
    value_candidates = ["sales", "y", "valor", "value", "amount"]

    date_col = next((c for c in date_candidates if c in df.columns), None)
    val_col  = next((c for c in value_candidates if c in df.columns), None)
    if date_col is None or val_col is None:
        raise ValueError(f"No encontré columnas de fecha/ventas en: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors="coerce")
    df = df.dropna(subset=[date_col, val_col]).sort_values(date_col)
    df = df.set_index(date_col).rename(columns={val_col: "sales"})

    
    if df.index.inferred_freq is None:
        df = df.resample("MS").sum()
    else:
        df = df.asfreq(df.index.inferred_freq)

    df.to_csv("data/processed/sales_processed.csv")
    return df


def plot_series(df: pd.DataFrame, title: str, path: str):
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df["sales"])
    plt.title(title)
    plt.xlabel("Fecha"); plt.ylabel("Ventas")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def adf_test(series: pd.Series) -> dict:
    res = adfuller(series.dropna(), autolag="AIC")
    keys = ["adf_statistic", "p_value", "used_lag", "n_obs", "critical_values", "icbest"]
    out = dict(zip(keys, [res[0], res[1], res[2], res[3], res[4], res[5]]))
    return out

def pick_seasonal_period(freq_str: str | None) -> int | None:
    if not freq_str:
        return None
    f = freq_str.upper()
    if "M" in f:   
        return 12
    if "W" in f:   
        return 52
    if f == "D":   
        return 7
    return None

def plot_decomposition(df: pd.DataFrame, period: int, out_prefix: str):
    dec = seasonal_decompose(df["sales"], model="additive", period=period, extrapolate_trend="freq")
    fig = dec.plot()
    fig.set_size_inches(10,8); fig.tight_layout()
    plt.savefig(f"{out_prefix}_decompose.png", dpi=150); plt.close()
    return {
        "trend_head": dec.trend.dropna().head(3).round(3).tolist(),
        "seasonal_head": dec.seasonal.dropna().head(3).round(3).tolist(),
        "resid_head": dec.resid.dropna().head(3).round(3).tolist()
    }

def train_test_split_series(df: pd.DataFrame, test_periods: int = 12):
    train = df.iloc[:-test_periods].copy()
    test  = df.iloc[-test_periods:].copy()
    return train, test


def sarimax_grid_search(y_train: pd.Series, seasonal_period: int | None):

    p = d = q = [0,1,2]
    P = D = Q = [0,1]
    m = seasonal_period if (seasonal_period and seasonal_period > 1) else 0

    best_aic = np.inf
    best_cfg = None
    best_res = None

    for order in itertools.product(p, d, q):
        if m > 0:
            for sorder in itertools.product(P, D, Q):
                try:
                    res = SARIMAX(
                        y_train,
                        order=order,
                        seasonal_order=(sorder[0], sorder[1], sorder[2], m),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    ).fit(disp=False)
                    if res.aic < best_aic:
                        best_aic, best_cfg, best_res = res.aic, (order, (sorder[0], sorder[1], sorder[2], m)), res
                except Exception:
                    continue
        else:
            try:
                res = SARIMAX(
                    y_train,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
                if res.aic < best_aic:
                    best_aic, best_cfg, best_res = res.aic, (order, None), res
            except Exception:
                continue

    if best_res is None:
        raise RuntimeError("No se pudo ajustar ningún SARIMAX; revisa datos/periodicidad.")
    return best_res, best_cfg


def metrics_dict(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_true, y_pred)
 
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = (np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))).mean() * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}


def plot_forecast(train: pd.DataFrame, test: pd.DataFrame, pred: pd.Series, path: str, title: str = "Predicción de Ventas"):
    plt.figure(figsize=(10,5))
    plt.plot(train.index, train["sales"], label="Train")
    plt.plot(test.index, test["sales"], label="Test")
    plt.plot(pred.index, pred.values, label="Predicción", linestyle="--")
    plt.title(title); plt.xlabel("Fecha"); plt.ylabel("Ventas"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-url", default=DATA_URL_DEFAULT, help="URL o ruta local del CSV de ventas")
    parser.add_argument("--test-periods", type=int, default=12, help="Períodos para test (p.ej., 12 meses)")
    parser.add_argument("--seasonal-period", type=int, default=-1, help="Forzar período estacional (e.g., 12). -1 = auto")
    args = parser.parse_args()

    ensure_dirs()


    print(f"Cargando datos desde: {args.data_url}")
    df = load_sales_dataframe(args.data_url)
    print(f"Filas: {len(df)} | índice: {df.index.min().date()} → {df.index.max().date()} | freq: {df.index.inferred_freq}")


    freq = df.index.inferred_freq
    tensor = freq if freq is not None else "No inferido (regularizado a mensual MS)"
    plot_series(df, "Serie de Ventas", "reports/figures/series_ventas.png")

    seasonal_period = args.seasonal_period if args.seasonal_period > 0 else pick_seasonal_period(freq)
    if seasonal_period is None:
        seasonal_period = 12  
    deco_info = plot_decomposition(df, seasonal_period, "reports/figures/series")

    adf_res = adf_test(df["sales"])
    analysis = {
        "tensor_unidad_tiempo": str(tensor),
        "tendencia_muestra_trend_head": deco_info["trend_head"],
        "estacionalidad_periodo_usado": int(seasonal_period),
        "ruido_muestra_resid_head": deco_info["resid_head"],
        "adf_test": {
            "adf_statistic": adf_res["adf_statistic"],
            "p_value": adf_res["p_value"],
            "critical_values": adf_res["critical_values"]
        },
        "nota_estacionaria": "Probablemente NO estacionaria si p_value >= 0.05; SARIMAX maneja diferenciación via d/D."
    }
    with open("reports/analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)


    train_df, test_df = train_test_split_series(df, test_periods=args.test_periods)
    y_train, y_test = train_df["sales"], test_df["sales"]

   
    print("Buscando mejor configuración SARIMAX (grid pequeño)...")
    model_res, model_cfg = sarimax_grid_search(y_train, seasonal_period)
    print("Mejor configuración:", model_cfg, "| AIC:", round(model_res.aic, 2))


    n_test = len(y_test)
    y_pred = pd.Series(model_res.forecast(steps=n_test), index=y_test.index)

  
    mets = metrics_dict(y_test.values, y_pred.values)
    with open("models/metrics.json", "w") as f:
        json.dump(mets, f, indent=2)
    print("Métricas:", mets)

    plot_forecast(train_df, test_df, y_pred, "reports/figures/forecast_vs_real.png")

 
    model_res.save("models/arima.pkl")
    print("Modelo guardado en models/arima.pkl")
    print("Listo. Figuras en reports/figures/, métricas en models/metrics.json, análisis en reports/analysis.json.")

if __name__ == "__main__":
    main()
