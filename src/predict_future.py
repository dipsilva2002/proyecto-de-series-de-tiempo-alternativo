import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.tseries.frequencies import to_offset
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


def infer_freq_safe(idx):

    freq = idx.inferred_freq
    if freq is None:
        try:
            freq = pd.infer_freq(idx)
        except Exception:
            freq = None
    return freq or "D"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=30, help="Pasos a pronosticar hacia adelante")
    parser.add_argument("--out-prefix", dest="out_prefix", default="forecast_next", help="Prefijo para archivos de salida")
    args = parser.parse_args()

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

   
    df = pd.read_csv("data/processed/sales_processed.csv", parse_dates=[0], index_col=0)
    model_path = Path("models/arima.pkl")
    if not model_path.exists():
        raise FileNotFoundError("No se encontró models/arima.pkl. Primero ejecuta: python src/app.py")

    res = SARIMAXResults.load(model_path.as_posix())



    horizon = int(args.horizon)
    freq = infer_freq_safe(df.index)
    start = df.index[-1] + to_offset(freq)
    future_idx = pd.date_range(start, periods=horizon, freq=freq)


    fc_values = res.forecast(steps=horizon)
    fc = pd.Series(fc_values, index=future_idx)


    csv_path = Path("reports") / f"{args.out_prefix}{horizon}.csv"
    fc.to_csv(csv_path, header=["forecast"])


    lookback = max(120, horizon * 2)
    plt.figure(figsize=(10,5))
    plt.plot(df.index[-lookback:], df["sales"].iloc[-lookback:], label="Histórico")
    plt.plot(fc.index, fc.values, "--", label=f"Pronóstico {horizon}")
    plt.title(f"Pronóstico de Ventas ({horizon} pasos)")
    plt.xlabel("Fecha"); plt.ylabel("Ventas"); plt.legend()
    fig_path = Path("reports/figures") / f"{args.out_prefix}{horizon}.png"
    plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()

    print(f"OK -> {csv_path} y {fig_path}")

if __name__ == "__main__":
    main()
