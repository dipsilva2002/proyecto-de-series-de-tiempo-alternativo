import os
import re
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
OUT_CSV = os.path.join(RAW_DIR, "water.csv")

DATE_CANDIDATES = [
    r"^date$", r"^data$", r"^fecha$", r"^time$", r"^timestamp$"
]


VOLUME_PATTERNS = [
    r"flow", r"discharge", r"depth", r"level", r"rain", r"rainfall",
    r"precip", r"temperature", r"temp", r"volume"
]

def find_date_col(cols):
    cols_norm = [c.strip().lower() for c in cols]

    for pat in DATE_CANDIDATES:
        for c in cols_norm:
            if re.search(pat, c):
                return cols[cols_norm.index(c)]
   
    for c in cols:
        try:
            pd.to_datetime(pd.Series([ "2000-01-01", "2001-02-03" ]))  
            s = pd.to_datetime(pd.Series([ "2000-01-01", "2001-02-03" ])) 
           
        except Exception:
            pass
  
    for c in cols:
        if 'date' in c.lower():
            return c
    return None

def find_volume_col(cols):
    cols_norm = [c.strip().lower() for c in cols]
    for pat in VOLUME_PATTERNS:
        for i, c in enumerate(cols_norm):
            if pat in c:
                return cols[i]
   
    return None

def try_read(path):
   
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        pass
    
    try:
        return pd.read_csv(path, sep=";")
    except Exception:
        pass
    
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")
    except Exception:
        pass
   
    if path.lower().endswith((".xls", ".xlsx")):
        try:
            return pd.read_excel(path, sheet_name=0)
        except Exception:
            pass
    raise RuntimeError(f"No se pudo leer el archivo: {path}")

def normalize_one_file(path):
    df = try_read(path)
   
    df.columns = [str(c).strip() for c in df.columns]

    
    date_col = find_date_col(list(df.columns))
    if date_col is None:
       
        first = df.columns[0]
        try:
            parsed = pd.to_datetime(df[first], errors="coerce", dayfirst=False, infer_datetime_format=True)
            if parsed.notna().mean() > 0.8:
                date_col = first
                df[first] = parsed
        except Exception:
            pass
    if date_col is None:
        
        best = None; best_ratio = 0
        for c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=False, infer_datetime_format=True)
            ratio = parsed.notna().mean()
            if ratio > best_ratio:
                best_ratio, best = ratio, c
        if best is not None and best_ratio > 0.6:
            date_col = best
            df[best] = pd.to_datetime(df[best], errors="coerce")

    if date_col is None:
        return None  

   
    vol_col = find_volume_col([c for c in df.columns if c != date_col])
    if vol_col is None:
        
        num_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            vol_col = num_cols[0]
        else:
            
            for c in df.columns:
                if c == date_col: 
                    continue
                try:
                    asnum = pd.to_numeric(df[c], errors="coerce")
                    if asnum.notna().mean() > 0.8:
                        df[c] = asnum
                        vol_col = c
                        break
                except Exception:
                    continue

    if vol_col is None:
        return None

    out = df[[date_col, vol_col]].copy()
    out.columns = ["date", "volume"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["date", "volume"])
    if out.empty:
        return None
    return out

def main():
    files = [f for f in os.listdir(RAW_DIR) 
             if f.lower().endswith((".csv",".txt",".xls",".xlsx")) and f != "water.csv"]
    if not files:
        raise RuntimeError(f"No se encontraron archivos de datos en {RAW_DIR}. Descarga primero los datos de Kaggle.")

    dfs = []
    for f in files:
        path = os.path.join(RAW_DIR, f)
        try:
            norm = normalize_one_file(path)
            if norm is None:
                print(f"Advertencia: {f} no tiene columnas detectables de fecha/volumen. Se omite.")
                continue
            zone = os.path.splitext(f)[0]
            norm["zone"] = zone
            dfs.append(norm)
        except Exception as e:
            print(f"Advertencia: no se pudo procesar {f}: {e}")

    if not dfs:
        raise RuntimeError("No se pudo construir dataset: ningún archivo aportó columnas válidas de fecha/volumen.")

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values(["zone", "date"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Dataset combinado guardado en {OUT_CSV} con {len(out)} filas")

if __name__ == "__main__":
    main()
