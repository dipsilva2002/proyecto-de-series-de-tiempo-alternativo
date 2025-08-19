from __future__ import annotations
import os
import argparse
import zipfile
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

def download_competition(competition: str, dest: str):
    api = KaggleApi()
    api.authenticate()
    os.makedirs(dest, exist_ok=True)
    print(f"Descargando '{competition}' en {dest} ...")
    api.competition_download_files(competition, path=dest, quiet=False)
    zips = [p for p in os.listdir(dest) if p.endswith('.zip')]
    for z in zips:
        zpath = os.path.join(dest, z)
        print(f"Extrayendo {zpath} ...")
        with zipfile.ZipFile(zpath, 'r') as zf:
            zf.extractall(dest)

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--competition', default=os.getenv('COMPETITION', 'acea-water-prediction'))
    parser.add_argument('--dest', default=os.path.join('data', 'raw'))
    args = parser.parse_args()
    try:
        download_competition(args.competition, args.dest)
        print("Descarga completada.")
    except Exception as e:
        print("\nError al descargar. Verifica Kaggle instalado, credenciales configuradas y reglas aceptadas en la web.")
        print(f"Detalle: {e}")

if __name__ == '__main__':
    main()
