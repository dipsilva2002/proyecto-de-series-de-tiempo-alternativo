import os
from dataclasses import dataclass
from typing import Tuple, Optional
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

@dataclass
class Paths:
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    models_dir: str = "models"
    reports_dir: str = "reports/figures"

def ensure_dirs(p: Paths = Paths()):
    os.makedirs(p.raw_dir, exist_ok=True)
    os.makedirs(p.interim_dir, exist_ok=True)
    os.makedirs(p.processed_dir, exist_ok=True)
    os.makedirs(p.models_dir, exist_ok=True)
    os.makedirs(p.reports_dir, exist_ok=True)

def db_connect():
    url = os.getenv("DATABASE_URL")
    if not url:
        return None
    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return engine
    except Exception:
        return None

def get_env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except Exception:
        return default

def get_env_str(key: str, default: str) -> str:
    return os.getenv(key, default)
