from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from siams_prep import unify_columns, choose_light_column

root = Path(__file__).resolve().parent
load_dotenv(root / '.env')
SHEETS_CSV_URL = os.getenv('SHEETS_CSV_URL')
TZ = os.getenv('TZ', 'UTC')
print('URL:', SHEETS_CSV_URL)

raw = pd.read_csv(SHEETS_CSV_URL)
print('Raw shape:', raw.shape)
print('Raw columns:', list(raw.columns))
print(raw.head())

clean = unify_columns(raw)
print('After unify shape:', clean.shape)
print('After unify columns:', list(clean.columns))
print(clean.head())

if 'timestamp' in clean.columns:
    ts = pd.to_datetime(clean['timestamp'], errors='coerce', utc=True)
    clean['timestamp'] = ts.dt.tz_convert(TZ)

choose_light_column(clean)
print('Light column head:', clean.get('light_norm', [])[:5])
print('NaNs per column:', clean.isna().sum().sort_values(ascending=False).head())

