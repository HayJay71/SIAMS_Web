"""
Feature engineering utilities for SIAMS models.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = ["timestamp", "soil_moisture_pct", "temperature_c", "humidity_pct", "site_id"]
_NUMERIC_COLUMNS = ["soil_moisture_pct", "temperature_c", "humidity_pct", "light_adc"]

_NORMALIZED_NAME_MAP = {
    "date": "timestamp",
    "date_": "timestamp",
    "timestamp": "timestamp",
    "soil_moisture": "soil_moisture_pct",
    "soil_moisture_pct": "soil_moisture_pct",
    "soil_moisture_percent": "soil_moisture_pct",
    "soil_moisture__pct": "soil_moisture_pct",
    "soil_moisture_pct_": "soil_moisture_pct",
    "temperature": "temperature_c",
    "temperature_c": "temperature_c",
    "temperature_deg_c": "temperature_c",
    "temperature_degc": "temperature_c",
    "temperature__c": "temperature_c",
    "temperature_aoc": "temperature_c",
    "temperature_a_c": "temperature_c",
    "humidity": "humidity_pct",
    "humidity_percent": "humidity_pct",
    "humidity_pct": "humidity_pct",
    "humidity__pct": "humidity_pct",
    "relative_light_intensity": "light_adc",
    "relative_light_intensity_adc_units": "light_adc",
    "light_intensity": "light_adc",
    "light_intensity_adc_units": "light_adc",
    "light_adc": "light_adc",
    "site": "site_id",
    "site_id": "site_id",
}


def _normalize_header(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def unify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names across heterogeneous sheets."""
    df = df.copy()

    rename_map = {}
    for original in df.columns:
        key = _normalize_header(original)
        canonical = _NORMALIZED_NAME_MAP.get(key)
        if canonical:
            rename_map[original] = canonical
    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()]

    for column in _REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    for column in _NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "site_id" in df.columns:
        df["site_id"] = df["site_id"].where(df["site_id"].notna(), "").astype(str).str.strip()

    df = df.drop(columns=["site"], errors="ignore")
    return df


def choose_light_column(df: pd.DataFrame) -> None:
    """Create a light_norm column on a 0-1 scale."""
    if "light_adc" in df.columns:
        adc = df["light_adc"].astype(float)
        rng = adc.max() - adc.min()
        if pd.notna(rng) and rng > 0:
            df["light_norm"] = (adc - adc.min()) / rng
        else:
            df["light_norm"] = 0.5
        return

    light_cols = [col for col in df.columns if "light" in col.lower()]
    if not light_cols:
        df["light_norm"] = 0.5
        return

    light_values = pd.to_numeric(df[light_cols[0]], errors="coerce")
    df["light_norm"] = np.clip(light_values / 100000, 0, 1)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features aligned with the trained models."""
    df = df.copy()
    if "timestamp" not in df.columns:
        return df

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)

    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear
    df["is_weekend"] = ts.dt.weekday >= 5
    return df


def add_lags_and_rolls(df: pd.DataFrame, lags: tuple = (1, 2, 3, 6), roll_window: int = 6) -> pd.DataFrame:
    """Add lagged and rolling statistics per site."""
    df = df.copy()
    df = df.sort_values(["site_id", "timestamp"])
    grouped = df.groupby("site_id", group_keys=False)

    if "soil_moisture_pct" in df.columns:
        for lag in lags:
            df[f"soil_moisture_lag{lag}"] = grouped["soil_moisture_pct"].shift(lag)
        df[f"soil_moisture_roll{roll_window}"] = (
            grouped["soil_moisture_pct"]
            .rolling(roll_window, min_periods=roll_window)
            .mean()
            .reset_index(level=0, drop=True)
        )

    for column in ["temperature_c", "humidity_pct", "light_norm"]:
        if column in df.columns:
            for lag in lags:
                df[f"{column}_lag{lag}"] = grouped[column].shift(lag)
            df[f"{column}_roll_mean_{roll_window}"] = (
                grouped[column]
                .rolling(roll_window)
                .mean()
                .reset_index(level=0, drop=True)
            )
            df[f"{column}_roll_std_{roll_window}"] = (
                grouped[column]
                .rolling(roll_window)
                .std()
                .reset_index(level=0, drop=True)
            )

    return df


def add_plant_health_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate plant health indicators such as VPD and dew point."""
    df = df.copy()

    if {"temperature_c", "humidity_pct"}.issubset(df.columns):
        temp_c = df["temperature_c"]
        humidity_pct = df["humidity_pct"]
        svp = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
        avp = svp * (humidity_pct / 100)
        df["vpd_kpa"] = svp - avp
        df["dew_point_c"] = temp_c - ((100 - humidity_pct) / 5)

    if "temperature_c" in df.columns:
        df["heat_flag"] = (df["temperature_c"] > 35).astype(int)
        df["frost_flag"] = (df["temperature_c"] < 5).astype(int)

    if "soil_moisture_pct" in df.columns:
        df["waterlog_flag"] = (df["soil_moisture_pct"] > 80).astype(int)
        df["dryspell_flag"] = (df["soil_moisture_pct"] < 20).astype(int)

    if {"temperature_c", "humidity_pct"}.issubset(df.columns):
        temp_band = ((df["temperature_c"] > 25) & (df["temperature_c"] < 30)).astype(int)
        humidity_band = (df["humidity_pct"] > 70).astype(int)
        df["disease_risk"] = (temp_band & humidity_band) * 100

    df["sensor_issue_flag"] = df.get("sensor_issue_flag", 0)
    return df


def one_hot_encode_sites(df: pd.DataFrame, known_sites: List[str]) -> pd.DataFrame:
    """Create deterministic one-hot columns for each known site."""
    df = df.copy()
    if "site_id" not in df.columns:
        return df

    site_series = df["site_id"].astype(str).str.strip()
    for site in known_sites:
        label = site.strip()
        if not label:
            continue
        column_name = f"site_id_{label}"
        df[column_name] = (site_series.str.lower() == label.lower()).astype(int)
    return df