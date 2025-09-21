#!/usr/bin/env python3
"""
Quick test script to debug column mapping issues
"""

import pandas as pd
import numpy as np
from app.siams_prep import unify_columns, add_lags_and_rolls, add_plant_health_metrics

# Simulate the Google Sheets data format
print("=== Testing Column Mapping ===")

# Create test data that mimics your Google Sheets headers
test_data = {
    "Date": ["2024-01-01 10:00:00", "2024-01-01 11:00:00", "2024-01-01 12:00:00", "2024-01-01 13:00:00", "2024-01-01 14:00:00", "2024-01-01 15:00:00", "2024-01-01 16:00:00"],
    "Soil_Moisture_pct": [45.2, 44.8, 44.1, 43.7, 43.2, 42.9, 42.5],
    "Temperature_degC": [25.3, 26.1, 26.8, 27.2, 26.9, 26.4, 25.8],
    "Humidity_pct": [68.2, 67.5, 66.8, 66.2, 66.7, 67.1, 67.8],
    "Relative_Light_Intensity_ADC_Units": [2847, 3241, 3689, 3521, 3287, 2945, 2634],
    "site_id": ["Unilag", "Unilag", "Unilag", "Unilag", "Unilag", "Unilag", "Unilag"]
}

df = pd.DataFrame(test_data)

print(f"Original data shape: {df.shape}")
print(f"Original columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head(3))

# Test column mapping
print("\n=== Testing unify_columns ===")
df_unified = unify_columns(df)

print(f"\nUnified data shape: {df_unified.shape}")
print(f"Unified columns: {list(df_unified.columns)}")
print("\nFirst few rows after unify:")
print(df_unified.head(3))

# Check for required columns
required = ["timestamp", "soil_moisture_pct", "temperature_c", "humidity_pct", "site_id"]
for col in required:
    if col in df_unified.columns:
        non_null = df_unified[col].notna().sum()
        print(f"[OK] {col}: {non_null} non-null values")
    else:
        print(f"[MISSING] {col}: column not found")

# Test feature engineering
print("\n=== Testing add_lags_and_rolls ===")
df_features = add_lags_and_rolls(df_unified)

print(f"\nFeature data shape: {df_features.shape}")
print(f"Feature columns: {list(df_features.columns)}")

# Check key lag features
lag_features = ["soil_moisture_lag1", "soil_moisture_lag2", "soil_moisture_lag3", "soil_moisture_lag6", "soil_moisture_roll6"]
print("\nLag feature status:")
for feat in lag_features:
    if feat in df_features.columns:
        non_null = df_features[feat].notna().sum()
        print(f"[OK] {feat}: {non_null} non-null values")
    else:
        print(f"[MISSING] {feat}: column not found")

print("\nTest completed!")
