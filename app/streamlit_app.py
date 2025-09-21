# --- MUST BE FIRST LINE ---
from __future__ import annotations


# --- at top of streamlit_app.py ---
import os
import gspread
from google.oauth2.service_account import Credentials
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent / '.env')  # <-- move here, near the top

# Support SHEET_ID from Streamlit Secrets (for compatibility)
try:
    secrets_available = bool(st.secrets)  # Check if secrets are loaded without error
except:
    secrets_available = False

if secrets_available and "SHEET_ID" in st.secrets:
    SHEET_ID = st.secrets["SHEET_ID"].strip()
else:
    SHEET_ID = os.getenv("SHEET_ID", "").strip()

# Support service account from secrets or file
def has_gcp_secret():
    try:
        return secrets_available and ("gcp_service_account" in st.secrets) and bool(SHEET_ID)
    except Exception:
        return False

if has_gcp_secret():
    SA_JSON = ""  # we won't use the file if secrets are present
else:
    SA_JSON = os.getenv("GOOGLE_SA_JSON", "").strip()

def _gs_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    if has_gcp_secret():
        try:
            creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        except Exception:
            if not SA_JSON or not os.path.exists(SA_JSON):
                raise FileNotFoundError(f"GOOGLE_SA_JSON file path is missing or invalid: '{SA_JSON}'. Please check your .env file or provide a valid service account JSON file path.")
            creds = Credentials.from_service_account_file(SA_JSON, scopes=scopes)
    else:
        if not SA_JSON or not os.path.exists(SA_JSON):
            raise FileNotFoundError(f"GOOGLE_SA_JSON file path is missing or invalid: '{SA_JSON}'. Please check your .env file or provide a valid service account JSON file path.")
        creds = Credentials.from_service_account_file(SA_JSON, scopes=scopes)
    return gspread.authorize(creds)

def set_current_site_in_sheet(site_id: str) -> str:
    has_file = bool(SHEET_ID and SA_JSON)
    has_secret = has_gcp_secret()
    if not (has_file or has_secret):
        return "Missing SHEET_ID and service account credentials (GOOGLE_SA_JSON file or st.secrets['gcp_service_account'])."
    try:
        gc = _gs_client()
        sh = gc.open_by_key(SHEET_ID)
        cfg = sh.worksheet("Config")  # (Sheet named "Config")
        # set current site (A2)
        cfg.update_acell("A2", site_id)
        # ensure site is in registry list (A3..)
        values = cfg.col_values(1)  # whole col A
        existing = {v.strip() for v in values[2:]}  # after header & A2
        if site_id not in existing:
            cfg.append_row([site_id])
        # Done: your Apps Script trigger will auto-stamp new rows
        return f"Current site set to '{site_id}'. New rows will be stamped automatically."
    except Exception as e:
        if "StreamlitSecretNotFoundError" in str(type(e)) or "No secrets found" in str(e):
            return "Streamlit secrets not found. Please provide a valid .env file or secrets.toml for credentials."
        return f"Error updating site in sheet: {e}"


# Cleaned imports
import json
from typing import List

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import joblib
import textwrap
import traceback

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

import requests

# LLMs
try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None

try:
    import openai  # v1 SDK
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# --- Import the same feature-engineering functions you trained with ---
from siams_prep import (
    unify_columns,
    choose_light_column,
    add_calendar_features,
    add_lags_and_rolls,
    add_plant_health_metrics,
    one_hot_encode_sites,
)

# ============================
# Config
# ============================
# load_dotenv() moved to top
SHEETS_CSV_URL: str = os.getenv("SHEETS_CSV_URL", "").strip()
TZ: str = os.getenv("TZ", "Africa/Lagos")
KNOWN_SITES: List[str] = [s.strip() for s in os.getenv("KNOWN_SITES", "").split(",") if s.strip()]

MODEL_PATH = os.getenv("MODEL_PATH", "").strip()
FEATURES_JSON = os.getenv("FEATURES_JSON", "").strip()
DRYNESS_CLF_PATH = os.getenv("DRYNESS_CLF", "").strip()
T1_MODEL_PATH = os.getenv("T1_MODEL", "").strip()
T1_FEATURES_JSON = os.getenv("T1_FEATURES_JSON", "").strip()

DRY_THRESHOLD_DEFAULT = float(os.getenv("DRY_THRESHOLD", "20"))
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "60"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").strip().lower()  # none|openai|hf|gemini

# Require ALL artifacts (they're important for full functionality)
_required = [
    (MODEL_PATH, "MODEL_PATH"),
    (FEATURES_JSON, "FEATURES_JSON"),
    (DRYNESS_CLF_PATH, "DRYNESS_CLF"),
    (T1_MODEL_PATH, "T1_MODEL"),
    (T1_FEATURES_JSON, "T1_FEATURES_JSON"),
]
missing = [name for path, name in _required if not path or not os.path.exists(path)]
if missing:
    missing_info = [f"{name} ({path})" for path, name in _required if not path or not os.path.exists(path)]
    st.error(f"Critical artifacts missing: {', '.join(missing_info)}. Please ensure all model files are present for full functionality.")
    st.stop()

# ============================
# Helpers / Loaders
# ============================
@st.cache_resource
def load_artifacts():
    model = None
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    
    expected_feats = []
    if FEATURES_JSON and os.path.exists(FEATURES_JSON):
        expected_feats = json.load(open(FEATURES_JSON))
    
    dryness_clf = None
    if DRYNESS_CLF_PATH and os.path.exists(DRYNESS_CLF_PATH):
        dryness_clf = joblib.load(DRYNESS_CLF_PATH)
    
    t1_model = None
    if T1_MODEL_PATH and os.path.exists(T1_MODEL_PATH):
        t1_model = joblib.load(T1_MODEL_PATH)
    
    t1_expected_feats = []
    if T1_FEATURES_JSON and os.path.exists(T1_FEATURES_JSON):
        t1_expected_feats = json.load(open(T1_FEATURES_JSON))
    
    # Load dryness metadata (optional)
    dryness_meta_path = os.path.join(os.path.dirname(DRYNESS_CLF_PATH or "."), "dryness_meta.json")
    dryness_meta = {}
    if os.path.exists(dryness_meta_path):
        with open(dryness_meta_path, "r") as f:
            dryness_meta = json.load(f)
    
    return model, expected_feats, dryness_clf, t1_model, t1_expected_feats, dryness_meta


@st.cache_resource
def load_llm():
    if LLM_PROVIDER == "hf":
        if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
            st.info("LLM_PROVIDER=hf but torch/transformers is not installed. Running without AI recommendations.")
            return None, None
        try:
            model_name = "deepseek-ai/DeepSeek-R1"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")
            return tokenizer, model
        except Exception as e:
            st.info(f"HF model could not load ({e}). Running without AI recommendations.")
            return None, None

    if LLM_PROVIDER == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        model_id = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
        if not api_key:
            st.info("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set.")
            return None, None
        if not GEMINI_AVAILABLE:
            st.info("google-generativeai is not installed. pip install google-generativeai.")
            return None, None
        try:
            genai.configure(api_key=api_key)
        except Exception as exc:
            st.info(f"Gemini client init failed ({exc}). Running without AI recommendations.")
            return None, None
        return {"model": model_id}, None

    if LLM_PROVIDER == "openai" and OPENAI_AVAILABLE:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            st.info("LLM_PROVIDER=openai but OPENAI_API_KEY is not set.")
            return None, None
        try:
            client = openai.OpenAI(api_key=api_key)
            return client, None
        except Exception as e:
            st.info(f"OpenAI client init failed ({e}). Running without AI recommendations.")
            return None, None

    return None, None


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def pull_raw(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
    except Exception as exc:
        st.error(f"Failed to load feed: {type(exc).__name__}: {exc}")
        st.text(traceback.format_exc())
        raise

    df = unify_columns(df)  # This should fix the column names

    if 'timestamp' in df.columns:
        # Parse timestamp safely
        ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)  # always UTC-aware
        localized = ts.dt.tz_convert(TZ)
        df['timestamp'] = pd.to_datetime(localized.dt.tz_localize(None), errors='coerce')
    else:
        st.error("No timestamp column found after unify_columns")

    choose_light_column(df)  # ensures light_norm in [0,1]

    return df


def build_features(df: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    feats = df.copy()
    feats = add_calendar_features(feats)
    feats = add_lags_and_rolls(feats, lags=(1, 2, 3, 6), roll_window=6)
    feats = add_plant_health_metrics(feats)
    feats = one_hot_encode_sites(feats, KNOWN_SITES)

    # Ensure all expected columns exist
    for c in expected_cols:
        if c not in feats.columns:
            feats[c] = 0.0 if c.startswith("site_id_") else np.nan

    X = feats[expected_cols].copy()

    site_cols = [c for c in expected_cols if c.startswith("site_id_")]
    if site_cols:
        X[site_cols] = X[site_cols].fillna(0.0)

    return X


def latest_per_site(df: pd.DataFrame) -> pd.DataFrame:
    if "site_id" not in df.columns:
        return df.tail(1)
    return df.sort_values("timestamp").groupby("site_id", as_index=False).tail(1)


def generate_recommendation(latest: pd.DataFrame, llm_tokenizer, llm_model, alerts_str: str) -> str:
    # Load context metadata for enhanced LLM context
    context_meta_path = os.path.join(os.path.dirname(MODEL_PATH), "context_meta.json")
    context_meta = {}
    if os.path.exists(context_meta_path):
        with open(context_meta_path, "r") as f:
            context_meta = json.load(f)

    # safe best_r2 formatting
    r2_val = context_meta.get("model_info", {}).get("best_r2", None)
    r2_text = f"{float(r2_val):.3f}" if isinstance(r2_val, (int, float)) else "N/A"

    vpd_series = latest["vpd_kpa"] if "vpd_kpa" in latest.columns else pd.Series(dtype=float)
    disease_series = latest["disease_risk"] if "disease_risk" in latest.columns else pd.Series(dtype=float)

    sites = "Unknown"
    if "site_id" in latest.columns:
        unique_sites = latest["site_id"].dropna().unique()
        if len(unique_sites):
            sites = ", ".join(sorted(str(s) for s in unique_sites))

    def _mean_value(col: str):
        if col not in latest.columns:
            return None
        series = pd.to_numeric(latest[col], errors="coerce").dropna()
        if series.empty:
            return None
        return float(series.mean())

    def _fmt(value, suffix=""):
        if value is None or pd.isna(value):
            return "N/A"
        return f"{value:.1f}{suffix}"

    context = textwrap.dedent(
        f"""
        Site summary: {sites}
        Alerts: {alerts_str or 'None'}
        Avg predicted moisture: {_fmt(_mean_value('pred_moisture'), '%')}
        Avg measured moisture: {_fmt(_mean_value('soil_moisture_pct'), '%')}
        Avg dry-risk probability: {_fmt(_mean_value('dry_prob'), '%')}
        Average VPD: {_fmt(_mean_value('vpd_kpa'), ' kPa')}
        Average temperature: {_fmt(_mean_value('temperature_c'), ' °C')}
        Average humidity: {_fmt(_mean_value('humidity_pct'), '%')}
        """
    ).strip()



    if LLM_PROVIDER == "none":
        return "LLM recommendations disabled. Set LLM_PROVIDER to 'hf' or 'openai' or 'gemini'."

    if LLM_PROVIDER == "hf" and llm_tokenizer and llm_model:
        prompt = (
            "You are an agricultural expert. Based on this smart-farm context, provide concise, "
            "actionable irrigation and plant-care recommendations. Focus on soil moisture, disease prevention, "
            "and timing.\n\n" + context + "\nRecommendations:"
        )
        try:
            inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
            with torch.no_grad():
                outputs = llm_model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
            text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text.replace(prompt, "").strip()
        except Exception as e:
            return f"LLM (HF) error: {e}.\n\nFallback: irrigate if predicted moisture < threshold and VPD is high."

    if LLM_PROVIDER == "gemini" and isinstance(llm_tokenizer, dict):
        prompt = (
            "You are an agricultural expert. Provide concise, actionable irrigation and plant-care recommendations "
            "from this context:\n\n" + context
        )
        try:
            model = genai.GenerativeModel(llm_tokenizer["model"])
            response = model.generate_content(prompt, safety_settings=None)
            return (response.text or "").strip()
        except Exception as e:
            return f"Gemini error: {e}.\n\nFallback: monitor moisture and irrigate early when VPD is high."

    if LLM_PROVIDER == "openai" and OPENAI_AVAILABLE:
        try:
            prompt = (
                "You are an agricultural expert. Provide concise, actionable irrigation and plant-care recommendations "
                "from this context:\n\n" + context
            )
            resp = llm_tokenizer.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"OpenAI error: {e}.\n\nFallback: monitor moisture and irrigate at dawn when heat risk is high."

    return "LLM provider not available.\n\nFallback: monitor moisture closely and irrigate early when VPD is high."


# ============================
# UI LAYOUT
# ============================
st.set_page_config(page_title="SIAMS", page_icon="🌱", layout="wide")

st.title("🌱SIAMS - Smart Integrated Agricultural Monitoring System")
colA, colB, colC = st.columns([1, 1, 1])

# Load models/artifacts once
model, expected_feats, dryness_clf, t1_model, t1_expected, dryness_meta = load_artifacts()

# Use threshold from dryness_meta if available, else env
dry_threshold_default = dryness_meta.get("threshold", float(os.getenv("DRY_THRESHOLD", "20")))

# Load LLM
llm_tokenizer, llm_model = load_llm()

with st.sidebar:
    st.header("Site Manager")
    site_input = st.text_input("Current site_id", placeholder="e.g., Ondo")
    if st.button("Set current site"):
        if not site_input.strip():
            st.warning("Please enter a site_id.")
        else:
            msg = set_current_site_in_sheet(site_input.strip())
            st.success(msg) if msg.startswith("Current site set") else st.error(msg)

        # Optional: refresh your feed so UI reflects changes faster
        pull_raw.clear()
        st.rerun()

    st.header("Settings")
    if not SHEETS_CSV_URL:
        st.error("SHEETS_CSV_URL missing in .env")
    else:
        st.caption("Connected to Google Sheets feed")

    dry_threshold = st.slider(
        "Dryness threshold (%)",
        min_value=5,
        max_value=50,
        value=int(dry_threshold_default),
        help="Below this predicted moisture, we flag 'Dry'",
    )

    if st.button("Refresh now"):
        pull_raw.clear()
        # Clear all caches to ensure column mapping changes take effect
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# Pull raw data
try:
    raw = pull_raw(SHEETS_CSV_URL)
except Exception as e:
    st.error(f"Failed to load feed: {e}")
    st.stop()

# Infer KNOWN_SITES if not set
if not KNOWN_SITES:
    KNOWN_SITES[:] = list(raw['site_id'].dropna().unique())

# Show quick peek
# Build features and predict for all valid rows
if not expected_feats or not model:
    st.error("Main model or features not available. Cannot make predictions.")
    st.stop()

X = build_features(raw, expected_feats)

# Debug: Show what's happening with column mapping
# Build features and predict for all valid rows
mask = ~X.isna().any(axis=1)   # rows with no NaNs (one-hots are already 0)
idx = X.index[mask]
X_valid = X[mask]

if len(X_valid) == 0:
    st.warning("No rows with complete features yet. Need at least 7 readings per site for lag features (lags 1,2,3,6). Add more data.")
    st.stop()

pred = np.clip(model.predict(X_valid), 0, 100)

# Dryness proba
if dryness_clf:
    try:
        dry_proba = dryness_clf.predict_proba(X_valid)[:, 1]
    except Exception:
        dry_proba = np.full(len(X_valid), np.nan)
else:
    dry_proba = np.full(len(X_valid), np.nan)

# Merge with raw for display
result = raw.loc[idx, ["timestamp", "site_id", "soil_moisture_pct", "temperature_c", "humidity_pct"]].copy()
result["pred_moisture"] = pred.round(1)
result["dry_prob"] = np.round(dry_proba * 100, 1)
result["is_dry"] = (result["pred_moisture"] < dry_threshold).astype(int)

# Keep health metrics for insights
health_cols = [c for c in [
    "vpd_kpa", "dew_point_c", "heat_flag", "frost_flag",
    "waterlog_flag", "dryspell_flag", "disease_risk", "sensor_issue_flag"
] if c in X.columns]
health = X.loc[idx, health_cols]
result = pd.concat([result.reset_index(drop=True), health.reset_index(drop=True)], axis=1)

# Prepare view for downstream displays
if result.empty:
    st.warning('No rows match the current data yet.')
    st.stop()
view = result.copy()

# --- KPI Cards (latest rows) ---
latest = latest_per_site(view)

with colA:
    st.metric('Latest predicted moisture (%)', f"{latest['pred_moisture'].mean():.1f}")
with colB:
    st.metric('Dry risk (%)', f"{np.nanmean(latest['dry_prob']):.1f}")
with colC:
    vpd_series = latest.get('vpd_kpa', pd.Series([np.nan]))
    vpd_val = np.nanmean(vpd_series)
    st.metric('VPD (kPa)', f"{vpd_val:.2f}" if not np.isnan(vpd_val) else '--')

# --- Alerts ---
alerts = []
if (latest['is_dry'] == 1).any():
    alerts.append('Low soil moisture: irrigation recommended.')
if 'waterlog_flag' in latest.columns and (latest['waterlog_flag'] == 1).any():
    alerts.append('Possible waterlogging: avoid over-irrigation.')
if 'heat_flag' in latest.columns and (latest['heat_flag'] == 1).any():
    alerts.append('Heat stress risk: favor early morning irrigation.')
if 'frost_flag' in latest.columns and (latest['frost_flag'] == 1).any():
    alerts.append('Frost risk: avoid late evening irrigation.')
if 'sensor_issue_flag' in latest.columns and (latest['sensor_issue_flag'] == 1).any():
    alerts.append('Sensor stability issue: check probes or wiring.')
if 'disease_risk' in latest.columns and (latest['disease_risk'] > 75).any():
    alerts.append('High disease risk: improve airflow and monitor closely.')

if alerts:
    st.warning('\n'.join(f'- {a}' for a in alerts))
else:
    st.success('All clear: no immediate risks detected.')

# --- Recommendations ---
st.subheader('Recommendations')
recommendation = generate_recommendation(latest, llm_tokenizer, llm_model, '; '.join(alerts) if alerts else 'None')
st.info(recommendation)

# --- Charts ---
st.subheader('Trends')
plot_df = view.sort_values('timestamp').copy()
if 'timestamp' in plot_df.columns:
    plot_df['timestamp'] = pd.to_datetime(plot_df['timestamp'], errors='coerce')
    if getattr(plot_df['timestamp'].dt, 'tz', None) is not None:
        plot_df['timestamp'] = plot_df['timestamp'].dt.tz_localize(None)
if len(plot_df) > 6000:
    plot_df = plot_df.iloc[-6000:]

c1, c2 = st.columns(2)
with c1:
    st.line_chart(plot_df.set_index('timestamp')['pred_moisture'], use_container_width=True)
    st.caption('Predicted soil moisture')
with c2:
    if 'soil_moisture_pct' in plot_df.columns:
        st.line_chart(plot_df.set_index('timestamp')['soil_moisture_pct'], use_container_width=True)
        st.caption('Measured soil moisture')

c3, c4 = st.columns(2)
with c3:
    if 'temperature_c' in plot_df.columns:
        st.line_chart(plot_df.set_index('timestamp')['temperature_c'], use_container_width=True)
        st.caption('Temperature')
with c4:
    if 'humidity_pct' in plot_df.columns:
        st.line_chart(plot_df.set_index('timestamp')['humidity_pct'], use_container_width=True)
        st.caption('Humidity')

with st.expander('Advanced analytics (forecasts & raw data)', expanded=False):
    st.markdown('Detailed tables for support and agronomy teams.')

    st.subheader('One-step ahead forecast (t+1)')
    if not t1_expected or not t1_model:
        st.info('t+1 model or features not available. Forecast disabled.')
    else:
        feats_t1 = build_features(raw, t1_expected)
        mask_t1 = feats_t1.notna().all(axis=1)
        X_t1 = feats_t1[mask_t1]
        idx_t1 = feats_t1.index[mask_t1]

        if len(X_t1):
            yhat_t1 = np.clip(t1_model.predict(X_t1), 0, 100)
            fut = raw.loc[idx_t1, ['timestamp', 'site_id']].copy()
            fut['forecast_t1'] = yhat_t1.round(1)
            if 'timestamp' in fut.columns:
                fut['timestamp'] = pd.to_datetime(fut['timestamp'], errors='coerce')
                if getattr(fut['timestamp'].dt, 'tz', None) is not None:
                    fut['timestamp'] = fut['timestamp'].dt.tz_localize(None)
                if is_datetime64_any_dtype(fut['timestamp']):
                    fut['timestamp'] = fut['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            fut_latest = latest_per_site(fut)
            st.dataframe(fut_latest.sort_values('site_id'), use_container_width=True)
        else:
            st.info('Not enough rows for t+1 forecast yet. Need at least 7 readings per site for lag features.')

    st.subheader('Predictions table')
    pred_table = view.sort_values('timestamp').tail(500).copy()
    if 'timestamp' in pred_table.columns:
        pred_table['timestamp'] = pd.to_datetime(pred_table['timestamp'], errors='coerce')
        if getattr(pred_table['timestamp'].dt, 'tz', None) is not None:
            pred_table['timestamp'] = pred_table['timestamp'].dt.tz_localize(None)
        if is_datetime64_any_dtype(pred_table['timestamp']):
            pred_table['timestamp'] = pred_table['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    st.dataframe(pred_table, use_container_width=True)

    csv = view.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='Download CSV',
        data=csv,
        file_name='siams_predictions.csv',
        mime='text/csv',
    )

    st.caption(
        'Models: regression for moisture, classifier for dry risk, gradient boosting t+1 forecaster. Feature pipeline mirrors training: calendar, lags, rolling mean, light_norm, plant-health metrics, and fixed site one-hot columns.'
    )

