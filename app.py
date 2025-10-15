# app.py — Signal Alert AVTR (Pro / simple)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from io import StringIO

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------ CONFIG & THEME ------------------
st.set_page_config(page_title="Signal Alert AVTR", layout="wide")
tz_choice = "Asia/Seoul"

st.markdown("""
<style>
body {background:#050608;}
.stApp {color:#d5f5ff;}
.title {font-family:'Courier New',monospace;color:#7afcff;font-size:28px;}
.small {color:#9fd;}
.dataframe {font-size:14px;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="title">⚡ Signal Alert AVTR — Prédictions 60 min (KST)</div>', unsafe_allow_html=True)

# ------------------ DATA HELPERS ------------------
SAMPLE_HISTORY = [1.3,1.23,1.56,2.25,1.15,13.09,20.91,2.05,10.17,3.82,
                  1,1.46,1.4,1.73,1.17,1.00,26.60,8.6,1.27,1.46,
                  1.36,1.76,3.61,2.74,1.47,3.7,1.05]

def _looks_like_one_column_numbers(lines):
    if not lines: return False
    ok = True
    for ln in lines:
        t = ln.strip()
        if not t: 
            continue
        if not any(ch.isdigit() for ch in t):
            ok = False; break
        if not all(ch.isdigit() or ch in " .,-+" for ch in t):
            ok = False; break
    return ok

@st.cache_data(show_spinner=False)
def load_sample_df():
    tz = pytz.timezone(tz_choice)
    last_ts = datetime.now(tz)
    ts = [last_ts - timedelta(minutes=(len(SAMPLE_HISTORY)-1-i)) for i in range(len(SAMPLE_HISTORY))]
    return pd.DataFrame({"timestamp": ts, "multiplier": SAMPLE_HISTORY})

@st.cache_data(show_spinner=False)
def load_csv_bytes(uploaded_bytes: bytes):
    text = uploaded_bytes.decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]

    # 1) One-column numeric list (accepts comma decimal)
    if _looks_like_one_column_numbers(lines) and all(line.count(",") <= 1 for line in lines):
        vals = []
        for ln in lines:
            try:
                vals.append(float(ln.replace(",", ".")))
            except: pass
        tz = pytz.timezone(tz_choice)
        last_ts = datetime.now(tz)
        ts = [last_ts - timedelta(minutes=(len(vals)-1-i)) for i in range(len(vals))]
        return pd.DataFrame({"timestamp": ts, "multiplier": vals})

    # 2) CSV with timestamp + multiplier (semicolon tolerated)
    text = text.replace(";", ",")
    df = pd.read_csv(StringIO(text))
    # detect columns
    ts_col, mult_col = None, None
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["time","date","timestamp"]): ts_col = c
        if any(k in cl for k in ["multiplier","mult","cote","value","rate"]): mult_col = c
    if ts_col is None and df.shape[1] >= 2: ts_col = df.columns[0]
    if mult_col is None: mult_col = df.columns[1] if df.shape[1] >= 2 else df.columns[0]

    df = df[[ts_col, mult_col]].copy()
    df.columns = ["timestamp", "multiplier"]
    df["multiplier"] = pd.to_numeric(df["multiplier"].astype(str).str.replace(",", "."), errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna()
    df["timestamp"] = df["timestamp"].dt.tz_convert(pytz.timezone(tz_choice))
    # resample to 1-minute bins (last value in the minute, ffill gaps)
    df = df.set_index("timestamp").resample("1T").last().ffill().reset_index()
    return df

# ------------------ FEATURE ENGINEERING ------------------
def add_time_features(df):
    t = df["timestamp"]
    df["minute"] = t.dt.minute
    df["hour"] = t.dt.hour
    df["dow"] = t.dt.dayofweek
    # cyclic encodings
    df["min_sin"] = np.sin(2*np.pi*df["minute"]/60.0)
    df["min_cos"] = np.cos(2*np.pi*df["minute"]/60.0)
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
    return df

def add_lags_and_rolls(df, lags=30):
    s = df["multiplier"]
    for k in range(1, lags+1):
        df[f"lag_{k}"] = s.shift(k)
    # rolling stats (shifted to avoid leakage)
    df["roll_mean_5"]  = s.rolling(5).mean().shift(1)
    df["roll_std_5"]   = s.rolling(5).std().shift(1).fillna(0)
    df["roll_mean_15"] = s.rolling(15).mean().shift(1)
    df["roll_std_15"]  = s.rolling(15).std().shift(1).fillna(0)
    df["roll_mean_30"] = s.rolling(30).mean().shift(1)
    df["roll_std_30"]  = s.rolling(30).std().shift(1).fillna(0)
    # momentum & pct changes
    df["mom_3"] = s / s.shift(3) - 1
    df["pct_1"] = s.pct_change(1).shift(1).fillna(0)
    df["pct_3"] = s.pct_change(3).shift(1).fillna(0)
    # volatility normalized
    df["vol_15"] = df["roll_std_15"] / (df["roll_mean_15"] + 1e-9)
    return df

def build_feature_matrix(df, lags=30):
    df2 = df.copy()
    df2 = add_time_features(df2)
    df2 = add_lags_and_rolls(df2, lags=lags)
    df2 = df2.dropna().reset_index(drop=True)
    y = df2["multiplier"].values
    X = df2.drop(columns=["multiplier","timestamp"]).values
    cols = [c for c in df2.columns if c not in ["multiplier","timestamp"]]
    return X, y, df2[["timestamp"]].copy(), cols

# ------------------ MODELING (STACK + QUANTILES) ------------------
def fit_stack_and_quantiles(X, y, n_estimators=200):
    # Base learners
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=1)
    et = ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=2)
    gbr = GradientBoostingRegressor(random_state=3)

    # OOF to train meta
    tscv = TimeSeriesSplit(n_splits=5)
    meta_X = np.zeros((len(X), 3))
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr = y[train_idx]
        rf.fit(X_tr, y_tr); et.fit(X_tr, y_tr); gbr.fit(X_tr, y_tr)
        meta_X[val_idx,0] = rf.predict(X_val)
        meta_X[val_idx,1] = et.predict(X_val)
        meta_X[val_idx,2] = gbr.predict(X_val)

    # Fit base models on full
    rf.fit(X, y); et.fit(X, y); gbr.fit(X, y)

    # Meta model
    meta = Ridge(alpha=1.0)
    meta.fit(meta_X, y)

    # Quantile models (for uncertainty)
    q10 = GradientBoostingRegressor(loss="quantile", alpha=0.10, random_state=11)
    q90 = GradientBoostingRegressor(loss="quantile", alpha=0.90, random_state=12)
    q10.fit(X, y); q90.fit(X, y)

    base = {"rf": rf, "et": et, "gbr": gbr}
    quant = {"q10": q10, "q90": q90}
    return base, meta, quant

def stack_predict(base, meta, X):
    preds = np.vstack([base["rf"].predict(X), base["et"].predict(X), base["gbr"].predict(X)]).T
    return meta.predict(preds)

def confidence_from_band(p, lo, hi):
    # narrow band => higher confidence. Map relative width to 0..100
    width = np.maximum(hi - lo, 1e-9)
    rel = width / np.maximum(np.abs(p)+1e-6, 1.0)
    # heuristic mapping
    conf = 100 * (1 - np.clip(rel, 0, 1))
    return np.clip(conf, 0, 100)

# ------------------ ITERATIVE FORECAST ------------------
def iterative_forecast(df, lags, base, meta, quant, steps=60):
    # build last feature row repeatedly
    work = df.copy()
    X_all, y_all, ts_all, cols = build_feature_matrix(work, lags=lags)
    if len(X_all) == 0:
        raise ValueError("Historique trop court pour générer des features.")
    last_ts = work["timestamp"].iloc[-1]
    preds, lows, highs, confs, times = [], [], [], [], []
    for i in range(steps):
        X_all, _, _, _ = build_feature_matrix(work, lags=lags)
        x_last = X_all[-1:].copy()
        p = float(stack_predict(base, meta, x_last)[0])
        lo = float(quant["q10"].predict(x_last)[0])
        hi = float(quant["q90"].predict(x_last)[0])
        c = float(confidence_from_band(p, lo, hi))
        next_ts = last_ts + timedelta(minutes=1)

        # append predicted row (so next step can use it as lag)
        new_row = pd.DataFrame({"timestamp":[next_ts], "multiplier":[p]})
        work = pd.concat([work, new_row], ignore_index=True)
        last_ts = next_ts

        preds.append(p); lows.append(lo); highs.append(hi); confs.append(c); times.append(next_ts)

    out = pd.DataFrame({
        "timestamp_kst": times,
        "predicted_multiplier": np.round(preds, 4),
        "conf_%": np.round(confs, 1),
        "p10": np.round(lows, 4),
        "p90": np.round(highs, 4),
    })
    return out

# ------------------ SIDEBAR (minimal) ------------------
st.sidebar.header("Données")
uploaded = st.sidebar.file_uploader("CSV (timestamp,multiplier) ou une colonne de nombres", type=["csv"])
lags = st.sidebar.slider("Mémoire (lags, minutes)", 10, 60, 30)
n_estimators = st.sidebar.slider("Arbres par modèle", 100, 400, 200)
steps = 60  # exactement 60 minutes à prédire
run = st.button("🚀 Prédire les 60 prochaines minutes (KST)")

# ------------------ DATA LOAD ------------------
if uploaded is not None:
    df = load_csv_bytes(uploaded.getvalue())
    if df is None or df.empty:
        st.warning("CSV non lisible. Utilisation de l’historique intégré.")
        df = load_sample_df()
else:
    df = load_sample_df()

# On n'affiche pas d'autres éléments — app minimaliste
# st.dataframe(df.tail())  # (décommenter si tu veux voir l'historique)

# ------------------ RUN ------------------
if run:
    # préparation features
    base_df = df[["timestamp","multiplier"]].copy()
    # S'assure d'avoir au moins lags+30 points
    if len(base_df) < lags + 30:
        st.error(f"Historique trop court ({len(base_df)} lignes). Minimum recommandé: {lags+30}.")
    else:
        # Standardisation intégrée au modèle GBR, pas nécessaire pour arbres
        # Fit stack + quantiles
        X, y, _, _ = build_feature_matrix(base_df, lags=lags)
        base, meta, quant = fit_stack_and_quantiles(X, y, n_estimators=n_estimators)

        # Forecast 60 minutes
        out = iterative_forecast(base_df, lags, base, meta, quant, steps=steps)

        # UI: seulement le tableau des prédictions (simple)
        st.dataframe(out[["timestamp_kst","predicted_multiplier","conf_%"]])

        # Téléchargement
        st.download_button(
            "Télécharger les 60 prédictions (CSV)",
            out[["timestamp_kst","predicted_multiplier","conf_%"]].to_csv(index=False),
            file_name="signal_alert_avtr_predictions_60min.csv"
        )

