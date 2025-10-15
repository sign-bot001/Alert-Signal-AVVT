# app.py — Signal Alert AVTR (simple & sans upload)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from io import StringIO

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------ CONFIG & THEME ------------------
st.set_page_config(page_title="Signal Alert AVTR", layout="wide")
TZ = pytz.timezone("Asia/Seoul")

st.markdown("""
<style>
body {background:#050608;}
.stApp {color:#d5f5ff;}
.title {font-family:'Courier New',monospace;color:#7afcff;font-size:28px;}
.small {color:#9fd;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">⚡ Signal Alert AVTR — Prédictions 60 min (KST)</div>', unsafe_allow_html=True)

# ------------------ BASE HISTORY (aucun upload requis) ------------------
DEFAULT_HISTORY = [
    1.3,1.23,1.56,2.25,1.15,13.09,20.91,2.05,10.17,3.82,
    1.00,1.46,1.40,1.73,1.17,1.00,26.60,8.60,1.27,1.46,
    1.36,1.76,3.61,2.74,1.47,3.70,1.05
]

def history_to_df(values):
    """Crée un DataFrame minute par minute, dernier point = maintenant (KST)."""
    now = datetime.now(TZ)
    ts = [now - timedelta(minutes=(len(values)-1-i)) for i in range(len(values))]
    return pd.DataFrame({"timestamp": ts, "multiplier": values})

# Mémoire de session pour éviter de re-coller à chaque fois
if "history" not in st.session_state:
    st.session_state.history = DEFAULT_HISTORY.copy()

# Zone de collage (facultatif)
with st.expander("➕ (Optionnel) Coller un nouvel historique — une valeur par ligne, décimales ',' ou '.'"):
    txt = st.text_area("Colle ici (ex: 1,23 sur chaque ligne)", height=160, key="pastebox")
    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("✅ Mettre à jour l'historique collé"):
            if txt.strip():
                lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
                vals = []
                for ln in lines:
                    try:
                        vals.append(float(ln.replace(",", ".")))
                    except:
                        pass
                if len(vals) >= 15:
                    st.session_state.history = vals
                    st.success(f"Historique mis à jour ({len(vals)} points).")
                else:
                    st.error("Historique trop court. Fournis au moins 15 lignes.")
    with col_b:
        if st.button("↩️ Restaurer l'historique par défaut"):
            st.session_state.history = DEFAULT_HISTORY.copy()
            st.info("Historique par défaut restauré.")

# Construire le DF à partir de l'historique en session
base_df = history_to_df(st.session_state.history)

# ------------------ FEATURES ------------------
def add_time_features(df):
    t = df["timestamp"]
    df["minute"] = t.dt.minute
    df["hour"] = t.dt.hour
    df["dow"] = t.dt.dayofweek
    # encodage cyclique
    df["min_sin"] = np.sin(2*np.pi*df["minute"]/60.0)
    df["min_cos"] = np.cos(2*np.pi*df["minute"]/60.0)
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
    return df

def add_lags_and_rolls(df, lags=30):
    s = df["multiplier"]
    for k in range(1, lags+1):
        df[f"lag_{k}"] = s.shift(k)
    # rolling (décalées pour éviter la fuite d'info)
    df["roll_mean_5"]  = s.rolling(5).mean().shift(1)
    df["roll_std_5"]   = s.rolling(5).std().shift(1).fillna(0)
    df["roll_mean_15"] = s.rolling(15).mean().shift(1)
    df["roll_std_15"]  = s.rolling(15).std().shift(1).fillna(0)
    df["roll_mean_30"] = s.rolling(30).mean().shift(1)
    df["roll_std_30"]  = s.rolling(30).std().shift(1).fillna(0)
    # dynamics
    df["mom_3"] = s / s.shift(3) - 1
    df["pct_1"] = s.pct_change(1).shift(1).fillna(0)
    df["pct_3"] = s.pct_change(3).shift(1).fillna(0)
    df["vol_15"] = df["roll_std_15"] / (df["roll_mean_15"] + 1e-9)
    return df

def build_feature_matrix(df, lags=30):
    df2 = df.copy()
    df2 = add_time_features(df2)
    df2 = add_lags_and_rolls(df2, lags=lags)
    df2 = df2.dropna().reset_index(drop=True)
    y = df2["multiplier"].values
    X = df2.drop(columns=["multiplier","timestamp"]).values
    return X, y, df2[["timestamp"]].copy()

# ------------------ MODEL (stack + quantiles) ------------------
def fit_stack_and_quantiles(X, y, n_estimators=200):
    rf  = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=1)
    et  = ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=2)
    gbr = GradientBoostingRegressor(random_state=3)

    # OOF for meta
    tscv = TimeSeriesSplit(n_splits=5)
    meta_X = np.zeros((len(X), 3))
    for tr, va in tscv.split(X):
        X_tr, X_va = X[tr], X[va]
        y_tr = y[tr]
        rf.fit(X_tr, y_tr); et.fit(X_tr, y_tr); gbr.fit(X_tr, y_tr)
        meta_X[va,0] = rf.predict(X_va)
        meta_X[va,1] = et.predict(X_va)
        meta_X[va,2] = gbr.predict(X_va)

    # fit full
    rf.fit(X, y); et.fit(X, y); gbr.fit(X, y)
    meta = Ridge(alpha=1.0).fit(meta_X, y)

    # quantiles pour l'incertitude
    q10 = GradientBoostingRegressor(loss="quantile", alpha=0.10, random_state=11).fit(X, y)
    q90 = GradientBoostingRegressor(loss="quantile", alpha=0.90, random_state=12).fit(X, y)

    base = {"rf": rf, "et": et, "gbr": gbr}
    quant = {"q10": q10, "q90": q90}
    return base, meta, quant

def stack_predict(base, meta, X):
    P = np.vstack([base["rf"].predict(X), base["et"].predict(X), base["gbr"].predict(X)]).T
    return meta.predict(P)

def confidence_from_band(p, lo, hi):
    width = np.maximum(hi - lo, 1e-9)
    rel = width / np.maximum(np.abs(p)+1e-6, 1.0)  # largeur relative
    conf = 100 * (1 - np.clip(rel, 0, 1))
    return np.clip(conf, 0, 100)

# ------------------ FORECAST 60 MIN ------------------
def iterative_forecast(df, lags, base, meta, quant, steps=60):
    work = df.copy()
    # assure les features existent
    X_all, y_all, _ = build_feature_matrix(work, lags=lags)
    if len(X_all) == 0:
        raise ValueError("Historique trop court pour générer des features.")
    last_ts = work["timestamp"].iloc[-1]

    preds, lows, highs, confs, times = [], [], [], [], []
    for _ in range(steps):
        X_all, _, _ = build_feature_matrix(work, lags=lags)
        x_last = X_all[-1:].copy()
        p  = float(stack_predict(base, meta, x_last)[0])
        lo = float(quant["q10"].predict(x_last)[0])
        hi = float(quant["q90"].predict(x_last)[0])
        c  = float(confidence_from_band(p, lo, hi))
        next_ts = last_ts + timedelta(minutes=1)

        # ajoute la prédiction pour servir de lag à l'étape suivante
        work = pd.concat([work, pd.DataFrame({"timestamp":[next_ts], "multiplier":[p]})], ignore_index=True)
        last_ts = next_ts

        preds.append(p); lows.append(lo); highs.append(hi); confs.append(c); times.append(next_ts)

    out = pd.DataFrame({
        "timestamp_kst": times,
        "predicted_multiplier": np.round(preds, 4),
        "confidence_0_100": np.round(cfs:=confs, 1)
    })
    return out

# ------------------ CONTROLES MINIMAUX ------------------
st.sidebar.header("Paramètres (optionnels)")
lags = st.sidebar.slider("Mémoire (lags)", 10, 60, 30)
n_estimators = st.sidebar.slider("Arbres par modèle", 100, 400, 200)

run = st.button("🚀 Prédire les 60 prochaines minutes (KST)")

# ------------------ RUN ------------------
if run:
    df = base_df[["timestamp","multiplier"]].copy()

    # besoin d'un minimum de données
    if len(df) < lags + 20:
        st.error(f"Historique trop court ({len(df)} points). Minimum recommandé : {lags+20}.")
    else:
        X, y, _ = build_feature_matrix(df, lags=lags)
        base, meta, quant = fit_stack_and_quantiles(X, y, n_estimators=n_estimators)
        out = iterative_forecast(df, lags, base, meta, quant, steps=60)

        # Tableau simple : heure + prédiction + confiance
        st.dataframe(out)

        # Téléchargement CSV
        st.download_button(
            "Télécharger les 60 prédictions (CSV)",
            out.to_csv(index=False),
            file_name="signal_alert_avtr_predictions_60min.csv"
        )

