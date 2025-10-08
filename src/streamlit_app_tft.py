import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# PyTorch Forecasting / Lightning
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.data.encoders import NaNLabelEncoder

import datetime as dt
import matplotlib.pyplot as plt

# ============================
# App config
# ============================
st.set_page_config(page_title="TFT – Tetouan Energy Forecast", layout="wide")
st.title("⚡ Temporal Fusion Transformer – Inferencia (Tetouan 10 min)")

st.markdown(
    """
Esta app carga un **TFT entrenado** desde `tft_model_state_dict.pt` y realiza **inferencia de 24 h (144 pasos / 10 min)**.

**Entrada requerida:** un **CSV** con **exactamente los últimos 7 días** (1008 filas) a 10 minutos, con las columnas mínimas:
- `datetime` (ISO `YYYY-MM-DD HH:MM:SS`)
- `zone` (ej. `zone_1`)
- `zone_1` (target de consumo)
- `temperature`, `humidity`, `wind_speed`, `general_diffuse_flows`

> Las variables de calendario se calculan automáticamente.
"""
)

# ============================
# Hiperparámetros / configuración HARD-CODE (según notebook)
# ============================
# Referencia del cuaderno: prediction_length=24*6=144; max_encoder_length=7*24*6=1008
# hidden_size=64, attention_head_size=4, dropout=0.1, hidden_continuous_size=32, QuantileLoss
# time_varying_known_reals: ["time_idx","hour","day","day_of_week","month","is_weekend","is_holiday","temperature","humidity","wind_speed","general_diffuse_flows"]
# time_varying_unknown_reals: ["zone_1"]
# static_categoricals: ["zone"]

ENC_LEN = 7 * 24 * 6         # 1008
DEC_LEN = 24 * 6             # 144
HIDDEN_SIZE = 64
ATTN_HEADS = 4
DROPOUT = 0.1
HIDDEN_CONT_SIZE = 32
TARGET_COL = "zone_1"
STATIC_CATEGORICALS = ["zone"]
KNOWN_REAL_COLS = [
    "time_idx", "hour", "day", "day_of_week", "month", "is_weekend", "is_holiday",
    "temperature", "humidity", "wind_speed", "general_diffuse_flows"
]
UNKNOWN_REAL_COLS = [TARGET_COL]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.caption(f"Dispositivo: **{DEVICE}**")

# ============================
# Uploaders
# ============================
model_file = st.file_uploader("Sube el modelo (state_dict .pt)", type=["pt"]) 
enc_csv = st.file_uploader("Sube el CSV con los **últimos 7 días (1008 filas)** a 10 min", type=["csv"]) 

# ============================
# Utilidades
# ============================
HOLIDAYS = set()  # agrega feriados locales si aplica

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    # Asegurar dtype consistente en toda la app
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False, errors="coerce")
    dt_series = df["datetime"]
    df["hour"] = dt_series.dt.hour.astype(float)
    df["day"] = dt_series.dt.day.astype(float)
    df["day_of_week"] = dt_series.dt.dayofweek.astype(float)
    df["month"] = dt_series.dt.month.astype(float)
    df["is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(float)
    df["is_holiday"] = dt_series.dt.date.astype(str).isin(HOLIDAYS).astype(float)
    return df


def build_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True).copy()
    # time_idx a 10 min desde el inicio del df
    t0 = pd.to_datetime(df["datetime"].iloc[0])
    df["time_idx"] = ((pd.to_datetime(df["datetime"]) - t0) / pd.Timedelta(minutes=10)).astype(int)
    return df

@st.cache_data(show_spinner=False)
def load_state_dict(buffer):
    return torch.load(buffer, map_location="cpu")

# ============================
# Construcción de dataset y modelo
# ============================

def build_full_frame_from_encoder(enc_df: pd.DataFrame) -> pd.DataFrame:
    enc_df = enc_df.copy()
    enc_df = add_calendar_features(enc_df)

    # crear 24h futuras a 10 min (decoder) con persistencia de meteorología
    last_dt = pd.to_datetime(enc_df["datetime"].iloc[-1])
    future_times = pd.date_range(last_dt + pd.Timedelta(minutes=10), periods=DEC_LEN, freq="10min")
    last_row = enc_df.iloc[-1]
    dec_df = pd.DataFrame({
        "datetime": future_times,
        "zone": last_row["zone"],
        TARGET_COL: np.nan,
        "temperature": last_row.get("temperature", np.nan),
        "humidity": last_row.get("humidity", np.nan),
        "wind_speed": last_row.get("wind_speed", np.nan),
        "general_diffuse_flows": last_row.get("general_diffuse_flows", np.nan),
    })
    dec_df = add_calendar_features(dec_df)

    full = pd.concat([enc_df, dec_df], ignore_index=True)
    full = build_time_idx(full)
    return full


def clean_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # quitar inf y -inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def make_dataset(enc_only: pd.DataFrame, full: pd.DataFrame) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    # Limpiar numéricos
    enc_only = clean_numeric(enc_only, [TARGET_COL, "temperature", "humidity", "wind_speed", "general_diffuse_flows"]) 
    full = clean_numeric(full, [TARGET_COL, "temperature", "humidity", "wind_speed", "general_diffuse_flows"]) 

    # Quitar NaN del target en el ENCODER (histórico). IMPORTANTÍSIMO para TimeSeriesDataSet base
    enc_only = enc_only.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # Encoders categóricos
    cat_enc = {"zone": NaNLabelEncoder().fit(enc_only["zone"]) }

    # Dataset base usando SOLO encoder sin NaN en target
    training = TimeSeriesDataSet(
        enc_only,
        time_idx="time_idx",
        target=TARGET_COL,
        group_ids=["zone"],
        min_encoder_length=ENC_LEN // 2,
        max_encoder_length=ENC_LEN,
        min_prediction_length=1,
        max_prediction_length=DEC_LEN,
        static_categoricals=STATIC_CATEGORICALS,
        time_varying_known_reals=KNOWN_REAL_COLS,
        time_varying_unknown_reals=UNKNOWN_REAL_COLS,
        target_normalizer=GroupNormalizer(groups=["zone"]),
        categorical_encoders=cat_enc,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Dataset de predicción a partir del "training" pero con el FULL (que incluye NaN en decoder target)
    predict_ds = TimeSeriesDataSet.from_dataset(
        training,
        full,
        predict=True,
        stop_randomization=True,
    )

    return training, predict_ds


def build_model(training: TimeSeriesDataSet) -> TemporalFusionTransformer:
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTN_HEADS,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONT_SIZE,
        loss=QuantileLoss(),
        optimizer="adam",
    )
    return model


def predict_from_state_dict(state_dict, training: TimeSeriesDataSet, predict_ds: TimeSeriesDataSet):
    model = build_model(training)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        st.warning(f"Parámetros faltantes al cargar state_dict (muestra): {sorted(list(missing))[:8]}{ ' ...' if len(missing)>8 else ''}")
    if unexpected:
        st.warning(f"Parámetros inesperados (muestra): {sorted(list(unexpected))[:8]}{ ' ...' if len(unexpected)>8 else ''}")

    model.to(DEVICE)
    model.eval()

    dl = predict_ds.to_dataloader(train=False, batch_size=1, shuffle=False)
    with torch.no_grad():
        raw, x = model.predict(dl, mode="raw", return_x=True)

    preds = raw if isinstance(raw, torch.Tensor) else raw.get("prediction", raw)
    p10 = preds[..., 0].cpu().numpy().ravel()
    p50 = preds[..., 1].cpu().numpy().ravel()
    p90 = preds[..., 2].cpu().numpy().ravel()
    return p10, p50, p90

# ============================
# UI principal
# ============================
st.subheader("1) Cargar archivos")
col1, col2 = st.columns([2, 1])
with col1:
    if enc_csv is not None:
        enc_df = pd.read_csv(enc_csv, parse_dates=["datetime"], dayfirst=False)
        st.write("**Vista previa (últimos 7 días):**", enc_df.head())
        # Validación rápida de dtype
        if not np.issubdtype(enc_df["datetime"].dtype, np.datetime64):
            enc_df["datetime"] = pd.to_datetime(enc_df["datetime"], errors="coerce")
        if enc_df["datetime"].isna().any():
            st.warning("Hay filas con `datetime` inválido (NaT). Se filtrarán antes de predecir.")
            enc_df = enc_df.dropna(subset=["datetime"]).reset_index(drop=True)
    else:
        enc_df = None

with col2:
    if model_file is not None:
        state_dict = load_state_dict(model_file)
        st.success("Modelo (.pt) cargado")
    else:
        state_dict = None

st.divider()

# Botón predecir
if st.button("🚀 Predecir 24 h"):
    if state_dict is None:
        st.error("Falta el modelo (.pt)")
        st.stop()
    if enc_df is None:
        st.error("Falta el CSV con los últimos 7 días")
        st.stop()

    # Validaciones
    needed_cols = {"datetime", "zone", TARGET_COL, "temperature", "humidity", "wind_speed", "general_diffuse_flows"}
    missing_cols = needed_cols - set(enc_df.columns)
    if missing_cols:
        st.error(f"El CSV no tiene columnas requeridas: {sorted(list(missing_cols))}")
        st.stop()

    # Chequeo de longitud
    if len(enc_df) < ENC_LEN:
        st.warning(f"Se esperaban **{ENC_LEN}** filas (7 días a 10 min). Se recibieron {len(enc_df)}. Se continuará, pero el rendimiento puede variar.")

    try:
        full = build_full_frame_from_encoder(enc_df)
        # Partición encoder-only: las últimas ENC_LEN filas del histórico enc_df (ya con calendar y time_idx en full)
        # Para simplificar, tomamos del FULL las primeras len(enc_df) filas y nos quedamos con la cola de ENC_LEN.
        enc_only = full.iloc[: len(enc_df)].tail(ENC_LEN).reset_index(drop=True)

        training, predict_ds = make_dataset(enc_only, full)
        p10, p50, p90 = predict_from_state_dict(state_dict, training, predict_ds)

        # Armar salida del decoder
        full_sorted = full.sort_values("time_idx").reset_index(drop=True)
        dec_part = full_sorted.tail(DEC_LEN).copy()
        dec_part["p10"] = p10
        dec_part["p50"] = p50
        dec_part["p90"] = p90

        st.subheader("2) Resultado de la predicción (24 h)")
        st.dataframe(dec_part[["datetime", "zone", "p10", "p50", "p90"]], use_container_width=True)

        # Gráfico
        fig = plt.figure(figsize=(10, 4))
        t = pd.to_datetime(dec_part["datetime"])  # x-axis
        plt.plot(t, dec_part["p50"], label="p50")
        plt.fill_between(t, dec_part["p10"], dec_part["p90"], alpha=0.2, label="p10–p90")
        plt.xticks(rotation=30)
        plt.title("Pronóstico 24 h (10 min)")
        plt.legend()
        st.pyplot(fig, clear_figure=True)

        # Descargar CSV
        out_csv = dec_part[["datetime", "zone", "p10", "p50", "p90"]]
        st.download_button(
            label="Descargar predicción CSV",
            data=out_csv.to_csv(index=False).encode("utf-8"),
            file_name="tft_forecast_24h.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.exception(e)
        st.error("Fallo en la inferencia. Verifica columnas, cantidad de filas (1008) y que el modelo corresponda a esta configuración.")
