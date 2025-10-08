# app.py
import io
import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from io import BytesIO
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


# =========================
# Constantes (GitHub RAW)
# =========================
RAW_DATA_URL = "https://raw.githubusercontent.com/DCajiao/Time-series-forecast-of-energy-consumption-in-Tetouan-City/main/data/enriched_zone1_power_consumption_of_tetouan_city.csv"
RAW_STATE_DICT_URL = "https://raw.githubusercontent.com/DCajiao/Time-series-forecast-of-energy-consumption-in-Tetouan-City/main/models/tft_model_state_dict.pt"

# Hiperparámetros modelo (coinciden con el checkpoint)
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
ATTN_HEADS = 4
DROPOUT = 0.1
HIDDEN_CONT = 32
QUANTILES = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]  # 7 cuantiles (como el checkpoint)

PRED_LEN_DEFAULT = 24 * 6          # 144 pasos (24h a 10min)
ENCODER_LEN = 7 * 24 * 6           # ~7 días encoder
FREQ = "10min"                     # resolución base


# =========================
# Helpers
# =========================
@st.cache_data(show_spinner=False)
def load_hist_df(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    t0 = df["datetime"].min()
    df["time_idx"] = ((df["datetime"] - t0) / pd.Timedelta(minutes=10)).astype(int)

    # calendario
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["day_of_week"] >= 5
    df["is_holiday"] = False

    # grupo
    df["zone"] = "zone_1"
    df = df.sort_values(["zone", "time_idx"]).reset_index(drop=True)

    # sanity
    expected_cols = {"temperature", "humidity", "general_diffuse_flows", "zone_1"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {missing}")
    if df["zone_1"].isna().any():
        raise ValueError("NaN en target 'zone_1'. Limpia o imputa antes.")

    df["is_future"] = 0
    return df


def add_calendar_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["day_of_week"] >= 5
    df["is_holiday"] = False
    return df


def make_future_df(
    hist_df: pd.DataFrame,
    start_dt: pd.Timestamp,
    horizon: int,
    exo_source: str,
    const_values: dict | None = None,
    exo_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Construye el DF futuro (decoder) con covariables conocidas para el horizonte.
    exo_source: 'const' o 'table'
    const_values: dict con valores por variable (se replican en todo el horizonte)
    exo_table: DataFrame con columnas ['temperature','humidity','wind_speed','general_diffuse_flows'] y len==horizon
    """
    hist_df = hist_df.copy()
    hist_df["is_future"] = False
    t0 = hist_df["datetime"].min()

    # rango temporal para el futuro
    future_dt = pd.date_range(start=start_dt, periods=horizon, freq=FREQ, inclusive="left")
    fut = pd.DataFrame({"datetime": future_dt})
    fut["time_idx"] = ((fut["datetime"] - t0) / pd.Timedelta(minutes=10)).astype(int)
    fut["zone"] = "zone_1"

    # calendario
    fut = add_calendar_feats(fut)

    # exógenas
    exo_cols = ["temperature", "humidity", "wind_speed", "general_diffuse_flows"]
    if exo_source == "const":
        const_values = const_values or {}
        for c in exo_cols:
            val = const_values.get(c, 0.0)
            fut[c] = float(val)
    else:
        # tabla subida por usuario
        if exo_table is None:
            raise ValueError("Debes subir el CSV de exógenas o elegir 'Constantes'.")
        # normalizamos columnas esperadas
        table = exo_table.copy()
        table.columns = [c.strip().lower() for c in table.columns]
        col_map = {
            "temperature": "temperature",
            "temp": "temperature",
            "humidity": "humidity",
            "hum": "humidity",
            "wind_speed": "wind_speed",
            "wind": "wind_speed",
            "general_diffuse_flows": "general_diffuse_flows",
            "gdf": "general_diffuse_flows",
            "radiation": "general_diffuse_flows",
        }
        # crear frame vacío con columnas esperadas
        tbl = pd.DataFrame(index=range(len(fut)), columns=exo_cols, dtype=float)
        for src_col, dst_col in col_map.items():
            if src_col in table.columns:
                tbl[dst_col] = pd.to_numeric(table[src_col], errors="coerce")

        if len(tbl) != len(fut):
            raise ValueError(f"El CSV debe tener {len(fut)} filas (una por paso del horizonte).")
        if tbl[exo_cols].isna().any().any():
            raise ValueError("Hay NaNs en las exógenas del CSV. Revisa el archivo.")
        fut[exo_cols] = tbl[exo_cols].values

    # target futuro: placeholder numérico (NO NaN) para evitar error del TimeSeriesDataSet
    fut["zone_1"] = 0.0
    fut["is_future"] = True

    # concatenamos histórico + futuro
    full = pd.concat([hist_df, fut], ignore_index=True)

    # asegura tipo numérico para known real
    full["is_future"] = full["is_future"].astype(int)

    return full


# =========================
# Data / Model Managers
# =========================
class DataManager:
    def __init__(self, prediction_length=PRED_LEN_DEFAULT, max_encoder_length=ENCODER_LEN, weather_as_known=True):
        self.prediction_length = prediction_length
        self.max_encoder_length = max_encoder_length
        self.weather_as_known = weather_as_known

        self.training: Optional[TimeSeriesDataSet] = None

        self.time_varying_known_reals: List[str] = []
        self.time_varying_unknown_reals: List[str] = []

    def make_training_from_hist(self, df: pd.DataFrame) -> TimeSeriesDataSet:
        # Asegurar columna is_future presente y numérica (0/1),
        # pero NO la pasamos como feature al modelo para calzar con el checkpoint.
        if "is_future" not in df.columns:
            df = df.copy()
            df["is_future"] = 0
        else:
            df["is_future"] = df["is_future"].astype(int)

        # === IMPORTANTEEEE: NO incluir is_future como known real ===
        known_reals_base = ["time_idx", "hour", "day", "day_of_week", "month", "is_weekend", "is_holiday"]
        weather = ["temperature", "humidity"] + (["wind_speed"] if "wind_speed" in df.columns else []) + ["general_diffuse_flows"]

        if self.weather_as_known:
            self.time_varying_known_reals = known_reals_base + weather
            self.time_varying_unknown_reals = ["zone_1"]
        else:
            self.time_varying_known_reals = known_reals_base
            self.time_varying_unknown_reals = ["zone_1"] + weather

        training = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="zone_1",
            group_ids=["zone"],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.prediction_length,
            static_categoricals=["zone"],
            time_varying_known_reals=self.time_varying_known_reals,
            time_varying_unknown_reals=self.time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=["zone"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        self.training = training
        return training

    def make_predict_from_training(self, training: TimeSeriesDataSet, df_with_future: pd.DataFrame) -> DataLoader:
        # creamos dataset de predicción reusando normalizadores
        predict_ds = TimeSeriesDataSet.from_dataset(training, df_with_future, predict=True, stop_randomization=True)
        predict_dl = predict_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
        return predict_dl


class ModelManager:
    def __init__(self, training_dataset: TimeSeriesDataSet, quantiles: Optional[List[float]] = None):
        self.training_dataset = training_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quantiles = quantiles or QUANTILES
        self.tft: Optional[TemporalFusionTransformer] = None

    def build_model(_self):
        tft = TemporalFusionTransformer.from_dataset(
            _self.training_dataset,
            learning_rate=LEARNING_RATE,
            hidden_size=HIDDEN_SIZE,
            attention_head_size=ATTN_HEADS,
            dropout=DROPOUT,
            hidden_continuous_size=HIDDEN_CONT,
            loss=QuantileLoss(quantiles=_self.quantiles),
        )
        tft.to(_self.device)
        tft.eval()
        _self.tft = tft
        return tft

    def load_state_dict_from_url(self, url: str = RAW_STATE_DICT_URL):
        if self.tft is None:
            raise RuntimeError("Primero ejecuta build_model().")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        buffer = BytesIO(resp.content)
        state = torch.load(buffer, map_location=self.device)
        self.tft.load_state_dict(state)
        self.tft.to(self.device)
        self.tft.eval()

    @torch.no_grad()
    def predict_raw(self, dataloader: DataLoader):
        if self.tft is None:
            raise RuntimeError("Primero build_model() y load_state_dict_from_url().")
        return self.tft.predict(dataloader, mode="raw", return_x=True)


# =========================
# Plotting
# =========================
def plot_prediction_from_raw(raw, idx: int = 0, title="Predicción (p50) con banda p10–p90"):
    x = raw.x
    encoder_target = x["encoder_target"][idx].cpu().numpy().flatten()
    decoder_target = x["decoder_target"][idx].cpu().numpy().flatten()
    decoder_time = x["decoder_time_idx"][idx].cpu().numpy().flatten()
    encoder_time = np.arange(decoder_time[0] - len(encoder_target), decoder_time[0])

    preds = raw.output[0][idx].detach().cpu().numpy()  # (L, n_quant)
    q_list = QUANTILES
    q_idx = {q: q_list.index(q) for q in [0.1, 0.5, 0.9] if q in q_list}

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(encoder_time, encoder_target, label="Histórico (encoder)", linewidth=1.5)

    # Evita dibujar "real futuro" si es todo 0.0 (placeholder)
    if len(decoder_target) and not (np.isnan(decoder_target).all() or np.allclose(decoder_target, 0.0)):
        ax.plot(decoder_time, decoder_target, label="Real futuro (si disponible)", linewidth=1.5)

    if 0.5 in q_idx:
        ax.plot(decoder_time, preds[:, q_idx[0.5]], label="Pred p50", linestyle="--")
    if 0.1 in q_idx and 0.9 in q_idx:
        ax.fill_between(decoder_time, preds[:, q_idx[0.1]], preds[:, q_idx[0.9]], alpha=0.3, label="Banda p10–p90")

    ax.set_title(title)
    ax.set_xlabel("time_idx (10-min)")
    ax.set_ylabel("Consumo (kW)")
    ax.legend()
    st.pyplot(fig)


# =========================
# UI
# =========================
st.set_page_config(page_title="TFT — Pronóstico con exógenas", layout="wide")
st.title("🔌 TFT — Pronóstico de consumo (Zone_1) con entradas exógenas")
st.caption("Carga modelo/datos desde GitHub, ingresa exógenas y genera pronóstico a 10 minutos.")

with st.sidebar:
    st.header("Parámetros de pronóstico")
    horizon = st.number_input("Horizonte (pasos de 10 min)", min_value=6, max_value=7*24*6, value=PRED_LEN_DEFAULT, step=6)
    exo_mode = st.radio("Modo de exógenas", options=["Constantes", "CSV (subir)"], index=0)

    # placeholders de inputs
    const_cols = {}
    uploaded_file = None

# 1) Cargar histórico + preparar training
with st.spinner("Cargando histórico y preparando normalizadores…"):
    hist_df = load_hist_df(RAW_DATA_URL)
    last_dt = hist_df["datetime"].max()
    default_start = (last_dt + pd.Timedelta(minutes=10)).floor(FREQ)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha de inicio", value=default_start.date())
    with col2:
        start_time = st.time_input("Hora de inicio", value=default_start.time())

    start_dt = datetime.datetime.combine(start_date, start_time)

    dm = DataManager(prediction_length=int(horizon), max_encoder_length=ENCODER_LEN, weather_as_known=True)
    training = dm.make_training_from_hist(hist_df)

# 2) Entradas exógenas
with st.sidebar:
    if exo_mode == "Constantes":
        st.subheader("Valores constantes (para todo el horizonte)")
        const_cols["temperature"] = st.number_input("temperature (°C)", value=float(hist_df["temperature"].tail(144).median()))
        const_cols["humidity"] = st.number_input("humidity (%)", value=float(hist_df["humidity"].tail(144).median()))
        wind_guess = float(hist_df["wind_speed"].tail(144).median()) if "wind_speed" in hist_df.columns else 0.0
        const_cols["wind_speed"] = st.number_input("wind_speed (m/s)", value=wind_guess)
        const_cols["general_diffuse_flows"] = st.number_input("general_diffuse_flows (W/m²)", value=float(hist_df["general_diffuse_flows"].tail(144).median()))
    else:
        st.subheader("Sube CSV de exógenas")
        st.markdown("Columnas esperadas (en cualquier mayúsc/minúsc): `temperature, humidity, wind_speed, general_diffuse_flows` con **una fila por paso**.")
        uploaded_file = st.file_uploader("CSV exógenas para el horizonte", type=["csv"])
        # botón para descargar template
        template = pd.DataFrame({
            "temperature": [float(hist_df["temperature"].tail(144).median())]*int(horizon),
            "humidity": [float(hist_df["humidity"].tail(144).median())]*int(horizon),
            "wind_speed": [float(hist_df["wind_speed"].tail(144).median()) if "wind_speed" in hist_df.columns else 0.0]*int(horizon),
            "general_diffuse_flows": [float(hist_df["general_diffuse_flows"].tail(144).median())]*int(horizon),
        })
        buf = io.StringIO(); template.to_csv(buf, index=False)
        st.download_button("⬇️ Descargar template CSV", data=buf.getvalue(), file_name="exo_template.csv", mime="text/csv")

go_btn = st.button("🚀 Generar pronóstico")

if go_btn:
    try:
        # 3) Construir DF con futuro según entradas
        if exo_mode == "Constantes":
            full_df = make_future_df(
                hist_df,
                pd.to_datetime(start_dt),
                int(horizon),
                exo_source="const",
                const_values=const_cols
            )
        else:
            if uploaded_file is None:
                st.error("Sube un CSV con exógenas o cambia a modo 'Constantes'.")
                st.stop()
            exo_df = pd.read_csv(uploaded_file)
            full_df = make_future_df(
                hist_df,
                pd.to_datetime(start_dt),
                int(horizon),
                exo_source="table",
                exo_table=exo_df
            )

        # 4) Dataset de predicción reutilizando normalizadores del training
        predict_dl = dm.make_predict_from_training(training, full_df)

        # 5) Modelo + pesos
        mm = ModelManager(training_dataset=training)
        mm.build_model()
        mm.load_state_dict_from_url(RAW_STATE_DICT_URL)

        # 6) Predicción
        with st.spinner("Inferencia…"):
            raw = mm.predict_raw(predict_dl)

        # 7) Armar DF de salida
        preds_all = raw.output[0].detach().cpu().numpy()  # (N, L, Q)
        q_list = mm.tft.loss.quantiles
        def qidx(q): return q_list.index(q) if q in q_list else None
        mid = qidx(0.5) if qidx(0.5) is not None else (len(q_list)//2)
        p50 = preds_all[:, :, mid]

        out_rows = []
        for i in range(p50.shape[0]):
            dec_time = raw.x["decoder_time_idx"][i].cpu().numpy().flatten()
            y_true = raw.x["decoder_target"][i].cpu().numpy().flatten()
            row = pd.DataFrame({
                "sample_idx": i,
                "decoder_time_idx": dec_time,
                "y_true": y_true,
                "y_pred_p50": p50[i]
            })
            out_rows.append(row)
        pred_df = pd.concat(out_rows, ignore_index=True)

        st.success("Pronóstico listo ✅")

        # 8) Gráfico
        st.subheader("Predicción (primer sample)")
        plot_prediction_from_raw(raw, idx=0)

        # 9) Métricas SOLO sobre histórico (antes del inicio del futuro)
        t0 = hist_df["datetime"].min()
        future_start_idx = int(((pd.to_datetime(start_dt) - t0) / pd.Timedelta(minutes=10)))
        mask_hist = pred_df["decoder_time_idx"] < future_start_idx

        yt_hist = pred_df.loc[mask_hist, "y_true"].values
        yp_hist = pred_df.loc[mask_hist, "y_pred_p50"].values

        if yt_hist.size > 0 and not np.isnan(yt_hist).all():
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            mae = mean_absolute_error(yt_hist, yp_hist)
            try:
                rmse = mean_squared_error(yt_hist, yp_hist, squared=False)
            except TypeError:
                rmse = np.sqrt(mean_squared_error(yt_hist, yp_hist))
            st.markdown(f"**MAE (histórico):** {mae:.3f} — **RMSE (histórico):** {rmse:.3f}")
        else:
            st.info("Pronóstico puro: el horizonte seleccionado cae íntegramente en futuro (no hay valores reales para métricas).")

        # 10) Tabla + descarga
        st.subheader("Predicciones (todas las ventanas generadas)")
        st.dataframe(pred_df.head(500), width='stretch')
        csv_buf = io.StringIO(); pred_df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Descargar predicciones (CSV)", data=csv_buf.getvalue(),
                           file_name="predicciones_tft_input_exog.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ Error durante el pronóstico: {e}")
