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

from utils import helper

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

    # indicador auxiliar (no es feature del modelo)
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


def _make_future_block(
    hist_df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    const_values: dict | None = None,
    exo_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Crea un bloque futuro contiguo en [start_idx, end_idx] (inclusive), SIN solapar histórico.
    """
    assert end_idx >= start_idx
    t0 = hist_df["datetime"].min()
    future_dt = pd.date_range(
        start=t0 + pd.Timedelta(minutes=10 * start_idx),
        periods=(end_idx - start_idx + 1),
        freq="10min",
    )
    fut = pd.DataFrame({
        "datetime": future_dt,
        "time_idx": np.arange(start_idx, end_idx + 1, dtype=int),
        "zone": "zone_1",
    })
    fut = add_calendar_feats(fut)

    # exógenas
    exo_cols = ["temperature", "humidity", "wind_speed", "general_diffuse_flows"]
    if exo_table is not None:
        tbl = exo_table.copy()
        tbl.columns = [c.strip().lower() for c in tbl.columns]
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
        block = pd.DataFrame(index=range(len(fut)), columns=exo_cols, dtype=float)
        for src, dst in col_map.items():
            if src in tbl.columns:
                block[dst] = pd.to_numeric(tbl[src], errors="coerce")
        if block.isna().any().any():
            raise ValueError("NaNs en CSV de exógenas para el futuro.")
        fut[exo_cols] = block.values
    else:
        const_values = const_values or {}
        for c in exo_cols:
            fut[c] = float(const_values.get(c, 0.0))

    # target placeholder (NO NaN)
    fut["zone_1"] = 0.0
    fut["is_future"] = 1
    return fut


def build_df_for_prediction(
    hist_df: pd.DataFrame,
    start_dt: pd.Timestamp,
    horizon: int,
    exo_mode: str,
    const_values: dict | None,
    exo_table: pd.DataFrame | None,
) -> tuple[pd.DataFrame, int, int, bool]:
    """
    Devuelve:
      - df_pred: DataFrame listo para from_dataset(..., predict=True)
      - dec_start_idx: índice donde EMPIEZA el decoder (efectivo)
      - dec_end_idx: índice donde TERMINA el decoder (efectivo)
      - is_backtest: True si hay solape con histórico (al menos 1 paso real en decoder)
    """
    df = hist_df.copy()
    df["is_future"] = df.get("is_future", 0)
    df["is_future"] = df["is_future"].astype(int)

    t0 = df["datetime"].min()
    last_idx = int(df["time_idx"].max())
    start_idx = int(((pd.to_datetime(start_dt) - t0) / pd.Timedelta(minutes=10)))
    req_end_idx = start_idx + int(horizon) - 1  # lo deseado por el usuario

    # --- Caso 1: Todo el decoder dentro de histórico (BACKTEST total)
    if req_end_idx <= last_idx and start_idx >= 0:
        # recorto hasta end para que el decoder quede justo [start_idx, req_end_idx]
        df_bt = df[df["time_idx"] <= req_end_idx].copy()

        # what-if: sobreescribir exógenas en [start_idx, req_end_idx]
        if exo_mode.lower().startswith("const"):
            for col, val in (const_values or {}).items():
                if col in df_bt.columns:
                    mask = (df_bt["time_idx"] >= start_idx) & (df_bt["time_idx"] <= req_end_idx)
                    df_bt.loc[mask, col] = float(val)
        elif exo_table is not None:
            expected_len = req_end_idx - start_idx + 1
            if len(exo_table) != expected_len:
                raise ValueError(f"CSV de exógenas debe tener {expected_len} filas para el tramo histórico.")
            block = _make_future_block(df, start_idx, req_end_idx, exo_table=exo_table)
            for c in ["temperature","humidity","wind_speed","general_diffuse_flows"]:
                df_bt.loc[(df_bt["time_idx"] >= start_idx) & (df_bt["time_idx"] <= req_end_idx), c] = block[c].values

        return df_bt, start_idx, req_end_idx, True

    # --- Caso 2: Empieza en histórico pero se pasa del final (BACKTEST parcial + FORECAST parcial)
    if start_idx <= last_idx < req_end_idx:
        # parte histórica: hasta last_idx (para que el decoder termine en last_idx)
        df_mix = df[df["time_idx"] <= last_idx].copy()

        # parte futura: desde last_idx+1 hasta req_end_idx SIN huecos
        fut_start = last_idx + 1
        fut_end = req_end_idx

        if exo_mode.lower().startswith("const"):
            fut = _make_future_block(df, fut_start, fut_end, const_values=const_values)
        else:
            expected_len = fut_end - fut_start + 1
            if exo_table is None or len(exo_table) != expected_len:
                raise ValueError(f"CSV de exógenas debe tener {expected_len} filas para el tramo futuro.")
            fut = _make_future_block(df, fut_start, fut_end, exo_table=exo_table)

        df_mix = pd.concat([df_mix, fut], ignore_index=True)
        df_mix["is_future"] = df_mix["is_future"].astype(int)

        # el decoder empieza donde pidió el usuario, aunque parte sea histórica y parte futura
        return df_mix, start_idx, req_end_idx, True

    # --- Caso 3: Todo el decoder cae en el futuro (FORECAST total)
    if start_idx > last_idx:
        fut_start = last_idx + 1
        fut_end = req_end_idx
        if exo_mode.lower().startswith("const"):
            fut = _make_future_block(df, fut_start, fut_end, const_values=const_values)
        else:
            expected_len = fut_end - fut_start + 1
            if exo_table is None or len(exo_table) != expected_len:
                raise ValueError(f"CSV de exógenas debe tener {expected_len} filas para el tramo futuro.")
            fut = _make_future_block(df, fut_start, fut_end, exo_table=exo_table)

        df_fc = pd.concat([df, fut], ignore_index=True)
        df_fc["is_future"] = df_fc["is_future"].astype(int)
        return df_fc, start_idx, req_end_idx, False

    # --- Caso 4: start_idx < 0 (fuera por la izquierda) -> ajustamos al inicio
    if start_idx < 0:
        # forzamos a empezar en 0 y avisamos que se movió el inicio efectivo
        eff_start = 0
        eff_end = int(horizon) - 1
        df_bt = df[df["time_idx"] <= min(eff_end, last_idx)].copy()
        return df_bt, eff_start, min(eff_end, last_idx), True


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
        # Asegurar is_future existe y numérica (0/1) — NO es feature del modelo
        if "is_future" not in df.columns:
            df = df.copy()
            df["is_future"] = 0
        else:
            df["is_future"] = df["is_future"].astype(int)

        # NO incluir is_future como known real para calzar con el checkpoint
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
        predict_ds = TimeSeriesDataSet.from_dataset(training, df_with_future, predict=True, stop_randomization=True)
        predict_dl = predict_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
        return predict_dl


class ModelManager:
    def __init__(self, training_dataset: TimeSeriesDataSet, quantiles: Optional[List[float]] = None):
        self.training_dataset = training_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quantiles = quantiles or QUANTILES
        self.tft: Optional[TemporalFusionTransformer] = None

    # ⛔️ sin @st.cache_resource para evitar issues de idempotencia
    def build_model(self):
        tft = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=LEARNING_RATE,
            hidden_size=HIDDEN_SIZE,
            attention_head_size=ATTN_HEADS,
            dropout=DROPOUT,
            hidden_continuous_size=HIDDEN_CONT,
            loss=QuantileLoss(quantiles=self.quantiles),
        )
        tft.to(self.device)
        tft.eval()
        self.tft = tft
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
st.title("⚡ TFT — Pronóstico de consumo en la zona 1 de Tetuán, Marruecos")
st.caption("Ingresa variables exógenas y genera pronósticos con nuestro TFT entrenado.")

with st.sidebar:
    st.header("Parámetros de pronóstico")
    horizon = st.number_input("Horizonte (pasos de 10 min)", min_value=6, max_value=7*24*6, value=PRED_LEN_DEFAULT, step=6)
    exo_mode = st.radio("Modo de exógenas", options=["Constantes"])  # puedes reactivar CSV cuando quieras

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

    # Alinear a rejilla de 10 min
    start_dt = datetime.datetime.combine(start_date, start_time)
    start_dt = helper.align_to_10min(pd.to_datetime(start_dt))
    if start_dt != pd.to_datetime(datetime.datetime.combine(start_date, start_time)):
        st.info(f"Fecha/hora alineada a rejilla de 10 min: {start_dt}")

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
        # 3) DF predicción: decide backtest vs forecast y garantiza contigüidad
        if exo_mode == "Constantes":
            full_df, dec_start_idx, dec_end_idx, has_history = build_df_for_prediction(
                hist_df, start_dt, int(horizon), exo_mode, const_cols, None
            )
        else:
            if uploaded_file is None:
                st.error("Sube un CSV con exógenas o cambia a modo 'Constantes'.")
                st.stop()
            exo_df = pd.read_csv(uploaded_file)
            full_df, future_start_idx, is_backtest = build_df_for_prediction(
                hist_df, start_dt, int(horizon), exo_mode, None, exo_df
            )

        # 4) Dataset de predicción reutilizando normalizadores del training
        predict_dl = dm.make_predict_from_training(training, full_df)

        # 5) Modelo + pesos (idempotente)
        mm = st.session_state.get("mm")
        need_new = (
            mm is None
            or getattr(mm, "training_dataset", None) is not training
            or getattr(mm, "tft", None) is None
        )
        if need_new:
            mm = ModelManager(training_dataset=training)
            mm.build_model()
            mm.load_state_dict_from_url(RAW_STATE_DICT_URL)
            st.session_state["mm"] = mm

        # 6) Predicción
        with st.spinner("Inferencia…"):
            raw = mm.predict_raw(predict_dl)

        # 7) Armar DF de salida
        preds_all = raw.output[0].detach().cpu().numpy()  # (N, L, Q)
        q_list = mm.tft.loss.quantiles
        def qidx(q): return q_list.index(q) if q in q_list else None
        mid = qidx(0.5) if qidx(0.5) is not None else (len(q_list) // 2)
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

        original_last_idx = int(hist_df["time_idx"].max())

        # Marcar como NaN SOLO lo que realmente no existe en histórico
        mask_future = pred_df["decoder_time_idx"] > original_last_idx
        pred_df.loc[mask_future, "y_true"] = np.nan

        # (opcional) Mensajes de contexto
        if dec_end_idx <= original_last_idx:
            st.caption(f"Modo backtest: decoder exacto en [{dec_start_idx}, {dec_end_idx}] dentro del histórico.")
        elif dec_start_idx <= original_last_idx < dec_end_idx:
            st.caption(f"Modo mixto: decoder [{dec_start_idx}, {dec_end_idx}] toca histórico hasta {original_last_idx} y luego futuro.")
        else:
            st.caption(f"Modo forecast: decoder [{dec_start_idx}, {dec_end_idx}] íntegramente en futuro.")

        st.success("Pronóstico listo ✅")

        # 8) Gráfico
        st.subheader("Predicción (primer sample)")
        plot_prediction_from_raw(raw, idx=0)

        # 9) Métricas SOLO histórico (<= original_last_idx)
        mask_hist = pred_df["decoder_time_idx"] <= original_last_idx
        yt_hist = pred_df.loc[mask_hist, "y_true"].values
        yp_hist = pred_df.loc[mask_hist, "y_pred_p50"].values

        if yt_hist.size > 0 and not np.isnan(yt_hist).all():
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            mae = mean_absolute_error(yt_hist, yp_hist)
            try:
                rmse = mean_squared_error(yt_hist, yp_hist, squared=False)
            except TypeError:
                rmse = np.sqrt(mean_squared_error(yt_hist, yp_hist))

            # MAPE (ignorar yt == 0 para evitar división por cero)
            nz = yt_hist != 0
            if nz.any():
                mape = (np.abs((yt_hist[nz] - yp_hist[nz]) / yt_hist[nz]).mean()) * 100.0
                st.markdown(f"**MAE (histórico):** {mae:.3f} — **RMSE (histórico):** {rmse:.3f} — **MAPE (histórico):** {mape:.2f}%")
            else:
                st.markdown(f"**MAE (histórico):** {mae:.3f} — **RMSE (histórico):** {rmse:.3f} — **MAPE (histórico):** N/A (target=0)")

        # 10) Tabla + descarga
        st.subheader("Predicciones (todas las ventanas generadas)")
        st.dataframe(pred_df.head(500), width='stretch')
        csv_buf = io.StringIO(); pred_df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Descargar predicciones (CSV)", data=csv_buf.getvalue(),
                           file_name="predicciones_tft_input_exog.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ Error durante el pronóstico: {e}")
