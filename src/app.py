# app.py
import io
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

PRED_LEN = 24 * 6          # 144 pasos (24h a 10min)
ENCODER_LEN = 7 * 24 * 6   # ~7 días


# =========================
# Utilidades métricas
# =========================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(yt, yp)
    try:
        rmse = mean_squared_error(yt, yp, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(yt, yp))

    mask = yt != 0
    mape = (np.abs((yt[mask] - yp[mask]) / yt[mask]).mean() * 100) if mask.any() else np.nan

    denom = (np.abs(yt) + np.abs(yp))
    sm_mask = denom != 0
    smape = (2.0 * np.abs(yt[sm_mask] - yp[sm_mask]) / denom[sm_mask]).mean() * 100 if sm_mask.any() else np.nan

    wape = (np.abs(yt - yp).sum() / np.abs(yt).sum()) * 100 if np.abs(yt).sum() != 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "WAPE": wape}


# =========================
# Data Manager
# =========================
class DataManager:
    def __init__(
        self,
        csv_url: str = RAW_DATA_URL,
        prediction_length: int = PRED_LEN,
        max_encoder_length: int = ENCODER_LEN,
        weather_as_known: bool = True,   # ← para calzar con el checkpoint
    ):
        self.csv_url = csv_url
        self.prediction_length = prediction_length
        self.max_encoder_length = max_encoder_length
        self.weather_as_known = weather_as_known

        self.df: Optional[pd.DataFrame] = None
        self.training: Optional[TimeSeriesDataSet] = None
        self.validation: Optional[TimeSeriesDataSet] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None

        self.time_varying_known_reals: List[str] = []
        self.time_varying_unknown_reals: List[str] = []

    @st.cache_data(show_spinner=False)
    def load_dataframe(_self) -> pd.DataFrame:
        df = pd.read_csv(_self.csv_url)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        t0 = df["datetime"].min()
        df["time_idx"] = ((df["datetime"] - t0) / pd.Timedelta(minutes=10)).astype(int)

        expected_cols = {"temperature", "humidity", "general_diffuse_flows", "zone_1"}
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en el dataset: {missing}.")

        # calendario
        df["day_of_week"] = df["datetime"].dt.dayofweek
        df["hour"] = df["datetime"].dt.hour
        df["day"] = df["datetime"].dt.day
        df["month"] = df["datetime"].dt.month
        df["is_weekend"] = df["day_of_week"] >= 5
        df["is_holiday"] = False

        df["zone"] = "zone_1"
        df = df.sort_values(["zone", "time_idx"]).reset_index(drop=True)

        if df["zone_1"].isna().any():
            raise ValueError("NaN en target 'zone_1'. Limpia o imputa antes.")

        _self.df = df
        return df

    def make_datasets(self) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        if self.df is None:
            raise RuntimeError("Primero ejecuta load_dataframe().")

        df = self.df
        training_cutoff = df["time_idx"].max() - self.prediction_length

        known_reals = ["time_idx", "hour", "day", "day_of_week", "month", "is_weekend", "is_holiday"]
        weather = ["temperature", "humidity"] + (["wind_speed"] if "wind_speed" in df.columns else []) + ["general_diffuse_flows"]

        if self.weather_as_known:
            self.time_varying_known_reals = known_reals + weather
            self.time_varying_unknown_reals = ["zone_1"]
        else:
            self.time_varying_known_reals = known_reals
            self.time_varying_unknown_reals = ["zone_1"] + weather

        training = TimeSeriesDataSet(
            df[df.time_idx <= training_cutoff],
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
        validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

        self.training = training
        self.validation = validation
        return training, validation

    def make_dataloaders(self, batch_size: int = 64, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
        # num_workers=0 para compatibilidad amplia con entornos locales/Windows
        if self.training is None or self.validation is None:
            raise RuntimeError("Primero ejecuta make_datasets().")
        self.train_dataloader = self.training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
        self.val_dataloader = self.validation.to_dataloader(train=False, batch_size=batch_size * 4, num_workers=num_workers)
        return self.train_dataloader, self.val_dataloader


# =========================
# Model Manager
# =========================
class ModelManager:
    def __init__(
        self,
        training_dataset: TimeSeriesDataSet,
        device: Optional[str] = None,
        state_dict_url: str = RAW_STATE_DICT_URL,
        learning_rate: float = LEARNING_RATE,
        hidden_size: int = HIDDEN_SIZE,
        attention_head_size: int = ATTN_HEADS,
        dropout: float = DROPOUT,
        hidden_continuous_size: int = HIDDEN_CONT,
        quantiles: Optional[List[float]] = None,
    ):
        self.training_dataset = training_dataset
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dict_url = state_dict_url
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.quantiles = quantiles or QUANTILES
        self.tft: Optional[TemporalFusionTransformer] = None

    @st.cache_resource(show_spinner=False)
    def build_model(_self):
        tft = TemporalFusionTransformer.from_dataset(
            _self.training_dataset,
            learning_rate=_self.learning_rate,
            hidden_size=_self.hidden_size,
            attention_head_size=_self.attention_head_size,
            dropout=_self.dropout,
            hidden_continuous_size=_self.hidden_continuous_size,
            loss=QuantileLoss(quantiles=_self.quantiles),
        )
        tft.to(_self.device)
        tft.eval()
        _self.tft = tft
        return tft

    def load_state_dict_from_url(self):
        if self.tft is None:
            raise RuntimeError("Primero ejecuta build_model().")
        resp = requests.get(self.state_dict_url, timeout=60)
        resp.raise_for_status()
        buffer = BytesIO(resp.content)
        state = torch.load(buffer, map_location=self.device)
        self.tft.load_state_dict(state)
        self.tft.to(self.device)
        self.tft.eval()

    @torch.no_grad()
    def predict_raw(self, dataloader: DataLoader):
        if self.tft is None:
            raise RuntimeError("Primero build_model() y luego load_state_dict_from_url().")
        return self.tft.predict(dataloader, mode="raw", return_x=True)

    @torch.no_grad()
    def predict_p50(self, dataloader: DataLoader) -> np.ndarray:
        raw = self.predict_raw(dataloader)
        preds = raw.output[0].detach().cpu().numpy()  # (n_samples, L, n_quantiles)
        q = self.tft.loss.quantiles
        try:
            mid = q.index(0.5)
        except ValueError:
            mid = len(q) // 2
        return preds[:, :, mid], raw  # devuelvo también el raw para gráficos


# =========================
# Plotting helpers (matplotlib)
# =========================
def plot_sample(raw, sample_idx: int, title: str = "Predicción (p50) con banda p10–p90"):
    """
    Dibuja histórico (encoder), futuro real (decoder) y cuantiles p10/p50/p90 del sample seleccionado.
    """
    x = raw.x

    encoder_target = x["encoder_target"][sample_idx].cpu().numpy().flatten()
    decoder_target = x["decoder_target"][sample_idx].cpu().numpy().flatten()
    decoder_time = x["decoder_time_idx"][sample_idx].cpu().numpy().flatten()
    encoder_time = np.arange(decoder_time[0] - len(encoder_target), decoder_time[0])

    preds = raw.output[0][sample_idx].detach().cpu().numpy()  # (L, n_quant)

    # Buscar índices de p10/p50/p90
    q_list = raw.temporal_fusion_transformer.loss.quantiles if hasattr(raw, "temporal_fusion_transformer") else QUANTILES
    def q_idx(q):
        return q_list.index(q) if q in q_list else None

    p10 = q_idx(0.1)
    p50 = q_idx(0.5)
    p90 = q_idx(0.9)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(encoder_time, encoder_target, label="Histórico (encoder)", linewidth=1.5)
    ax.plot(decoder_time, decoder_target, label="Real futuro (decoder)", linewidth=1.5)

    if p50 is not None:
        ax.plot(decoder_time, preds[:, p50], label="Pred p50", linestyle="--")

    if p10 is not None and p90 is not None:
        ax.fill_between(decoder_time, preds[:, p10], preds[:, p90], alpha=0.3, label="Banda p10–p90")

    ax.set_title(title)
    ax.set_xlabel("time_idx (10-min)")
    ax.set_ylabel("Consumo (unidades originales)")
    ax.legend()
    st.pyplot(fig)


def build_predictions_dataframe(raw, p50: np.ndarray) -> pd.DataFrame:
    """
    Construye un DataFrame con decoder_time_idx, y_true y pred_p50 para todas las muestras.
    """
    x = raw.x
    all_rows = []
    for i in range(len(x["decoder_target"])):
        decoder_target = x["decoder_target"][i].cpu().numpy().flatten()
        decoder_time = x["decoder_time_idx"][i].cpu().numpy().flatten()
        df_i = pd.DataFrame({
            "sample_idx": i,
            "decoder_time_idx": decoder_time,
            "y_true": decoder_target,
            "y_pred_p50": p50[i]
        })
        all_rows.append(df_i)
    return pd.concat(all_rows, ignore_index=True)


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="TFT — Consumo eléctrico Zone_1 (Tetuán)", layout="wide")
st.title("🔌 Temporal Fusion Transformer — Consumo eléctrico (Zone_1, 10-min)")
st.caption("Carga modelo y datos desde GitHub, ejecuta inferencia y visualiza predicciones con bandas cuantílicas.")

with st.sidebar:
    st.header("Configuración")
    st.markdown("**Fuentes (RAW GitHub):**")
    st.code(f"CSV:  {RAW_DATA_URL}\nSTATE: {RAW_STATE_DICT_URL}", language="text")
    weather_known = st.checkbox("Tratar clima como KNOWN (coincide con checkpoint)", value=True, help="Dejar activado para usar el state_dict publicado")
    batch_size = st.slider("Batch size validación", min_value=16, max_value=256, value=64, step=16)
    sample_to_plot = st.number_input("Sample del batch para graficar", min_value=0, value=0, step=1)
    run_btn = st.button("🚀 Cargar y predecir")

if run_btn:
    try:
        with st.spinner("Cargando datos desde GitHub y preparando dataset…"):
            dm = DataManager(
                csv_url=RAW_DATA_URL,
                prediction_length=PRED_LEN,
                max_encoder_length=ENCODER_LEN,
                weather_as_known=weather_known,
            )
            df = dm.load_dataframe()
            training, validation = dm.make_datasets()
            _, val_loader = dm.make_dataloaders(batch_size=batch_size, num_workers=0)

        st.success("Datos listos ✅")
        st.write("**Decoder KNOWN reals:**", dm.time_varying_known_reals)
        st.write("**Decoder UNKNOWN reals:**", dm.time_varying_unknown_reals)

        with st.spinner("Reconstruyendo TFT y cargando pesos…"):
            mm = ModelManager(training_dataset=training)
            tft = mm.build_model()
            mm.load_state_dict_from_url()

        st.success("Modelo cargado ✅")
        st.write(f"**Cuantiles del modelo:** {mm.tft.loss.quantiles}")

        with st.spinner("Haciendo predicción…"):
            p50, raw = mm.predict_p50(val_loader)

        # Construir y_true para métricas
        y_true_list = []
        for _, yb in val_loader:
            y_true_list.append(yb[0].detach().cpu().numpy())
        y_true = np.concatenate(y_true_list, axis=0)

        metrics = compute_metrics(y_true, p50)

        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.subheader("Métricas (validación, p50)")
            mtable = pd.DataFrame({
                "Métrica": ["MAE", "RMSE", "MAPE", "sMAPE", "WAPE"],
                "Valor": [
                    f"{metrics['MAE']:.3f}",
                    f"{metrics['RMSE']:.3f}",
                    f"{metrics['MAPE']:.2f}%",
                    f"{metrics['sMAPE']:.2f}%",
                    f"{metrics['WAPE']:.2f}%"
                ]
            })
            st.table(mtable)

            # Último punto del horizonte del sample seleccionado
            t_final = -1
            try:
                y_last = raw.x["decoder_target"][sample_to_plot][t_final].item()
                preds_sample = raw.output[0][sample_to_plot].detach().cpu().numpy()
                q_list = mm.tft.loss.quantiles
                row = {"Valor real (último paso)": y_last}
                for q in [0.1, 0.5, 0.9]:
                    if q in q_list:
                        qidx = q_list.index(q)
                        row[f"p{int(q*100)}"] = float(preds_sample[t_final, qidx])
                st.subheader("Último paso del horizonte (sample seleccionado)")
                st.table(pd.DataFrame([row]))
            except Exception as e:
                st.warning(f"No se pudo mostrar el último paso: {e}")

        with col2:
            st.subheader("Predicción — sample seleccionado")
            plot_sample(raw, sample_to_plot)

        # Tabla de predicciones y descarga
        st.subheader("Predicciones (todas las muestras del batch)")
        pred_df = build_predictions_dataframe(raw, p50)
        st.dataframe(pred_df.head(500), use_container_width=True)

        csv_buf = io.StringIO()
        pred_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="⬇️ Descargar predicciones (CSV)",
            data=csv_buf.getvalue(),
            file_name="predicciones_tft_zone1.csv",
            mime="text/csv",
        )

    except Exception as ex:
        st.error(f"⚠️ Ocurrió un error: {ex}")
        st.stop()


# ====== Footer ======
st.markdown("---")
st.caption("TFT con PyTorch Forecasting — Zone_1 (Tetuán, 2017, resolución 10-min). "
           "Predicción 24h, clima como KNOWN para coincidir con el checkpoint publicado.")
