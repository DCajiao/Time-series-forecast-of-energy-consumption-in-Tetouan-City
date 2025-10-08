# data_manager.py
from typing import Tuple, Optional
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

RAW_DATA_URL = "https://raw.githubusercontent.com/DCajiao/Time-series-forecast-of-energy-consumption-in-Tetouan-City/main/data/enriched_zone1_power_consumption_of_tetouan_city.csv"


class DataManager:
    """
    Carga y prepara el dataset de Tetuán (Zone_1) para inferencia con TFT.
    Reproduce las transformaciones del notebook:
      - parseo de datetime
      - time_idx cada 10 minutos
      - features de calendario
      - validación de columnas mínimas
      - construcción de TimeSeriesDataSet (train/val)
    """
    def __init__(
        self,
        csv_url: str = RAW_DATA_URL,
        prediction_length: int = 24 * 6,        # 24h * 6 = 144
        max_encoder_length: int = 7 * 24 * 6,   # ~7 días
        weather_as_known: bool = True,         # por defecto: clima como UNKNOWN a futuro
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

    # -------------------
    # Carga y features
    # -------------------
    def load_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_url)  # lee directo desde raw GitHub
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        # time_idx a resolución 10-min
        t0 = df["datetime"].min()
        df["time_idx"] = ((df["datetime"] - t0) / pd.Timedelta(minutes=10)).astype(int)

        # Validación mínimas (como en notebook)
        expected_cols = {"temperature", "humidity", "general_diffuse_flows", "zone_1"}
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en el dataset: {missing}. "
                             f"Columnas disponibles: {df.columns.tolist()}")

        # Features calendario
        df["day_of_week"] = df["datetime"].dt.dayofweek
        df["hour"] = df["datetime"].dt.hour
        df["day"] = df["datetime"].dt.day
        df["month"] = df["datetime"].dt.month
        df["is_weekend"] = df["day_of_week"] >= 5

        # ID de grupo (una sola zona)
        df["zone"] = "zone_1"

        # Orden y limpieza básica
        df = df.sort_values(["zone", "time_idx"]).reset_index(drop=True)

        # TFT no admite NaN en el target
        if df["zone_1"].isna().any():
            raise ValueError("Se encontraron NaN en el target 'zone_1'. Imputa o elimina antes de continuar.")

        self.df = df
        return df

    # -------------------
    # Datasets y loaders
    # -------------------
    def make_datasets(self) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        if self.df is None:
            raise RuntimeError("Primero ejecuta load_dataframe().")

        df = self.df
        training_cutoff = df["time_idx"].max() - self.prediction_length

        known_reals = ["time_idx", "hour", "day", "day_of_week", "month", "is_weekend", "is_holiday"]
        weather = ["temperature", "humidity", "wind_speed", "general_diffuse_flows"] if "wind_speed" in df.columns else ["temperature", "humidity", "general_diffuse_flows"]

        if self.weather_as_known:
            time_varying_known_reals = known_reals + weather
            time_varying_unknown_reals = ["zone_1"]
        else:
            time_varying_known_reals = known_reals
            time_varying_unknown_reals = ["zone_1"] + weather

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
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=["zone"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

        self.training = training
        self.validation = validation
        return training, validation

    def make_dataloaders(self, batch_size: int = 64, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
        if self.training is None or self.validation is None:
            raise RuntimeError("Primero ejecuta make_datasets().")

        self.train_dataloader = self.training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
        self.val_dataloader = self.validation.to_dataloader(train=False, batch_size=batch_size * 4, num_workers=num_workers)
        return self.train_dataloader, self.val_dataloader
