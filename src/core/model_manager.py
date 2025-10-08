# model_manager.py
from typing import Tuple, Optional, List
import numpy as np
import torch

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from torch.utils.data import DataLoader


class ModelManager:
    """
    Reconstruye el TFT desde un TimeSeriesDataSet y carga el state_dict exportado.
    Permite predecir (raw o p50) sobre un DataLoader dado (val o predict).
    """
    def __init__(
        self,
        training_dataset,  # TimeSeriesDataSet (data_manager.training)
        learning_rate: float = 1e-3,
        hidden_size: int = 64,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 32,
        quantiles: Optional[List[float]] = None,
        device: Optional[str] = None,
    ):
        self.training_dataset = training_dataset
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tft: Optional[TemporalFusionTransformer] = None

    def build_model(self):
        self.tft = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            loss=QuantileLoss(quantiles=self.quantiles),
        )
        self.tft.to(self.device)
        self.tft.eval()

    def load_state_dict(self, state_dict_path: str):
        if self.tft is None:
            raise RuntimeError("Primero ejecuta build_model().")
        state = torch.load(state_dict_path, map_location=self.device)
        self.tft.load_state_dict(state)
        self.tft.to(self.device)
        self.tft.eval()

    @torch.no_grad()
    def predict_raw(self, dataloader: DataLoader):
        if self.tft is None:
            raise RuntimeError("Primero ejecuta build_model() y load_state_dict().")
        return self.tft.predict(dataloader, mode="raw", return_x=True)

    @torch.no_grad()
    def predict_p50(self, dataloader: DataLoader) -> np.ndarray:
        """
        Devuelve la mediana (p50) con shape (n_samples, prediction_length).
        """
        raw = self.predict_raw(dataloader)
        # raw.output[0]: (n_samples, prediction_length, n_quantiles)
        preds = raw.output[0].detach().cpu().numpy()
        # localizar índice de p50
        q = self.tft.loss.quantiles
        try:
            mid = q.index(0.5)
        except ValueError:
            mid = len(q) // 2
        return preds[:, :, mid]
