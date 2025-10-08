# main.py
import argparse
from html import parser
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error

from core.data_manager import DataManager, RAW_DATA_URL
from core.model_manager import ModelManager, RAW_STATE_DICT_URL


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    mae = mean_absolute_error(yt, yp)
    try:
        rmse = root_mean_squared_error(yt, yp, squared=False)  # sklearn >= 0.22
    except TypeError:
        rmse = np.sqrt(mean_squared_error(yt, yp))        # fallback

    mask = yt != 0
    mape = (np.abs((yt[mask] - yp[mask]) / yt[mask]).mean() * 100) if mask.any() else np.nan

    denom = (np.abs(yt) + np.abs(yp))
    sm_mask = denom != 0
    smape = (2.0 * np.abs(yt[sm_mask] - yp[sm_mask]) / denom[sm_mask]).mean() * 100 if sm_mask.any() else np.nan

    wape = (np.abs(yt - yp).sum() / np.abs(yt).sum()) * 100 if np.abs(yt).sum() != 0 else np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "WAPE": wape}


def main():
    parser = argparse.ArgumentParser(description="Inferencia TFT — Consumo eléctrico Zone_1 (Tetuán)")
    parser.add_argument("--csv", type=str, default=RAW_DATA_URL, help="URL al CSV enriquecido (raw GitHub)")
    parser.add_argument("--state_dict_url", type=str, default=RAW_STATE_DICT_URL, help="URL raw al state_dict del TFT")
    parser.add_argument("--weather_as_known", action="store_true", help="(opcional) Ya está activado por defecto")

    args = parser.parse_args()

    # 1) Datos (usa por defecto el RAW de GitHub)
    dm = DataManager(
        csv_url=args.csv,
        prediction_length=24 * 6,
        max_encoder_length=7 * 24 * 6,
        weather_as_known=True,  # fijo a True
    )
    df = dm.load_dataframe()
    training, validation = dm.make_datasets()
    _, val_loader = dm.make_dataloaders()

    # 2) Modelo (descarga y carga state_dict desde RAW GitHub)
    mm = ModelManager(
        training_dataset=training,
        learning_rate=1e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        state_dict_url=args.state_dict_url,
    )
    mm.build_model()
    print("\n=== Chequeo de configuración ===")
    print("Decoder KNOWN reals (según DataManager):", dm.time_varying_known_reals)
    print("Decoder UNKNOWN reals (según DataManager):", dm.time_varying_unknown_reals)
    print("Nº de cuantiles en el modelo:", len(mm.tft.loss.quantiles), "| Quantiles:", mm.tft.loss.quantiles)

    mm.load_state_dict_from_url()

    # 3) Predicción (validación como proxy)
    y_pred = mm.predict_p50(val_loader)  # (n_samples, prediction_length)

    # Construir y_true a partir del val_loader
    y_true_list = []
    for _, y_batch in val_loader:
        y_true_list.append(y_batch[0].detach().cpu().numpy())
    y_true = np.concatenate(y_true_list, axis=0)

    # 4) Métricas
    metrics = compute_metrics(y_true, y_pred)
    print("\n=== Métricas validación (p50) ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}%" if k in ["MAPE", "sMAPE", "WAPE"] else f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")

    # 5) Ejemplo del último paso del primer sample
    t_final = -1
    print("\nEjemplo — último punto del horizonte del primer sample:")
    print(f"y_true: {y_true[0, t_final]:.2f} | y_pred(p50): {y_pred[0, t_final]:.2f}")


if __name__ == "__main__":
    main()
