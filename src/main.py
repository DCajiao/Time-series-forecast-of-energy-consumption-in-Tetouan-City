# main.py
import argparse
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from core.data_manager import DataManager
from core.model_manager import ModelManager


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    y_true, y_pred con shape (n_samples, prediction_length)
    Devuelve MAE, RMSE, MAPE, sMAPE y WAPE.
    """
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    mae = mean_absolute_error(yt, yp)
    rmse = root_mean_squared_error(yt, yp, squared=False)

    mask = yt != 0
    mape = (np.abs((yt[mask] - yp[mask]) / yt[mask]).mean() * 100) if mask.any() else np.nan

    # sMAPE
    denom = (np.abs(yt) + np.abs(yp))
    sm_mask = denom != 0
    smape = (2.0 * np.abs(yt[sm_mask] - yp[sm_mask]) / denom[sm_mask]).mean() * 100 if sm_mask.any() else np.nan

    # WAPE
    wape = (np.abs(yt - yp).sum() / np.abs(yt).sum()) * 100 if np.abs(yt).sum() != 0 else np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "WAPE": wape}


def main():
    parser = argparse.ArgumentParser(description="Inferencia TFT — Consumo eléctrico Zone_1 (Tetuán)")
    parser.add_argument("--csv", type=str, required=True, help="Ruta o URL al CSV enriquecido (Zone_1)")
    parser.add_argument("--state_dict", type=str, required=True, help="Ruta al archivo .pt con state_dict del TFT")
    parser.add_argument("--weather_as_known", action="store_true", help="Tratar variables clima como conocidas a futuro")
    args = parser.parse_args()

    # 1) Datos
    dm = DataManager(
        csv_path_or_url=args.csv,
        prediction_length=24 * 6,
        max_encoder_length=7 * 24 * 6,
        weather_as_known=args.weather_as_known,
    )
    df = dm.load_dataframe()
    training, validation = dm.make_datasets()
    train_loader, val_loader = dm.make_dataloaders()

    # 2) Modelo
    mm = ModelManager(
        training_dataset=training,
        learning_rate=1e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        quantiles=[0.1, 0.5, 0.9],
    )
    mm.build_model()
    mm.load_state_dict(args.state_dict)

    # 3) Predicción (usar validación como proxy de inferencia)
    y_pred = mm.predict_p50(val_loader)  # (n_samples, prediction_length)

    # Construir y_true a partir del val_loader
    y_true_list = []
    for _, y_batch in val_loader:
        # y_batch[0] = target del decoder
        y_true_list.append(y_batch[0].detach().cpu().numpy())
    y_true = np.concatenate(y_true_list, axis=0)

    # 4) Métricas
    metrics = compute_metrics(y_true, y_pred)
    print("\n=== Métricas validación (p50) ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            if k in ["MAPE", "sMAPE", "WAPE"]:
                print(f"{k}: {v:.2f}%")
            else:
                print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")

    # 5) Ejemplo de última fila (último punto del horizonte)
    t_final = -1
    print("\nEjemplo — último punto del horizonte del primer sample:")
    print(f"y_true: {y_true[0, t_final]:.2f} | y_pred(p50): {y_pred[0, t_final]:.2f}")

    # (Opcional) Guardar predicciones a CSV
    # np.save("pred_p50.npy", y_pred)
    # print("Predicciones guardadas en pred_p50.npy")


if __name__ == "__main__":
    main()
