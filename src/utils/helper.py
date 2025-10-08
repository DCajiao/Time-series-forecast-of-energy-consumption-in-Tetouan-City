import pandas as pd

def align_to_10min(dt: pd.Timestamp) -> pd.Timestamp:
    # fuerza a múltiplos de 10 minutos (redondeo hacia abajo)
    return dt.floor("10min")