# Time Series Forecast of Energy Consumption in Tetouan City

## Descripción del Proyecto

Este proyecto implementa un sistema de pronóstico de series temporales para predecir el consumo de energía eléctrica en la Zona 1 de la ciudad de Tetuán, Marruecos. Utiliza el modelo **Temporal Fusion Transformer (TFT)** de PyTorch Forecasting para realizar predicciones con una resolución de 10 minutos.

## Características Principales

- **Modelo TFT (Temporal Fusion Transformer)**: Arquitectura de deep learning especializada en series temporales con atención multi-cabeza
- **Predicción cuantílica**: Genera predicciones probabilísticas con 7 cuantiles (0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95)
- **Variables exógenas**: Incorpora temperatura, humedad, velocidad del viento y flujos difusos generales
- **Características temporales**: Día de la semana, hora, día del mes, mes, fin de semana
- **Interfaz web interactiva**: Aplicación Streamlit para visualización y predicción
- **Horizonte configurable**: Por defecto 24 horas (144 pasos de 10 minutos)
- **Encoder largo**: 7 días de contexto histórico para capturar patrones semanales

## Estructura del Proyecto

```
src/
├── app.py                    # Aplicación web Streamlit
├── main.py                   # Script CLI para inferencia y evaluación
├── core/
│   ├── data_manager.py       # Gestión de datos y TimeSeriesDataSet
│   └── model_manager.py      # Gestión del modelo TFT y predicciones
├── utils/
│   ├── helper.py             # Funciones auxiliares (alineación temporal)
│   ├── formatter.py          # Utilidades de formato (vacío)
│   └── logger.py             # Utilidades de logging (vacío)
└── test/                     # Scripts de prueba y validación
```

## Requisitos del Sistema

### Dependencias Principales

```
Python 3.8+
torch >= 1.9.0
pytorch-forecasting >= 0.10.0
streamlit >= 1.20.0
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 0.24.0
matplotlib >= 3.4.0
requests >= 2.26.0
```

### Hardware Recomendado

- **CPU**: Mínimo 4 cores
- **RAM**: Mínimo 8 GB
- **GPU**: Opcional (CUDA compatible) para inferencia más rápida

## Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/DCajiao/Time-series-forecast-of-energy-consumption-in-Tetouan-City.git
   cd Time-series-forecast-of-energy-consumption-in-Tetouan-City/src
   ```

2. **Crear entorno virtual** (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: .\venv\Scripts\activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install torch pytorch-forecasting streamlit pandas numpy scikit-learn matplotlib requests
   ```

## Uso

### 1. Aplicación Web (Streamlit)

Ejecuta la interfaz web interactiva:

```bash
streamlit run app.py
```

La aplicación permite:
- **Cargar datos históricos** desde GitHub o archivo local
- **Configurar parámetros de predicción** (fecha inicio, horizonte, variables exógenas)
- **Visualizar predicciones** con bandas de confianza (p10-p90)
- **Exportar resultados** en formato CSV
- **Modo backtest**: Evaluar predicciones contra datos históricos
- **Modo forecast**: Predecir consumo futuro con variables exógenas

#### Modos de Predicción

**Backtest (Validación Histórica)**:
- Evalúa el modelo contra datos históricos conocidos
- Permite análisis "what-if" modificando variables exógenas
- Calcula métricas de error (MAE, RMSE, MAPE, sMAPE, WAPE)

**Forecast (Predicción Futura)**:
- Predice consumo futuro más allá de los datos históricos
- Requiere valores de variables exógenas (constantes o CSV)
- Genera predicciones con intervalos de confianza

### 2. Script CLI (main.py)

Para inferencia y evaluación desde línea de comandos:

```bash
python main.py
```

**Argumentos opcionales**:
```bash
python main.py \
  --csv "URL_AL_CSV" \
  --state_dict_url "URL_AL_MODELO" \
  --weather_as_known
```

**Salida**:
- Configuración del modelo (variables conocidas/desconocidas, cuantiles)
- Métricas de validación (MAE, RMSE, MAPE, sMAPE, WAPE)
- Ejemplo de predicción vs valor real

### 3. Uso Programático

#### Cargar Datos y Crear Datasets

```python
from core.data_manager import DataManager

# Inicializar gestor de datos
dm = DataManager(
    csv_url="URL_AL_CSV",
    prediction_length=24 * 6,        # 24 horas
    max_encoder_length=7 * 24 * 6,   # 7 días
    weather_as_known=True             # Variables climáticas como conocidas
)

# Cargar y procesar datos
df = dm.load_dataframe()
training, validation = dm.make_datasets()
train_loader, val_loader = dm.make_dataloaders()
```

#### Construir y Cargar Modelo

```python
from core.model_manager import ModelManager

# Inicializar gestor de modelo
mm = ModelManager(
    training_dataset=training,
    learning_rate=1e-3,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=32
)

# Construir arquitectura y cargar pesos
mm.build_model()
mm.load_state_dict_from_url()
```

#### Realizar Predicciones

```python
import numpy as np

# Predicción p50 (mediana)
y_pred = mm.predict_p50(val_loader)  # Shape: (n_samples, prediction_length)

# Predicción raw (todos los cuantiles)
raw = mm.predict_raw(val_loader)
predictions = raw.output[0].detach().cpu().numpy()  # Shape: (n_samples, pred_len, n_quantiles)
```

#### Calcular Métricas

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Extraer valores reales
y_true_list = []
for _, y_batch in val_loader:
    y_true_list.append(y_batch[0].detach().cpu().numpy())
y_true = np.concatenate(y_true_list, axis=0)

# Calcular métricas
mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))

# sMAPE
denom = np.abs(y_true) + np.abs(y_pred)
smape = (2.0 * np.abs(y_true - y_pred) / denom).mean() * 100

# WAPE
wape = (np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()) * 100
```

## Arquitectura del Modelo

### Temporal Fusion Transformer (TFT)

El TFT es una arquitectura de deep learning diseñada específicamente para pronóstico de series temporales que combina:

1. **Variable Selection Networks**: Selección automática de las características más relevantes
2. **LSTM Encoder-Decoder**: Captura dependencias temporales de largo plazo
3. **Multi-Head Attention**: Identifica patrones temporales importantes
4. **Gated Residual Networks**: Permite flujo de información flexible

### Hiperparámetros del Modelo

```python
LEARNING_RATE = 1e-3              # Tasa de aprendizaje
HIDDEN_SIZE = 64                  # Dimensión de capas ocultas
ATTENTION_HEADS = 4               # Número de cabezas de atención
DROPOUT = 0.1                     # Tasa de dropout
HIDDEN_CONTINUOUS_SIZE = 32       # Tamaño de embedding para variables continuas
QUANTILES = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]  # Cuantiles para predicción
```

### Variables del Modelo

**Target (Variable a Predecir)**:
- `zone_1`: Consumo de energía en la Zona 1 (kWh)

**Variables Temporales Conocidas** (disponibles en encoder y decoder):
- `time_idx`: Índice temporal (cada 10 minutos)
- `hour`: Hora del día (0-23)
- `day`: Día del mes (1-31)
- `day_of_week`: Día de la semana (0=Lunes, 6=Domingo)
- `month`: Mes del año (1-12)
- `is_weekend`: Indicador de fin de semana (0/1)
- `is_holiday`: Indicador de día festivo (0/1)
- `temperature`: Temperatura ambiente (°C)
- `humidity`: Humedad relativa (%)
- `wind_speed`: Velocidad del viento (m/s)
- `general_diffuse_flows`: Radiación solar difusa (W/m²)

**Variables Temporales Desconocidas** (solo en encoder):
- Ninguna (configuración por defecto: clima como conocido)

## Datos

### Fuente de Datos

Los datos provienen del repositorio GitHub y contienen:
- **Período**: Datos históricos de consumo de Tetuán
- **Resolución**: 10 minutos
- **Variables**: Consumo eléctrico + variables meteorológicas

### URL de Recursos

```python
RAW_DATA_URL = "https://raw.githubusercontent.com/DCajiao/Time-series-forecast-of-energy-consumption-in-Tetouan-City/main/data/enriched_zone1_power_consumption_of_tetouan_city.csv"

RAW_STATE_DICT_URL = "https://raw.githubusercontent.com/DCajiao/Time-series-forecast-of-energy-consumption-in-Tetouan-City/main/models/tft_model_state_dict.pt"
```

### Preprocesamiento

El `DataManager` realiza automáticamente:
1. **Parseo de fechas**: Conversión a `pd.Timestamp`
2. **Creación de time_idx**: Índice temporal desde el inicio (cada 10 min)
3. **Features de calendario**: Extracción de hora, día, mes, día de semana
4. **Validación**: Verificación de columnas requeridas y ausencia de NaN en target
5. **Normalización**: GroupNormalizer para escalar por grupo (zona)

## Métricas de Evaluación

El sistema calcula las siguientes métricas:

- **MAE (Mean Absolute Error)**: Error absoluto promedio
- **RMSE (Root Mean Squared Error)**: Raíz del error cuadrático medio
- **MAPE (Mean Absolute Percentage Error)**: Error porcentual absoluto medio
- **sMAPE (Symmetric MAPE)**: MAPE simétrico (0-100%)
- **WAPE (Weighted Absolute Percentage Error)**: Error porcentual absoluto ponderado

## Ejemplos de Uso

### Ejemplo 1: Predicción Simple

```python
from core.data_manager import DataManager
from core.model_manager import ModelManager

# Cargar datos
dm = DataManager()
df = dm.load_dataframe()
training, validation = dm.make_datasets()
_, val_loader = dm.make_dataloaders()

# Cargar modelo
mm = ModelManager(training_dataset=training)
mm.build_model()
mm.load_state_dict_from_url()

# Predecir
predictions = mm.predict_p50(val_loader)
print(f"Predicciones shape: {predictions.shape}")
```

### Ejemplo 2: Análisis de Cuantiles

```python
# Obtener todas las predicciones cuantílicas
raw = mm.predict_raw(val_loader)
preds = raw.output[0].detach().cpu().numpy()

# Extraer cuantiles específicos
p10 = preds[:, :, 1]  # Cuantil 0.1
p50 = preds[:, :, 3]  # Cuantil 0.5 (mediana)
p90 = preds[:, :, 5]  # Cuantil 0.9

# Visualizar banda de confianza
import matplotlib.pyplot as plt

sample_idx = 0
plt.figure(figsize=(12, 6))
plt.plot(p50[sample_idx], label='p50 (mediana)', color='blue')
plt.fill_between(range(len(p50[sample_idx])), p10[sample_idx], p90[sample_idx], 
                 alpha=0.3, label='Banda p10-p90')
plt.xlabel('Paso temporal (10 min)')
plt.ylabel('Consumo (kWh)')
plt.legend()
plt.title('Predicción con Intervalo de Confianza')
plt.show()
```

### Ejemplo 3: Backtest con Variables Modificadas

```python
# En la aplicación Streamlit, puedes:
# 1. Seleccionar "Backtest (histórico)"
# 2. Elegir fecha de inicio dentro del rango histórico
# 3. Modificar variables exógenas (temperatura, humedad, etc.)
# 4. Ver cómo cambian las predicciones con diferentes escenarios
```

## Solución de Problemas

### Error: "Faltan columnas en el dataset"

**Causa**: El CSV no contiene las columnas requeridas.

**Solución**: Verifica que el CSV tenga: `datetime`, `temperature`, `humidity`, `general_diffuse_flows`, `zone_1`

### Error: "NaN en target 'zone_1'"

**Causa**: Valores faltantes en la variable objetivo.

**Solución**: Imputa o elimina filas con NaN antes de cargar:
```python
df = df.dropna(subset=['zone_1'])
```

### Error: "Dimensiones incompatibles en state_dict"

**Causa**: Los hiperparámetros del modelo no coinciden con el checkpoint.

**Solución**: Usa los hiperparámetros exactos del entrenamiento:
```python
mm = ModelManager(
    training_dataset=training,
    hidden_size=64,           # Debe coincidir
    attention_head_size=4,    # Debe coincidir
    hidden_continuous_size=32 # Debe coincidir
)
```

### Predicciones Fuera de Rango

**Causa**: Variables exógenas con valores atípicos.

**Solución**: Verifica que los valores de temperatura, humedad, etc. estén en rangos razonables.

## Contribución

Las contribuciones son bienvenidas. Para contribuir:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añade nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Contacto

Para preguntas, sugerencias o reportar problemas:
- Abre un issue en GitHub
- Contacta al mantenedor del proyecto

## Referencias

- **PyTorch Forecasting**: https://pytorch-forecasting.readthedocs.io/
- **Temporal Fusion Transformer Paper**: Lim et al. (2021) - "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
- **Dataset Original**: Power consumption of Tetouan city (UCI Machine Learning Repository)

---

**Última actualización**: 2025-10-07