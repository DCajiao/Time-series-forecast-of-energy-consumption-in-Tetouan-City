# Análisis técnico de métricas por modelo

## ARIMA

* **sMAPE (17% one-step)**: refleja que el modelo **no captura la forma intradía**. En consumo energético, esto implica fallar en identificar **picos de demanda** (ej. encendido masivo de aires acondicionados al mediodía).
* **WAPE (16.8%)**: el error agregado es crítico → en planeación energética significa subestimar o sobreestimar en un 17% el total de energía, lo cual puede traducirse en **miles de kWh mal planificados**.
* **Tiempo de entrenamiento (bajo)**: lo vuelve atractivo como baseline, pero irrelevante en operación real.
* **Insight**: ARIMA no sirve para gestión de red, porque ante cambios climáticos o estacionales **responde con un promedio plano** → genera déficit o sobrecarga en la red.

---

## SARIMAX

* **sMAPE (3.8% one-step, 1.4% 3h)**: la combinación de estacionalidad diaria (s=144) + exógenas logra **alinearse con ciclos reales de consumo** y absorber shocks climáticos.
* **WAPE (4% one-step, 1.4% 3h)**: muy bajo error acumulado → permite planificar generación y compras de energía con precisión. En un operador eléctrico, un WAPE < 5% puede significar **decenas de millones en ahorros** al evitar compras de emergencia en mercado spot.
* **Tiempo (\~1097 s)**: pesado; si se necesita reentrenar varias zonas en tiempo real no es viable.
* **Insight**: es un modelo **operativamente estable** para 1–3 h, ideal para toma de decisiones tácticas (ej. despachar turbinas de respaldo).

---

## Random Forest

* **sMAPE (7.2% one-step, 2.5% 3h)**: tiende a **suavizar la serie** → pierde precisión en picos inmediatos, pero en horizontes más largos genera curvas menos volátiles.
* **WAPE (6.6% one-step, 2.6% 3h)**: aceptable en promedio, pero no confiable para gestionar picos de consumo.
* **Tiempo (\~43 s)**: razonable, aunque menos eficiente que boosting.
* **Insight**: RF puede servir para escenarios donde se prefiera **robustez promedio** frente a picos, como planificación en bloques horarios, pero no es apto para **respuesta rápida de la red**.

---

## XGBoost

* **sMAPE (1.5% one-step, 4.1% 3h)**: sobresaliente en corto plazo, pero degrada en predicción recursiva → refleja su dependencia en condiciones inmediatas.
* **WAPE (1.5% one-step, 4% 3h)**: excelente error agregado inmediato; en 3h se vuelve comparable a SARIMAX.
* **Tiempo (\~8.4 s)**: muy eficiente, ideal para **sistemas que requieren recalibración continua**.
* **Insight**: XGBoost es perfecto para **monitoreo en tiempo real**: anticipar consumo a 10 min permite **ajustar voltajes, programar intercambios entre subestaciones y evitar micro-apagones**.

---

## LightGBM

* **sMAPE (3.6% one-step, 4.6% 3h)**: comportamiento intermedio; ni sobresaliente ni desastroso.
* **WAPE (3.1% one-step, 4.5% 3h)**: desempeño decente, pero sin ventaja clara.
* **Tiempo (\~72.7 s)**: más lento que XGB para resultados inferiores.
* **Insight**: LightGBM no aporta ventajas diferenciales → sería redundante frente a XGB o CatBoost.

---

## CatBoost

* **sMAPE (1.22% one-step, 5.8% 3h)**: el **mejor en predicción inmediata**. Captura no linealidades y picos de consumo de forma precisa.
* **WAPE (1.17% one-step, 5.5% 3h)**: mínimo error acumulado inmediato. Pero en 3h se degrada mucho.
* **Tiempo (\~27.9 s, GPU)**: razonable; más rápido que LSTM y más explicable.
* **Insight**: CatBoost es excelente para **anticipar picos de consumo inminentes**. Esto es vital para evitar **sobreuso de plantas de emergencia** o caídas de tensión en micro-redes.

---

## LSTM

* **sMAPE (1.08–1.27% one-step, 4.9% 3h)**: comparable a CatBoost en precisión inmediata; mejor que boosting en degradación 3h.
* **WAPE (1.05–1.23% one-step, 4.5% 3h)**: muestra que el modelo **aprendió la dinámica temporal interna**, no solo correlaciones puntuales.
* **Tiempo (\~401 s)**: alto; requiere infraestructura robusta para reentrenar.
* **Insight**: LSTM es la mejor opción si se quiere **modelar simultáneamente varias zonas** o integrar señales climáticas futuras → permite capturar dependencias largas (p. ej., efectos de calor acumulado en días consecutivos).

---

# Negociación de modelos (trade-offs en contexto energético)

* **Si quieres máxima precisión inmediata (10 min):**
  → Usa **CatBoost o LSTM**. Son los únicos capaces de captar con precisión los picos bruscos de demanda, clave para evitar fallos en la red en **tiempo real**.

* **Si buscas estabilidad en horizontes de operación (1–3 h):**
  → Usa **SARIMAX** (más estable) o **XGBoost** (más rápido). Perfectos para **despacho de generación y compras en mercado eléctrico**.

* **Si priorizas velocidad y costo computacional:**
  → Usa **XGBoost**. Entrena en segundos y ofrece precisión competitiva. Ideal para **sistemas de control en streaming** que recalibran constantemente.

* **Si quieres explicabilidad y robustez promedio:**
  → Usa **SARIMAX** (lineal, interpretable, con impacto claro de exógenas). Útil cuando se requiere **auditar las decisiones del modelo**.

* **Si buscas robustez “genérica” pero sin explotar al máximo:**
  → RF y LightGBM son opciones intermedias, aunque no ofrecen ventajas frente a los demás.

---

# Insight final (valor en energía)

La elección del modelo no es solo un asunto técnico: **impacta directamente en la confiabilidad del sistema eléctrico y en los costos de operación**.

* Un error de 5% en WAPE puede significar **sobrecargar una planta térmica innecesariamente**, con pérdidas de **miles de dólares por hora**.
* Anticipar un pico con 10 min de precisión permite **activar baterías, redirigir flujo en subestaciones o negociar energía en el mercado intradía**.
* Un modelo robusto a 3h asegura que los **operadores puedan planificar compras de energía** sin tener que recurrir al mercado spot, que suele ser hasta **20–30% más caro**.