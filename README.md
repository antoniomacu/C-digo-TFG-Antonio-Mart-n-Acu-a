# Código del Trabajo de Fin de Grado

**Título:** Mantenimiento predictivo de equipos rotativos mediante técnicas de Machine Learning  
**Autor:** Antonio Martín Acuña  
**Grado:** Ingeniería de Tecnologías Industriales — Universidad Loyola Andalucía  
**Curso:** 2025–2026

---

## Aviso de confidencialidad

Parte del código de este TFG fue desarrollada en el contexto de una beca de colaboración con una empresa privada del sector de mantenimiento predictivo industrial. Por acuerdo de confidencialidad, el código correspondiente a la **Sección 4.2** (integración con el sistema VISIOM 2.0 y el pipeline de ingestión de datos de la empresa) no puede incluirse en este repositorio. El tribunal puede solicitar acceso supervisado a esa parte del sistema directamente al autor.

El resto del código —diseño de modelos, preprocesado, análisis exploratorio, experimentación y comparación— se incluye íntegramente a continuación.

---

## Licencias de dependencias principales

| Biblioteca | Licencia |
|------------|----------|
| PyTorch / PyTorch Lightning | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| pandas / numpy | BSD-3-Clause |
| Optuna | MIT |
| matplotlib / seaborn | BSD / BSD-3-Clause |
| click | BSD-3-Clause |
| SQLAlchemy | MIT |
| python-dotenv | BSD-3-Clause |
| uv (gestor de paquetes) | MIT |
| Ruff (linter) | MIT |

---

## Estructura del repositorio

```
codigo/
├── cap4_eda/               §4.4 — Análisis estadístico exploratorio
├── cap4_preprocesado/      §4.3 — Pipeline de extracción y calidad de datos
├── cap5_modelo_cvae/       Cap. 5 — Modelos CVAE (Nivel 1 y Nivel 2)
│   ├── nivel_1/            §5.x — CVAE condicional (detección instantánea)
│   └── nivel_2/            §5.x — VAE estándar (degradación lenta)
├── cap6_comparacion/       Cap. 6 — Comparación con modelos alternativos
└── cap7_ensemble/          Cap. 7 — Sistema ensemble de dos niveles
```

> **§4.5 — Preprocesado de señales:** el preprocesado de las series temporales para el entrenamiento de cada modelo se implementa en `cap5_modelo_cvae/nivel_1/model/preprocessing.py` y `cap5_modelo_cvae/nivel_2/model/preprocessing.py`. Ambos módulos corresponden a la Sección 4.5 de la memoria.

---

## Descripción por sección

### `cap4_eda/` — Análisis Exploratorio de Datos (§4.4)

Análisis estadístico de las señales de las bombas HTF: distribuciones por variable, correlaciones entre sensores, análisis de componentes principales (PCA) y regímenes de velocidad.

| Archivo | Descripción |
|---------|-------------|
| `Stat-analysis.ipynb` | Notebook principal de EDA: distribuciones, boxplots por bomba, heatmaps de correlación, PCA y análisis de regímenes operativos |
| `regen_plots_v2.py` | Script para regenerar las figuras del EDA con etiquetado correcto de temperaturas de rodamiento |
| `regen_plots.py` | Versión anterior del script de generación de figuras |
| `pyproject.toml` | Dependencias del entorno de análisis (uv) |

> **Nota de rutas:** las rutas absolutas en estos archivos han sido sustituidas por marcadores `<PATH_TO_EDA_DIR>`, `<PATH_TO_DATA_DIR>` y `<PATH_TO_THESIS_FIGURES>`. Deben reemplazarse por las rutas locales del evaluador.

---

### `cap4_preprocesado/` — Pipeline de Extracción y Calidad de Datos (§4.3)

Pipeline completo para la extracción de datos desde la base de datos de la planta, asignación de códigos de calidad por sensor, filtrado operativo, remuestreo temporal y generación del dataset listo para entrenamiento.

**Módulo `quality/` — Pipeline de calidad**

| Archivo | Descripción |
|---------|-------------|
| `extract_fw_pump_data_v3.py` | Extrae datos crudos de la BD, añade columna `<sensor>_quality` con códigos numéricos (0=OK, 1=ausente, 2=nulo, 3=fuera de rango, 4=caída brusca, 5=señal congelada), y detecta días con datos completamente congelados |
| `cleaner.py` | Limpieza de filas con códigos de calidad no operativos |
| `labeler.py` | Asignación de etiquetas de estado (normal/anómalo) a las ventanas temporales |
| `operational_filter.py` | Filtrado de períodos en los que la bomba está operativa (velocidad > umbral) |
| `quality_config.py` | Configuración de umbrales y parámetros de calidad por sensor |
| `resampler.py` | Remuestreo a intervalos de 5 minutos (media de valor, máximo de código de calidad) |
| `reporter.py` | Generación de informe HTML del dashboard de calidad |
| `main.py` / `__main__.py` | Punto de entrada del pipeline de calidad |

**Módulo `pipeline/` — Detector de anomalías basado en reglas**

Sistema de detección multi-detector para generar etiquetas de anomalía que actúan como ground truth débil para el entrenamiento del modelo.

| Archivo | Descripción |
|---------|-------------|
| `main.py` | Orquestador del pipeline de clasificación |
| `baseline.py` | Generación de baseline estadístico (media/std por período operativo) |
| `classifier.py` | Clasificador principal: agrega votos de detectores y asigna estado global |
| `period_detector.py` | Detección de períodos operativos (arranques, paradas, régimen estacionario) |
| `data_selector.py` | Selección de ventanas de entrenamiento y test a partir de días etiquetados |
| `loader.py` | Carga de datos en formato CSV con manejo de manifiestos |
| `system_config.py` | Carga y validación del fichero de configuración YAML |
| `report.py` | Generación de informes de clasificación por bomba y día |

**Submódulo `pipeline/custom_detectors/`**

Cada detector evalúa una hipótesis de fallo específica e indica si la observación es anómala según criterios de ingeniería:

| Detector | Hipótesis de fallo |
|----------|-------------------|
| `differential_pressure.py` | Diferencia de presión entre entrada y salida fuera de rango normal |
| `flow_anomaly.py` | Anomalía en el caudal de descarga |
| `frequency_anomaly.py` | Variación anómala de la frecuencia de alimentación |
| `hydraulic_deficit.py` | Déficit hidráulico: presión baja respecto a velocidad actual |
| `outlet_pressure.py` | Presión de salida por debajo del umbral esperado |
| `winding_temp_imbalance.py` | Desequilibrio térmico entre devanados del motor |
| `filter_condition.py` | Condición de filtro (colmatación) detectada por caída de presión diferencial |
| `ml_ensemble.py` | Detector basado en modelos ML entrenados con las etiquetas del pipeline |

**Archivos de soporte**

| Archivo | Descripción |
|---------|-------------|
| `extract_model_data_standalone.py` | Script autónomo para extraer datos de la BD PostgreSQL de planta directamente en formato CSV compatible con el modelo. Requiere `.env` con credenciales (ver `.env.example`) |
| `compare_stages.py` | Comparación de métricas entre etapas del pipeline de calidad |
| `split_daily.py` | Partición de los datos diarios en conjuntos de entrenamiento y test |
| `configs/pumps.yaml` | Configuración del sistema: columnas de sensores, umbrales de detección, parámetros de calidad y remuestreo |
| `.env.example` | Plantilla de credenciales de base de datos (las credenciales reales no se incluyen) |

> **Nota:** Los nombres de bases de datos han sido anonimizados (`plant_db`, `plant_db_raw`). El host de la BD aparece como `<DB_HOST>`.

---

### `cap5_modelo_cvae/` — Diseño y Entrenamiento de los Modelos CVAE (Cap. 5)

Implementación de los dos modelos que componen el sistema ensemble. El **Nivel 1** detecta anomalías instantáneas condicionadas a la velocidad de operación; el **Nivel 2** captura degradación lenta mediante un VAE estándar entrenado sobre ventanas largas.

#### §4.5 — Preprocesado de señales

El preprocesado de los datos de entrada es específico para cada nivel del modelo y se implementa en:

- `nivel_1/model/preprocessing.py` — normalización, ventaneo temporal corto y construcción de DataLoaders para el CVAE condicional
- `nivel_2/model/preprocessing.py` — normalización, ventaneo temporal largo y construcción de DataLoaders para el VAE estándar

Ambos módulos se corresponden con la Sección 4.5 de la memoria.

---

#### `nivel_1/` — CVAE Condicional (Nivel 1 del ensemble)

Autoencoder Variacional Condicional que predice el estado instantáneo de la bomba a partir de una ventana corta de señales, condicionado a la velocidad de operación. Incluye calibración conformal para control estadístico de la tasa de falsas alarmas.

| Archivo | Descripción |
|---------|-------------|
| `model/models.py` | Arquitectura CVAE: encoder con atención temporal sobre el histórico, decoder condicional por velocidad, función de pérdida ELBO |
| `model/preprocessing.py` | **§4.5** — Normalización por cuantiles, ventaneo deslizante corto, construcción de DataLoaders |
| `model/main.py` | Entrenamiento con PyTorch Lightning: bucle de entrenamiento, validación, early stopping y checkpointing |
| `model/inference.py` | Cálculo de la puntuación de anomalía (error de reconstrucción ponderado por canal) sobre nuevas ventanas |
| `model/conformal_calibration.py` | Calibración conformal (split conformal prediction) para garantía estadística del umbral de detección |
| `model/threshold_calibration.py` | Calibración de umbrales por cuantiles sobre el conjunto de calibración |
| `model/fine_tuning.py` | Fine-tuning del modelo pre-entrenado sobre datos recientes de la bomba objetivo |
| `model/failure_detector.py` | Umbralización de la puntuación de anomalía y generación de alarmas |
| `model/device.py` | Selección automática de dispositivo (CPU/GPU/MPS) |
| `pyproject.toml` | Dependencias del entorno (uv) |
| `scripts/validate_arch.py` | Validación de la arquitectura: comprueba dimensiones de tensores |
| `scripts/test_imports.py` | Test básico de importaciones del módulo |

---

#### `nivel_2/` — VAE Estándar (Nivel 2 del ensemble)

Autoencoder Variacional estándar entrenado sobre ventanas largas para capturar patrones de degradación lenta no visibles en ventanas cortas. Actúa como segundo nivel de detección complementario al CVAE condicional.

| Archivo | Descripción |
|---------|-------------|
| `model/models.py` | Definición del VAE estándar y variantes evaluadas durante el desarrollo (LSTM-AE, CNN-AE, USAD, Transformer-AE, Sparse-AE, Denoising-AE) |
| `model/preprocessing.py` | **§4.5** — Normalización, ventaneo temporal largo, separación train/val/test, construcción de DataLoaders |
| `model/main.py` | Entrenamiento con PyTorch Lightning |
| `model/inference.py` | Cálculo de puntuación de anomalía (error de reconstrucción) |
| `model/fine_tuning.py` | Fine-tuning sobre datos recientes |
| `model/failure_detector.py` | Umbralización y generación de alarmas |
| `model/device.py` | Selección de dispositivo |
| `recalibrate_thresholds.py` | Re-calibración de umbrales del Nivel 2 sobre nuevos datos |
| `pyproject.toml` | Dependencias del entorno (uv) |
| `tests/test_vae_models.py` | Tests unitarios de las arquitecturas de modelos |

---

### `cap6_comparacion/` — Comparación con Modelos Alternativos (Cap. 6)

Benchmark sistemático del CVAE frente a modelos clásicos de detección de anomalías y arquitecturas alternativas de deep learning, evaluados sobre los mismos datos de planta.

| Archivo | Descripción |
|---------|-------------|
| `classical_models.py` | Modelos clásicos: Isolation Forest, One-Class SVM, LOF, Z-score multivariante, PCA-based anomaly detection |
| `alternative_models.py` | Modelos de DL alternativos: LSTM-AE, CNN-AE, USAD, Transformer-AE (definidos en `cap5_modelo_cvae/nivel_2/model/models.py`) |
| `benchmark.py` | Evaluación unificada: carga modelos, ejecuta inferencia sobre el conjunto de test, calcula métricas (F1, precisión, recall, AUC-ROC) y genera tablas de resultados |
| `compare.py` | Comparación estadística de resultados: tests de significancia, tablas LaTeX y gráficas de radar |

---

### `cap7_ensemble/` — Sistema Ensemble de Dos Niveles (Cap. 7)

Integración del Nivel 1 (CVAE condicional, `cap5_modelo_cvae/nivel_1`) y el Nivel 2 (VAE estándar, `cap5_modelo_cvae/nivel_2`) en un sistema de detección ensemble con capacidad de inferencia en streaming en tiempo real.

**`model/` — Orquestación del ensemble**

| Archivo | Descripción |
|---------|-------------|
| `ensemble.py` | Lógica de fusión de los dos niveles: ponderación de puntuaciones, votación y generación de alarma global |
| `level1_detector.py` | Adaptador del Nivel 1 (CVAE condicional) para el ensemble |
| `level2_detector.py` | Adaptador del Nivel 2 (VAE de degradación lenta) para el ensemble |
| `scoring.py` | Normalización y combinación de puntuaciones de anomalía inter-nivel |
| `streaming.py` | Motor de inferencia en streaming: ventana deslizante, emite puntuación cada 5 minutos sin acceso al historial completo |
| `monitoring.py` | Monitorización continua: gestión de alarmas activas, persistencia de estado y lógica anti-flicker |
| `inference.py` | Punto de entrada unificado para inferencia batch y streaming |

**`demos/` — Evaluación y demostración**

| Archivo | Descripción |
|---------|-------------|
| `streaming_demo.py` | Demo de inferencia en streaming sobre datos históricos (simula llegada de datos en tiempo real) |
| `comparison_streaming_demo.py` | Demo comparativo: puntuaciones de Nivel 1 y Nivel 2 simultáneamente |
| `plotting_demo.py` | Generación de figuras de la tesis: puntuaciones de anomalía con anotación de eventos conocidos |
| `batch_evaluate.py` | Evaluación batch sobre el conjunto de test: métricas agregadas por bomba y período |
| `calibrate_thresholds.py` | Re-calibración de umbrales del ensemble sobre datos recientes |
| `report_generator.py` | Generación automática de informes PDF de rendimiento del sistema |
| `cond_reg_v1_bridge.py` | Adaptador de compatibilidad con versiones anteriores del Nivel 1 |

**`experiments/` — Experimentos de ablación y calibración**

Experimentos numerados que corresponden a decisiones de diseño descritas en el Capítulo 7.

| Archivo | Experimento |
|---------|-------------|
| `e2_mahalanobis_calibration.py` | E2: distancia de Mahalanobis como alternativa al error de reconstrucción |
| `e3_elbo_scoring.py` | E3: uso del ELBO completo (no solo la reconstrucción) como puntuación |
| `e4_benchmark.py` | E4: benchmark automatizado de todas las arquitecturas sobre el conjunto de test |
| `e6_fault_isolation.py` | E6: localización del sensor defectuoso mediante contribución por canal al error |
| `e7_fusion_weights.py` | E7: optimización de los pesos de fusión Nivel 1 / Nivel 2 |
| `e8_per_channel_maxz.py` | E8: puntuación basada en z-score máximo por canal en lugar de la media |
| `p7_seasonal_calibration.py` | P7: calibración estacional de umbrales por mes |
| `p8_joint_threshold_optimization.py` | P8: optimización conjunta del umbral de los dos niveles mediante Optuna |
| `inv1_pump3_investigation.py` | Investigación ad-hoc de la bomba 3 (alta tasa de falsas alarmas) |

**`scripts/` — Tests de integración**

| Archivo | Descripción |
|---------|-------------|
| `smoke_test.py` | Test de humo: verifica que el ensemble arranca y produce salida válida |
| `test_alarm_monitor.py` | Test del módulo de monitorización de alarmas |
| `test_channel_health.py` | Test de la función de salud por canal |
| `test_conformal_calibration.py` | Test de la calibración conformal |
| `test_seasonal_thresholds.py` | Test de los umbrales estacionales |
| `test_savgol_parity.py` | Test de paridad del suavizado Savitzky-Golay entre versiones |

---

## Cómo ejecutar el código

### Requisitos previos

- Python 3.11+  
- [`uv`](https://docs.astral.sh/uv/) para gestión de dependencias  
- Acceso a los datos de la planta (ficheros CSV en formato generado por `cap4_preprocesado/`)

### Instalación de dependencias

Cada submódulo tiene su propio `pyproject.toml`. Para instalar:

```bash
cd cap5_modelo_cvae/nivel_1   # o el submódulo que se quiera ejecutar
uv sync
```

### Orden de ejecución recomendado

1. **Extracción de datos** (requiere acceso a la BD de planta):
   ```bash
   cd cap4_preprocesado
   cp .env.example .env
   # Editar .env con credenciales reales
   python extract_model_data_standalone.py -o data/train/
   ```

2. **Pipeline de calidad:**
   ```bash
   python -m quality --start 2025-01-01 --end 2026-06-30 --source files
   ```

3. **Análisis exploratorio (§4.4):**
   ```bash
   cd cap4_eda
   jupyter notebook Stat-analysis.ipynb
   ```

4. **Entrenamiento del Nivel 1 (CVAE condicional):**
   ```bash
   cd cap5_modelo_cvae/nivel_1
   uv run python -m cond_reg_v2.model.main --data-dir <PATH_TO_DATA_DIR>
   ```

5. **Entrenamiento del Nivel 2 (VAE estándar):**
   ```bash
   cd cap5_modelo_cvae/nivel_2
   uv run python -m model.main --data-dir <PATH_TO_DATA_DIR>
   ```

6. **Evaluación comparativa (Cap. 6):**
   ```bash
   cd cap6_comparacion
   python comparison/benchmark.py
   ```

7. **Demo del ensemble en streaming (Cap. 7):**
   ```bash
   cd cap7_ensemble
   uv run python demos/streaming_demo.py --data-dir <PATH_TO_DATA_DIR>
   ```

---

*Código desarrollado íntegramente por el autor durante el período de prácticas (Sol-ution S.L., 2025–2026) y el trabajo de investigación del TFG. Los datos de planta pertenecen a la empresa y no se incluyen en este repositorio.*
