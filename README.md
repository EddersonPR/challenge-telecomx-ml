# 📡 TelecomX — Predicción de Cancelación de Clientes (Churn)
### Challenge Académico — Parte 2: Modelado Predictivo

## 🗂️ Tecnologías Utilizadas

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-green)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red)

| Librería | Uso |
|---|---|
| `pandas` | Manipulación y análisis de datos |
| `numpy` | Operaciones numéricas |
| `matplotlib` / `seaborn` | Visualizaciones |
| `scikit-learn` | Modelos, métricas, pipeline y preprocesamiento |
| `joblib` | Serialización del modelo |

---

## 🎯 Propósito del Análisis

Este proyecto desarrolla un **pipeline completo de Machine Learning** orientado a predecir qué clientes de TelecomX tienen mayor probabilidad de cancelar sus servicios (**Churn**).

El objetivo principal es anticiparse al problema de cancelación, permitiendo al equipo comercial actuar de forma proactiva sobre los clientes en riesgo antes de que abandonen el servicio.

El pipeline cubre todo el ciclo de modelado:

- Limpieza y preparación de los datos
- Análisis exploratorio (EDA) con identificación de patrones de churn
- Selección de variables mediante test estadístico (Chi-cuadrado)
- Entrenamiento y comparación de múltiples modelos de clasificación
- Evaluación con métricas orientadas al negocio (Recall, AUC-ROC)
- Serialización del modelo final como pipeline deployable para producción

---

## 📁 Estructura del Proyecto

```
challenge-telecomx-ml/
│
├── telecomx_machine_learning.ipynb       # Notebook principal con el pipeline completo
│
├── data/
│   └── datos_tratados.csv                # Dataset limpio y procesado (post EDA Parte 1)
│
├── modelos/
│   ├── pipeline_churn_telecom_*.joblib   # Pipeline serializado (preprocesamiento + modelo)
│   ├── feature_names_*.joblib            # Nombres de features post-encoding
│   └── metadata_*.json                   # Metadata del modelo (métricas, fecha, config)
│
├── visualizaciones/
│   ├── heatmap_correlacion.png           # Matriz de correlación variables numéricas
│   ├── roc_comparacion_modelos.png       # Curvas ROC de los 3 modelos
│   ├── matriz_confusion_balanced.png     # Matriz de confusión modelo final
│   └── feature_importance.png           # Importancia de variables (coeficientes)
│
└── README.md                             # Este archivo
```

---

## 🔢 Variables del Dataset

El dataset contiene **7,043 registros** y **22 variables originales**, clasificadas de la siguiente manera:

### Variables Numéricas

| Variable | Descripción | Decisión |
|---|---|---|
| `Tenure` | Meses como cliente | ✅ Retenida — alto poder predictivo |
| `ChargesMonthly` | Cargo mensual | ✅ Retenida — representativa del costo |
| `ChargesDaily` | Cargo diario | ❌ Eliminada — correlación = 1.00 con ChargesMonthly |
| `ChargesTotal` | Cargo total acumulado | ❌ Eliminada — correlación = 0.65 con ChargesMonthly |

> **Justificación:** `ChargesDaily` y `ChargesTotal` presentan multicolinealidad con `ChargesMonthly`.
> Mantener las tres variables introduciría redundancia que afecta la estabilidad de los coeficientes
> del modelo. Se conservó `ChargesMonthly` como única representante del costo del cliente.

### Variables Categóricas Retenidas (p-value < 0.05)

| Variable | Chi2 Score | Descripción |
|---|---|---|
| `Contract` | 1,115.78 | Tipo de contrato (mensual, anual, bianual) |
| `OnlineSecurity` | 147.30 | Servicio de seguridad online |
| `TechSupport` | 135.56 | Soporte técnico contratado |
| `SeniorCitizen` | 134.35 | Cliente mayor de 60 años |
| `Dependents` | 133.04 | Cliente con dependientes |
| `PaperlessBilling` | 105.68 | Facturación digital |
| `Partner` | 82.41 | Cliente con pareja |
| `PaymentMethod` | 58.49 | Método de pago |
| `OnlineBackup` | 31.22 | Backup online contratado |
| `DeviceProtection` | 20.23 | Protección de dispositivos |
| `StreamingTV` | 17.33 | Servicio de streaming TV |
| `StreamingMovies` | 16.24 | Servicio de streaming películas |
| `InternetService` | 9.82 | Tipo de internet (DSL, Fibra, No) |
| `MultipleLines` | 6.55 | Múltiples líneas telefónicas |

### Variables Eliminadas (p-value ≥ 0.05)

| Variable | p-value | Razón |
|---|---|---|
| `Gender` | 0.611 | Sin asociación estadística con Churn |
| `PhoneService` | 0.755 | Sin asociación estadística con Churn |
| `CustomerID` | — | Identificador único sin valor predictivo |

---

## ⚙️ Proceso de Preparación de los Datos

### 1. Limpieza Inicial
- Eliminación de `CustomerID` (no aporta valor predictivo)
- Unificación de `'No internet service'` → `'No'` en servicios adicionales de internet, ya que semánticamente son equivalentes

### 2. Análisis de Multicolinealidad
Se calculó la matriz de correlación entre variables numéricas. `ChargesDaily` mostró correlación perfecta (1.00) con `ChargesMonthly`, y `ChargesTotal` correlación de 0.65. Ambas fueron eliminadas para evitar redundancia y mejorar la interpretabilidad del modelo.

### 3. Selección de Variables
Se aplicó el **Test Chi-cuadrado** sobre todas las variables categóricas para medir su asociación estadística con el Churn. Se descartaron `Gender` y `PhoneService` al no superar el umbral de significancia (p < 0.05).

### 4. Análisis de Desbalance de Clases
El target presenta desbalance significativo:

```
No Churn : 5,174 registros → 73.46%
Churn    : 1,869 registros → 26.54%
```

Este desbalance fue gestionado mediante `class_weight='balanced'` en el modelo final, que pondera automáticamente las clases de forma inversamente proporcional a su frecuencia.

### 5. Partición del Dataset
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 80% entrenamiento / 20% prueba
    stratify=y,         # Mantiene la proporción de Churn en ambos conjuntos
    random_state=42
)
```
> Se usó `stratify=y` para garantizar que el desbalance original se preserve
> proporcionalmente en train y test, evitando sesgos en la evaluación.

### 6. Codificación de Variables Categóricas
Se aplicó **One-Hot Encoding** con `drop='first'` para evitar la trampa de la variable dummy (multicolinealidad perfecta entre categorías). El proceso se implementó dentro de un `ColumnTransformer` encapsulado en el `Pipeline` final, garantizando que el preprocesamiento sea parte del objeto serializado.

```python
preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', cols_numericas),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cols_categoricas)
])
```

---

## 📊 EDA — Insights Principales

### Desbalance de Clases
```
No Churn ██████████████████████████████████  73.5%
Churn    █████████████                       26.5%
```

### Tasa de Churn por Variable Clave

**Tipo de Contrato** — el predictor más poderoso:
```
Month-to-Month  ████████████████████  42.7% churn
One Year        █████                 11.3% churn
Two Year        █                      2.8% churn
```

**Tipo de Internet:**
```
Fiber Optic     ████████████████████  41.9% churn
DSL             █████████             19.0% churn
Sin internet    ███                    7.4% churn
```

**Método de Pago:**
```
Electronic Check  ██████████████████████  45.3% churn
Mailed Check      ██████████              19.1% churn
Bank Transfer     ████████               16.7% churn
Credit Card       ████████               15.2% churn
```

### Principales Hallazgos del EDA

- 📌 Clientes con **contrato mensual** tienen 15x más probabilidad de cancelar que los de contrato bianual
- 📌 El **42%** de los usuarios de Fibra Óptica cancela — la tasa más alta por tipo de servicio
- 📌 Clientes que pagan con **cheque electrónico** tienen la mayor tasa de abandono (45.3%)
- 📌 Tener **Tech Support** reduce el churn de 31% a 15% — factor protector clave
- 📌 Los **adultos mayores** (SeniorCitizen) cancelan al 41.7% vs 23.6% en clientes regulares
- 📌 Clientes **sin pareja ni dependientes** presentan mayor propensión al churn

---

## 🤖 Modelos Entrenados

| Modelo | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression (Normal) | 80.0% | 64.9% | 52.4% | 57.9% | 0.843 |
| Random Forest | 78.4% | 62.2% | 47.6% | 53.9% | 0.821 |
| **Logistic Regression (Balanced)** ⭐ | **74.3%** | **51.0%** | **78.6%** | **61.9%** | **0.843** |

### ¿Por qué se eligió la Regresión Logística Balanceada?

En un problema de churn, **no detectar a un cliente que va a cancelar tiene un costo mayor que una falsa alarma**. Por eso la métrica prioritaria es el **Recall**, no el Accuracy.

El modelo balanceado detecta correctamente al **78.6%** de los clientes que realmente cancelarán, con una estabilidad comprobada mediante validación cruzada (diferencia Train/Test < 0.006 en todas las métricas).

---

## 🔍 Importancia de Variables

**Factores que AUMENTAN el riesgo de churn:**
```
InternetService_Fiber optic    ████████████████  +1.315
StreamingTV_Yes                ██████             +0.505
PaymentMethod_Electronic check █████              +0.429
StreamingMovies_Yes            █████              +0.414
MultipleLines_Yes              █████              +0.404
```

**Factores que REDUCEN el riesgo de churn:**
```
Contract_Two year              ████████████████  -1.387
InternetService_No             ███████████████   -1.295
Contract_One year              ████████          -0.708
TechSupport_Yes                ███               -0.266
OnlineSecurity_Yes             ██                -0.181
```

---

## 🚀 Instrucciones de Ejecución

### 1. Requisitos — Instalación de Librerías

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

O con el archivo de dependencias:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
joblib>=1.2.0
```

### 2. Cargar los Datos

```python
import pandas as pd

# Cargar dataset tratado
datos = pd.read_csv('https://raw.githubusercontent.com/EddersonPR/challenge-telecomx-ml/refs/heads/main/data/datos_tratados.csv')
print(f"Dataset cargado: {datos.shape[0]} registros, {datos.shape[1]} columnas")
```

### 3. Ejecutar el Notebook

```bash
jupyter notebook telecomx_machine_learning.ipynb
```

Ejecutar las celdas en el siguiente orden:

```
1.  Carga e Inspección de Datos
2.  Preprocesamiento
3.  Análisis Exploratorio de Datos (EDA)
      3.1 Correlación de Variables Numéricas
      3.2 Tratamiento de Multicolinealidad
      3.3 Análisis de Churn por Variables Categóricas
4.  Selección de Variables — Test Chi-Cuadrado
5.  Análisis de Desbalance de Clases
6.  Partición del Dataset (Train/Test Split)
7.  Codificación de Variables Categóricas (One-Hot Encoding)
8.  Entrenamiento de Modelos
9.  Evaluación
      9.1 Obtención de Probabilidades para AUC
      9.2 Matrices de Confusión
      9.3 Comparación de Métricas
      9.4 Curvas ROC — Comparación de Modelos
10. Validación Cruzada 5-Fold
11. Importancia de Variables (Feature Importance)
12. Insights para el Negocio
13. Entrenamiento del Modelo Final (Pipeline Completo)
14. Serialización y Despliegue del Modelo
```

### 4. Usar el Modelo para Predecir Nuevos Clientes

```python
import joblib
import pandas as pd

# Opción 1:
# Cargar a Colab y pegar enlace de la carpeta donde se aloja el modelo 'pipeline'

# Opción 2:
# Guardar y almacenar modelo en repositorio local y pegar ruta o enlace de carpeta en VSC

# Cargar pipeline serializado
pipeline = joblib.load('/content/pipeline_churn_telecom_20260304_012256.joblib')

# Datos crudos de nuevos clientes (sin preprocesamiento previo)
nuevos_clientes = pd.DataFrame([{
    'Tenure': 4,
    'ChargesMonthly': 73.9,
    'SeniorCitizen': 'No',
    'Partner': 'No',
    'Dependents': 'No',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check'
}])

# Predicción directa sobre datos crudos
prediccion   = pipeline.predict(nuevos_clientes)
probabilidad = pipeline.predict_proba(nuevos_clientes)[:, 1]

print(f"Churn predicho   : {prediccion[0]}")
print(f"Probabilidad     : {probabilidad[0]:.1%}")
```

---

## 📋 Perfil del Cliente en Alto Riesgo

Un cliente con **alta probabilidad de cancelar** típicamente presenta:

| Característica | Valor de riesgo |
|---|---|
| Tipo de contrato | Month-to-Month |
| Tipo de internet | Fiber Optic |
| Método de pago | Electronic Check |
| Tech Support | No contratado |
| Online Security | No contratado |
| Tiempo como cliente | Menos de 12 meses |
| Streaming | TV y/o Películas activos |

---

## 📈 Métricas Finales del Modelo en Producción

```
Algoritmo        : Logistic Regression (class_weight='balanced')
Observaciones    : 7,043 clientes
Features         : 20 variables (post One-Hot Encoding)
─────────────────────────────────────────
AUC-ROC          : 0.843
Recall (Churn)   : 78.6%
Precision (Churn): 51.0%
F1-Score (Churn) : 61.9%
─────────────────────────────────────────
Validación       : StratifiedKFold (5 folds)
Diferencia max   : 0.006 (sin overfitting)
```

---

*Edderson Pomacanchari - AluraLatam — Data Science | Challenge TelecomX Parte 2 (ML)*
