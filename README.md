# Churn Prediction Model

Este proyecto tiene como objetivo predecir la deserción de clientes utilizando varios modelos de machine learning, incluyendo Regresión Logística, Random Forest.

## Estructura del Proyecto

```
churn-predict-model/
│
├── data/
│   ├── raw/
│   │   └── Customer-Churn-Records.csv
│   ├── processed/
│   │   └── df.csv
│   │   └── split_data.pkl
│
├── src/
│   ├── dataset.py
│   ├── modeling/
│   │   ├── train.py
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   ├── plots.py
│
└── README.md
```

## Descripción de Archivos

- **data/raw/Customer-Churn-Records.csv**: Archivo CSV con los datos originales de deserción de clientes.
- **data/processed/df.csv**: Archivo CSV con los datos preprocesados.
- **data/processed/split_data.pkl**: Archivo PKL con los datos divididos en conjuntos de entrenamiento y prueba.

- **src/dataset.py**: Script para cargar, preprocesar y guardar los datos.
- **src/modeling/train.py**: Script para cargar y preprocesar los datos, y dividirlos en conjuntos de entrenamiento y prueba.
- **src/modeling/logistic_regression.py**: Script para entrenar y evaluar el modelo de Regresión Logística.
- **src/modeling/random_forest.py**: Script para entrenar y evaluar el modelo de Random Forest.
- **src/plots.py**: Script para generar gráficos exploratorios de los datos.

## Dependencias

Para ejecutar este proyecto, necesitas instalar las siguientes dependencias:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Instrucciones

1. **Preprocesar los datos**:
   Ejecuta el script `dataset.py` para cargar, preprocesar y guardar los datos.

   ```bash
   python src/dataset.py
   ```

2. **Dividir los datos**:
   Ejecuta el script `train.py` para cargar y preprocesar los datos, y dividirlos en conjuntos de entrenamiento y prueba.

   ```bash
   python src/modeling/train.py
   ```

3. **Entrenar y evaluar los modelos**:
   Ejecuta los scripts correspondientes para entrenar y evaluar cada modelo.

   - Regresión Logística:

     ```bash
     python src/modeling/logistic_regression.py
     ```

   - Random Forest:

     ```bash
     python src/modeling/random_forest.py
     ```

4. **Generar gráficos exploratorios**:
   Ejecuta el script `plots.py` para generar gráficos exploratorios de los datos.

   ```bash
   python src/plots.py
   ```
