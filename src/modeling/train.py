# ====================
# 1. Librerías
# ====================

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# ====================
# 2. Funciones
# ====================

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    encoded_df = pd.get_dummies(df, columns=['geography', 'gender', 'card type'])
    X = encoded_df.drop(columns=['exited', 'surname'])
    Y = encoded_df['exited']
    return X, Y

def split_data(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8, random_state=42)
    return X_train, X_test, Y_train, Y_test

# ====================
# 3. Carga y Preprocesamiento de datos
# ====================

X, Y = load_and_preprocess_data('../../data/processed/df.csv')
X_train, X_test, Y_train, Y_test = split_data(X, Y)

# Guardar los datos preprocesados
joblib.dump((X_train, X_test, Y_train, Y_test), '../../data/processed/split_data.pkl')
