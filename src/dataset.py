# ====================
# 1. Librerías
# ====================

import pandas as pd
import numpy as np

# ====================
# 2. Cargar el dataset
# ====================

data = pd.read_csv('../data/raw/Customer-Churn-Records.csv')

# ====================
# 3. Información Básica
# ====================

## Muestra de la base

data.head()

## Largo de la base

len(data)

## Descripción

data.describe()

## Cambiar las columnas a letras pequeñas

data.columns = data.columns.str.lower()

## ver el tipo de columnas que son

data.dtypes

data['geography'] = data['geography'].astype('category')

data['gender'] = data['gender'].astype('category')

data['numofproducts'] = data['numofproducts'].astype('category')

data['hascrcard'] = data['hascrcard'].astype('category')

data['isactivemember'] = data['isactivemember'].astype('category')

data['exited'] = data['exited'].astype('category')

data['complain'] = data['complain'].astype('category')

data['satisfaction score'] = data['satisfaction score'].astype('category')

data['card type'] = data['card type'].astype('category')

## Quitar columnas que no son necesarias para el estudio

data = data.drop(columns=['rownumber','customerid'])

# ====================
# 4. Guardar data nueva
# ====================

data.to_csv('../data/processed/df.csv',index=False)