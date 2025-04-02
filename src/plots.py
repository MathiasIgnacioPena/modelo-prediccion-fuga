# ====================
# 1. Librerías
# ====================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ====================
# 2. Cargar la base
# ====================

df = pd.read_pickle('../data/processed/df.pkl')

# ====================
# 3. Gráficos
# ====================

df.head()

## Ver la cantidad de datos que son 1 y 0 de la variable respuesta

plt.figure(figsize=(12,6))
sns.countplot(x='exited', data=df)
plt.title('Count of Exited (0 = No, 1 = Yes)')
plt.xlabel('Exited')
plt.ylabel('Count')
plt.show()

## Ver la cantidad de datos que son hombre y mujer 

plt.figure(figsize=(12,6))
sns.countplot(x='gender',data=df)
plt.title('Count of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

## Ver la cantidad de personas correspondientes a los distintos paises que hay 

plt.figure(figsize=(12,6))
sns.countplot(x='geography',data=df)
plt.title('Count of Country')
plt.xlabel('Country')
plt.ylabel('Count')
plt.show()

## Ver la cantidad de personas que son 1 y 0 en exited, por pais y genero

plt.figure(figsize=(18,6))
plt.subplot(1, 2, 1)
sns.countplot(x='exited', hue='geography', data=df)
plt.title('Count of Exited by Country')
plt.xlabel('Exited')
plt.ylabel('Count')
plt.legend(title='Country')

plt.subplot(1, 2, 2)
sns.countplot(x='exited', hue='gender', data=df)
plt.title('Count of Exited by Gender')
plt.xlabel('Exited')
plt.ylabel('Count')
plt.legend(title='Gender')

plt.tight_layout()
plt.show()

## Gráfico para poder ver la cantidad de edades que se tienen

plt.figure(figsize=(12,6))
sns.histplot(x='age',bins=30,kde=True,data=df)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

## Gráfico de caja y bigote para conocer la distribución

plt.figure(figsize=(12,6))
sns.boxplot(x='geography', y='creditscore', data=df)
plt.title('CreditScore by Geography')
plt.xlabel('Geography')
plt.ylabel('CreditScore')
plt.show()

## Ver como se comportan los puntos con el tipo de tarjeta

plt.figure(figsize=(12,6))
sns.boxplot(x='card type',y='point earned',data=df)
plt.title('Point Earned by Card Type')
plt.xlabel('Card Type')
plt.ylabel('Point Earned')
plt.show()

