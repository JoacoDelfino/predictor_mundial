import pandas as pd
import numpy as np

# Cargar los datos de entrenamiento
df = pd.read_csv('results.csv')
df['date'] = pd.to_datetime(df['date'])

# Filtrar los datos de entrenamiento sin el mundial de 2026
df_train = df[df['date'] < '2026-06-01']

print(df_train.shape)
print(df_train.tail())