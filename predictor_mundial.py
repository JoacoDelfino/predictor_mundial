import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

df = pd.read_csv('results.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] < '2026-06-01'].copy()

ranking_fifa = {
    'France': 1, 'Spain': 2, 'Argentina': 3, 'England': 4, 'Portugal': 5,
    'Brazil': 6, 'Netherlands': 7, 'Morocco': 8, 'Belgium': 9, 'Germany': 10,
    'Croatia': 11, 'Colombia': 13, 'Senegal': 14, 'Mexico': 15, 'United States': 16,
    'Uruguay': 17, 'Japan': 18, 'Switzerland': 19, 'Iran': 21, 'Turkey': 22,
    'Ecuador': 23, 'Austria': 24, 'South Korea': 25, 'Australia': 27, 'Egypt': 28,
    'Algeria': 29, 'Canada': 30, 'Norway': 31, 'Panama': 33, 'Ivory Coast': 34,
    'Sweden': 38, 'Paraguay': 40, 'Czech Republic': 41, 'Scotland': 43, 'Tunisia': 44,
    'DR Congo': 46, 'Uzbekistan': 50, 'Qatar': 55, 'Iraq': 57, 'South Africa': 60,
    'Saudi Arabia': 61, 'Jordan': 63, 'Bosnia and Herzegovina': 65, 'Cape Verde': 69,
    'Ghana': 74, 'Curacao': 82, 'Haiti': 83, 'New Zealand': 85,
}

mundiales_ganados = {
    'Brazil': 5, 'Germany': 4, 'Italy': 4, 'Argentina': 3,
    'France': 2, 'Uruguay': 2, 'England': 1, 'Spain': 1,
}

def get_ultimos_partidos(equipo, fecha, n=10):
    mask = (
        ((df['home_team'] == equipo) | (df['away_team'] == equipo)) &
        (df['date'] < fecha)
    )
    return df[mask].sort_values('date').tail(n)

def calcular_features(equipo, rival, fecha):
    partidos = get_ultimos_partidos(equipo, fecha)
    wins = 0
    goles_favor = 0
    goles_contra = 0
    for _, p in partidos.iterrows():
        if p['home_team'] == equipo:
            goles_favor += p['home_score']
            goles_contra += p['away_score']
            if p['home_score'] > p['away_score']:
                wins += 1
        else:
            goles_favor += p['away_score']
            goles_contra += p['home_score']
            if p['away_score'] > p['home_score']:
                wins += 1
    n = len(partidos)
    win_rate = wins / n if n > 0 else 0
    avg_goles_favor = goles_favor / n if n > 0 else 0
    avg_goles_contra = goles_contra / n if n > 0 else 0
    h2h = df[
        ((df['home_team'] == equipo) & (df['away_team'] == rival)) |
        ((df['home_team'] == rival) & (df['away_team'] == equipo))
    ]
    h2h = h2h[h2h['date'] < fecha]
    h2h_wins = 0
    for _, p in h2h.iterrows():
        if p['home_team'] == equipo and p['home_score'] > p['away_score']:
            h2h_wins += 1
        elif p['away_team'] == equipo and p['away_score'] > p['home_score']:
            h2h_wins += 1
    h2h_win_rate = h2h_wins / len(h2h) if len(h2h) > 0 else 0.5
    return {
        'win_rate': win_rate,
        'avg_goles_favor': avg_goles_favor,
        'avg_goles_contra': avg_goles_contra,
        'h2h_win_rate': h2h_win_rate,
        'ranking_fifa': ranking_fifa.get(equipo, 50),
        'mundiales': mundiales_ganados.get(equipo, 0),
    }

df_wc = df[df['tournament'] == 'FIFA World Cup'].copy()

# ── MODELO 1: clasificacion 
rows = []
for _, partido in df_wc.iterrows():
    home = partido['home_team']
    away = partido['away_team']
    fecha = partido['date']
    feat_home = calcular_features(home, away, fecha)
    feat_away = calcular_features(away, home, fecha)
    if partido['home_score'] > partido['away_score']:
        resultado = 1
    elif partido['home_score'] < partido['away_score']:
        resultado = -1
    else:
        resultado = 0
    rows.append({
        'home_win_rate': feat_home['win_rate'],
        'home_avg_goles_favor': feat_home['avg_goles_favor'],
        'home_avg_goles_contra': feat_home['avg_goles_contra'],
        'home_h2h_win_rate': feat_home['h2h_win_rate'],
        'home_ranking_fifa': feat_home['ranking_fifa'],
        'home_mundiales': feat_home['mundiales'],
        'away_win_rate': feat_away['win_rate'],
        'away_avg_goles_favor': feat_away['avg_goles_favor'],
        'away_avg_goles_contra': feat_away['avg_goles_contra'],
        'away_h2h_win_rate': feat_away['h2h_win_rate'],
        'away_ranking_fifa': feat_away['ranking_fifa'],
        'away_mundiales': feat_away['mundiales'],
        'neutral': 1 if partido['neutral'] else 0,
        'resultado': resultado
    })

df_model = pd.DataFrame(rows)
X = df_model.drop('resultado', axis=1).values
Y = df_model['resultado'].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

le = LabelEncoder()
le.fit(Y)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

normalizador = tf.keras.layers.Normalization(axis=-1)
normalizador.adapt(x_train)
x_trainNorm = normalizador(x_train)
x_testNorm = normalizador(x_test)

modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='linear')
])
modelo.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer=Adam(0.001), metrics=['accuracy'])
modelo.fit(x_trainNorm, y_train, epochs=100, verbose=0)
modelo.save('modelo_mundial.h5')

with open('normalizador.pkl', 'wb') as f:
    pickle.dump(normalizador, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

loss_train, acc_train = modelo.evaluate(x_trainNorm, y_train, verbose=0)
loss_test, acc_test = modelo.evaluate(x_testNorm, y_test, verbose=0)

print(f"Train - Loss: {loss_train:.4f} - Accuracy: {acc_train*100:.1f}%")
print(f"Test  - Loss: {loss_test:.4f} - Accuracy: {acc_test*100:.1f}%")

# ── MODELO 2: regresion ß
rows_goles = []
for _, partido in df_wc.iterrows():
    home = partido['home_team']
    away = partido['away_team']
    fecha = partido['date']
    feat_home = calcular_features(home, away, fecha)
    feat_away = calcular_features(away, home, fecha)
    rows_goles.append({
        'home_win_rate': feat_home['win_rate'],
        'home_avg_goles_favor': feat_home['avg_goles_favor'],
        'home_avg_goles_contra': feat_home['avg_goles_contra'],
        'home_h2h_win_rate': feat_home['h2h_win_rate'],
        'home_ranking_fifa': feat_home['ranking_fifa'],
        'home_mundiales': feat_home['mundiales'],
        'away_win_rate': feat_away['win_rate'],
        'away_avg_goles_favor': feat_away['avg_goles_favor'],
        'away_avg_goles_contra': feat_away['avg_goles_contra'],
        'away_h2h_win_rate': feat_away['h2h_win_rate'],
        'away_ranking_fifa': feat_away['ranking_fifa'],
        'away_mundiales': feat_away['mundiales'],
        'neutral': 1 if partido['neutral'] else 0,
        'goles_home': partido['home_score'],
        'goles_away': partido['away_score'],
    })

df_goles = pd.DataFrame(rows_goles)
X_g = df_goles.drop(['goles_home', 'goles_away'], axis=1).values
Y_ambos = df_goles[['goles_home', 'goles_away']].values

x_train_g, x_test_g, y_train_g, y_test_g = train_test_split(X_g, Y_ambos, test_size=0.2, random_state=42)

norm_goles = tf.keras.layers.Normalization(axis=-1)
norm_goles.adapt(x_train_g)

modelo_goles = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2)
])
modelo_goles.compile(loss='mse', optimizer=Adam(0.001))
modelo_goles.fit(norm_goles(x_train_g), y_train_g, epochs=100, verbose=0)
modelo_goles.save('modelo_goles.h5')

with open('norm_goles.pkl', 'wb') as f:
    pickle.dump(norm_goles, f)

print("Modelos guardados.")