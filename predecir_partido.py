# Predice el resultado y marcador de un partido con heatmap de probabilidades
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

modelo = tf.keras.models.load_model('modelo_mundial.h5')
modelo_goles = tf.keras.models.load_model('modelo_goles.h5')

with open('normalizador.pkl', 'rb') as f:
    normalizador = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('norm_goles.pkl', 'rb') as f:
    norm_goles = pickle.load(f)

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

# Devuelve los últimos partidos antes de una fecha específica para un equipo dado
def get_ultimos_partidos(equipo, fecha, n=10):
    mask = (
        ((df['home_team'] == equipo) | (df['away_team'] == equipo)) &
        (df['date'] < fecha)
    )
    return df[mask].sort_values('date').tail(n)

# Calcula las estadísticas históricas de un equipo contra un rival específico
def calcular_features(equipo, rival, fecha):
    partidos = get_ultimos_partidos(equipo, fecha)
    wins = 0
    goles_favor = 0
    goles_contra = 0
    for _, partido in partidos.iterrows():
        if partido['home_team'] == equipo:
            goles_favor += partido['home_score']
            goles_contra += partido['away_score']
            if partido['home_score'] > partido['away_score']:
                wins += 1
        else:
            goles_favor += partido['away_score']
            goles_contra += partido['home_score']
            if partido['away_score'] > partido['home_score']:
                wins += 1
    n = len(partidos)
    win_rate = wins / n if n > 0 else 0
    avg_goles_favor = goles_favor / n if n > 0 else 0
    avg_goles_contra = goles_contra / n if n > 0 else 0

    # Historial de partidos entre los dos equipos específicamente
    h2h = df[
        ((df['home_team'] == equipo) & (df['away_team'] == rival)) |
        ((df['home_team'] == rival) & (df['away_team'] == equipo))
    ]
    h2h = h2h[h2h['date'] < fecha]
    h2h_wins = 0
    for _, partido in h2h.iterrows():
        if partido['home_team'] == equipo and partido['home_score'] > partido['away_score']:
            h2h_wins += 1
        elif partido['away_team'] == equipo and partido['away_score'] > partido['home_score']:
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

# Arma el array de 13 features que espera el modelo (6 por equipo + neutral)
def armar_features(equipo1, equipo2, fecha):
    features_equipo1 = calcular_features(equipo1, equipo2, fecha)
    features_equipo2 = calcular_features(equipo2, equipo1, fecha)
    return np.array([[
        features_equipo1['win_rate'], features_equipo1['avg_goles_favor'], features_equipo1['avg_goles_contra'],
        features_equipo1['h2h_win_rate'], features_equipo1['ranking_fifa'], features_equipo1['mundiales'],
        features_equipo2['win_rate'], features_equipo2['avg_goles_favor'], features_equipo2['avg_goles_contra'],
        features_equipo2['h2h_win_rate'], features_equipo2['ranking_fifa'], features_equipo2['mundiales'],
        1  # neutral = True, todos los partidos de mundial son en cancha neutral
    ]])

def predecir_partido(equipo1, equipo2, fecha='2026-07-15'):
    fecha = pd.Timestamp(fecha)
    x = armar_features(equipo1, equipo2, fecha)

    # Modelo 1: probabilidades de ganar/empatar/perder
    pred = modelo.predict(normalizador(x), verbose=0)
    probabilidades = tf.nn.softmax(pred[0]).numpy()  # pasa a probabilidades lo cálculado por el modelo
    clases = le.inverse_transform([0, 1, 2])  # devuelve a -1,0,1 porq entrenamiento no permite -1

    print(f"\n── {equipo1} vs {equipo2} ──")
    for clase, prob in zip(clases, probabilidades):
        if clase == -1:
            print(f"Gana {equipo2}: {prob*100:.1f}%")
        elif clase == 1:
            print(f"Gana {equipo1}: {prob*100:.1f}%")
        else:
            print(f"Empate: {prob*100:.1f}%")

    # Modelo 2: goles esperados por cada equipo
    pred_goles = modelo_goles.predict(norm_goles(x), verbose=0)[0]
    goles_equipo1 = max(0, round(pred_goles[0]))
    goles_equipo2 = max(0, round(pred_goles[1]))
    print(f"Marcador más probable: {equipo1} {goles_equipo1} - {goles_equipo2} {equipo2}")

    # Heatmap: usa Poisson para calcular la probabilidad de cada marcador exacto
    # lambda es el promedio de goles esperados, Poisson distribuye las probabilidades alrededor de ese valor
    max_goles = 5
    matriz = np.zeros((max_goles + 1, max_goles + 1))
    lambda1 = max(0.1, pred_goles[0])
    lambda2 = max(0.1, pred_goles[1])

    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            matriz[i][j] = poisson.pmf(i, lambda1) * poisson.pmf(j, lambda2)

    matriz = matriz / matriz.sum() * 100  ß

    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=range(max_goles + 1),
                yticklabels=range(max_goles + 1))
    plt.xlabel(f'Goles {equipo2}')
    plt.ylabel(f'Goles {equipo1}')
    plt.title(f'{equipo1} vs {equipo2}')
    plt.tight_layout()
    plt.savefig('heatmap.png')
    plt.show()

predecir_partido('Argentina', 'England')