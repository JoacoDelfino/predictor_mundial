# Simula el Mundial 2026 completo desde fase de grupos hasta la final
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from scipy.stats import poisson
from collections import Counter

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

grupos = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curacao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Tunisia', 'Sweden'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Norway', 'Iraq'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'Uzbekistan', 'Colombia', 'DR Congo'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
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

def armar_features(equipo1, equipo2, fecha):
    features_equipo1 = calcular_features(equipo1, equipo2, fecha)
    features_equipo2 = calcular_features(equipo2, equipo1, fecha)
    return np.array([[
        features_equipo1['win_rate'], features_equipo1['avg_goles_favor'], features_equipo1['avg_goles_contra'],
        features_equipo1['h2h_win_rate'], features_equipo1['ranking_fifa'], features_equipo1['mundiales'],
        features_equipo2['win_rate'], features_equipo2['avg_goles_favor'], features_equipo2['avg_goles_contra'],
        features_equipo2['h2h_win_rate'], features_equipo2['ranking_fifa'], features_equipo2['mundiales'],
        1
    ]])

def simular_partido(equipo1, equipo2, fecha='2026-06-15'):
    fecha = pd.Timestamp(fecha)
    x = armar_features(equipo1, equipo2, fecha)
    pred_goles = modelo_goles.predict(norm_goles(x), verbose=0)[0]
    lambda1 = max(0.1, pred_goles[0])
    lambda2 = max(0.1, pred_goles[1])
    goles1 = np.random.poisson(lambda1)
    goles2 = np.random.poisson(lambda2)
    return goles1, goles2

# En eliminatorias si hay empate simula tiempo extra y penales con win rate
def resolver_empate(equipo1, equipo2, g1, g2):
    fecha = pd.Timestamp('2026-06-15')
    feat1 = calcular_features(equipo1, equipo2, fecha)
    feat2 = calcular_features(equipo2, equipo1, fecha)
    lambda_et1 = max(0.05, feat1['win_rate'] * 0.5)
    lambda_et2 = max(0.05, feat2['win_rate'] * 0.5)
    g1 += np.random.poisson(lambda_et1)
    g2 += np.random.poisson(lambda_et2)
    if g1 == g2:
        prob_equipo1 = feat1['win_rate'] / (feat1['win_rate'] + feat2['win_rate'] + 0.001)
        if np.random.random() < prob_equipo1:
            g1 += 1
        else:
            g2 += 1
    return g1, g2

def simular_grupos():
    clasificados = {}
    for nombre_grupo, equipos in grupos.items():
        tabla = {equipo: {'pts': 0, 'gf': 0, 'gc': 0} for equipo in equipos}
        for i in range(len(equipos)):
            for j in range(i + 1, len(equipos)):
                equipo1 = equipos[i]
                equipo2 = equipos[j]
                g1, g2 = simular_partido(equipo1, equipo2)
                tabla[equipo1]['gf'] += g1
                tabla[equipo1]['gc'] += g2
                tabla[equipo2]['gf'] += g2
                tabla[equipo2]['gc'] += g1
                if g1 > g2:
                    tabla[equipo1]['pts'] += 3
                elif g2 > g1:
                    tabla[equipo2]['pts'] += 3
                else:
                    tabla[equipo1]['pts'] += 1
                    tabla[equipo2]['pts'] += 1
        tabla_ordenada = sorted(
            tabla.items(),
            key=lambda x: (x[1]['pts'], x[1]['gf'] - x[1]['gc']),
            reverse=True
        )
        clasificados[nombre_grupo] = [tabla_ordenada[0][0], tabla_ordenada[1][0]]
    return clasificados

def simular_ronda(partidos):
    ganadores = []
    for equipo1, equipo2 in partidos:
        g1, g2 = simular_partido(equipo1, equipo2)
        if g1 == g2:
            g1, g2 = resolver_empate(equipo1, equipo2, g1, g2)
        ganador = equipo1 if g1 > g2 else equipo2
        ganadores.append(ganador)
    return ganadores

def simular_mundial_una_vez():
    clasificados = simular_grupos()
    bracket_r32 = [
        (clasificados['A'][0], clasificados['B'][1]),
        (clasificados['B'][0], clasificados['A'][1]),
        (clasificados['C'][0], clasificados['D'][1]),
        (clasificados['D'][0], clasificados['C'][1]),
        (clasificados['E'][0], clasificados['F'][1]),
        (clasificados['F'][0], clasificados['E'][1]),
        (clasificados['G'][0], clasificados['H'][1]),
        (clasificados['H'][0], clasificados['G'][1]),
        (clasificados['I'][0], clasificados['J'][1]),
        (clasificados['J'][0], clasificados['I'][1]),
        (clasificados['K'][0], clasificados['L'][1]),
        (clasificados['L'][0], clasificados['K'][1]),
        (clasificados['A'][0], clasificados['C'][1]),
        (clasificados['B'][0], clasificados['D'][1]),
        (clasificados['E'][0], clasificados['G'][1]),
        (clasificados['F'][0], clasificados['H'][1]),
    ]
    r32 = simular_ronda(bracket_r32)
    r16_partidos = [(r32[i], r32[i+1]) for i in range(0, len(r32), 2)]
    r16 = simular_ronda(r16_partidos)
    qf_partidos = [(r16[i], r16[i+1]) for i in range(0, len(r16), 2)]
    qf = simular_ronda(qf_partidos)
    sf_partidos = [(qf[i], qf[i+1]) for i in range(0, len(qf), 2)]
    sf = simular_ronda(sf_partidos)
    final = simular_ronda([(sf[0], sf[1])])
    return final[0]

N = 300
campeones = Counter()

print("Simulando 300 mundiales...")
for i in range(N):
    campeon = simular_mundial_una_vez()
    campeones[campeon] += 1
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/300 simulaciones completadas")

print("\n🏆 PROBABILIDAD DE SER CAMPEÓN DEL MUNDO 🏆")
print("-" * 40)
for equipo, victorias in campeones.most_common():
    print(f"{equipo}: {victorias/N*100:.1f}%")