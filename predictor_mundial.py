import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Cargar los datos de entrenamiento
df = pd.read_csv('results.csv')
df['date'] = pd.to_datetime(df['date'])

# Filtrar los datos de entrenamiento sin el mundial de 2026
df_train = df[df['date'] < '2026-06-01']

ranking_fifa = {
    'France': 1,
    'Spain': 2,
    'Argentina': 3,
    'England': 4,
    'Portugal': 5,
    'Brazil': 6,
    'Netherlands': 7,
    'Morocco': 8,
    'Belgium': 9,
    'Germany': 10,
    'Croatia': 11,
    'Colombia': 13,
    'Senegal': 14,
    'Mexico': 15,
    'United States': 16,
    'Uruguay': 17,
    'Japan': 18,
    'Switzerland': 19,
    'Iran': 21,
    'Turkey': 22,
    'Ecuador': 23,
    'Austria': 24,
    'South Korea': 25,
    'Australia': 27,
    'Egypt': 28,
    'Algeria': 29,
    'Canada': 30,
    'Norway': 31,
    'Panama': 33,
    'Ivory Coast': 34,
    'Sweden': 38,
    'Paraguay': 40,
    'Czech Republic': 41,
    'Scotland': 43,
    'Tunisia': 44,
    'DR Congo': 46,
    'Uzbekistan': 50,
    'Qatar': 55,
    'Iraq': 57,
    'South Africa': 60,
    'Saudi Arabia': 61,
    'Jordan': 63,
    'Bosnia and Herzegovina': 65,
    'Cape Verde': 69,
    'Ghana': 74,
    'Curacao': 82,
    'Haiti': 83,
    'New Zealand': 85,
}

mundiales_ganados = {
    'Brazil': 5,
    'Germany': 4,
    'Italy': 4,
    'Argentina': 3,
    'France': 2,
    'Uruguay': 2,
    'England': 1,
    'Spain': 1,
}

# Devuelve los últimos n partidos de un equipo antes de una fecha dada
def get_ultimos_partidos(equipo, fecha, n=10):
    mask = (
        ((df['home_team'] == equipo) | (df['away_team'] == equipo)) &
        (df['date'] < fecha)
    )
    return df[mask].sort_values('date').tail(n)

# Calcula win rate, goles y head-to-head de un equipo contra un rival antes de una fecha
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
    
    # Head to head contra el rival específico
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

