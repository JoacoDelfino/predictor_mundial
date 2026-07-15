# Predictor Mundial 2026 ⚽

Proyecto de Machine Learning para predecir resultados de partidos del Mundial de Fútbol 2026. Usa redes neuronales entrenadas con historial de partidos internacionales para predecir quién gana y el marcador más probable.

## ¿Qué hace?

Dado dos selecciones, el modelo predice:
- Probabilidad de que gane cada equipo o que empaten
- Marcador más probable
- Heatmap con la probabilidad de cada marcador exacto (0-0 hasta 5-5)

## Dataset

Se usa el dataset [International Football Results 1872-2024](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) de Kaggle, que incluye más de 49.000 partidos internacionales con resultado, fecha, sede y torneo.

Archivos del dataset:
- `results.csv` — resultados de todos los partidos (el más importante)
- `shootouts.csv` — resultados de penales
- `goalscorers.csv` — goleadores
- `former_names.csv` — nombres históricos de selecciones

## Features usadas

Por cada equipo se calculan 6 features a partir del historial previo al partido:

| Feature | Descripción |
|---|---|
| win_rate | % de victorias en los últimos 10 partidos |
| avg_goles_favor | Promedio de goles a favor en los últimos 10 partidos |
| avg_goles_contra | Promedio de goles en contra en los últimos 10 partidos |
| h2h_win_rate | % de victorias en el historial head-to-head contra el rival |
| ranking_fifa | Ranking FIFA previo al Mundial 2026 |
| mundiales | Cantidad de mundiales ganados históricamente |

Más una feature del partido: `neutral` (siempre 1 en mundiales, ya que se juegan en cancha neutral).

Total: 13 features por partido (6 × 2 equipos + 1 neutral).

## Modelos

### Modelo 1 — Clasificación (quién gana)
Red neuronal que predice el resultado: gana equipo 1 / empate / gana equipo 2.

Arquitectura:
```
Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dropout(0.3) → Dense(3, linear)
```
Loss: SparseCategoricalCrossentropy  
Accuracy en test: ~55-58%

### Modelo 2 — Regresión (marcador)
Red neuronal que predice los goles de cada equipo. Los goles esperados se usan como parámetro lambda de una distribución de Poisson para generar el heatmap de marcadores.

Arquitectura:
```
Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dense(2, linear)
```
Loss: Mean Squared Error

## Archivos del proyecto

```
Predictor Mundial/
├── predictor_mundial.py   # entrena y guarda los modelos
├── predecir.py            # carga los modelos y hace predicciones
├── results.csv            # dataset principal
├── modelo_mundial.h5      # modelo de clasificación guardado
├── modelo_goles.h5        # modelo de regresión guardado
├── normalizador.pkl       # normalizador del modelo 1
├── norm_goles.pkl         # normalizador del modelo 2
└── label_encoder.pkl      # encoder de clases (-1, 0, 1)
```

## Cómo usarlo

### 1. Instalar dependencias
```bash
pip3 install pandas numpy tensorflow scikit-learn matplotlib seaborn scipy
```

### 2. Entrenar los modelos
```bash
python3 predictor_mundial.py
```
Esto entrena ambos modelos y los guarda en disco. Solo hay que correrlo una vez.

### 3. Hacer predicciones
```bash
python3 predecir.py
```
Modificá la última línea del archivo para elegir los equipos:
```python
predecir_partido('Argentina', 'England')
predecir_partido('Spain', 'France')
```

## Ejemplo de output

```
── Argentina vs England ──
Gana England: 35.2%
Empate: 26.6%
Gana Argentina: 38.2%
Marcador más probable: Argentina 1 - 1 England
```

Además se genera un archivo `heatmap.png` con la probabilidad de cada marcador exacto.

## Limitaciones

- El modelo tiene ~55% de accuracy en test 
- El ranking FIFA es estático (previo al Mundial de 2026)
- No considera lesiones, bajas ni estado físico de los jugadores
- Solo se entrena con partidos de Copa del Mundo, no con toda la historia de cada selección

## Tecnologías

- Python 3.9
- TensorFlow / Keras
- scikit-learn
- pandas / numpy
- matplotlib / seaborn
- scipy (distribución de Poisson)