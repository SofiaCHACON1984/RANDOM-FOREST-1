# Random Forest — Tensión de Liquidez

## Corrección del error de Streamlit

Esta versión deja el `app.py` **autocontenido**. Ya no depende de:

```python
from src.model import ...
```

Por lo tanto, evita el error:

```text
ModuleNotFoundError: No module named 'src.model'
```

## Ejecución

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

El archivo principal es:

```text
app.py
```

Puedes subir como mínimo:

```text
app.py
requirements.txt
README.md
```

La carpeta `src` ya no es necesaria para ejecutar esta versión.

## Modelo

- Random Forest
- 500 árboles
- max_depth = 10
- criterion = gini
- class_weight = balanced
- max_features configurable
- Train/Test = 80/20
- 5-fold Cross Validation
- Precision como métrica principal
- Umbral = 0.50
- Sector con One-Hot Encoding
- Valores faltantes: eliminación
- Outliers: sin tratamiento
- Fecha excluida
- Prob_Tension_Liquidez excluida para evitar data leakage

## Excel

El botón de Streamlit genera:

`random_forest_tension_liquidez_resultados.xlsx`

con hojas:

- Predicciones
- Importancia_Variables
- Metricas
- Configuracion
