import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

TARGET = "Tension_Liquidez_bin"
EXCLUDED = {"ID_Observacion", "Fecha", "Prob_Tension_Liquidez", TARGET}

def create_financial_features(df):
    df = df.copy()
    derived = []

    pairs = [
        ("Activo_Circulante_MXN", "Pasivo_Circulante_MXN", "Liquidez_Corriente"),
        ("Capital_Trabajo_Neto_MXN", "Ventas_12M_MXN", "Capital_Trabajo_Ventas"),
        ("Deuda_Total_MXN", "EBITDA_MXN", "Deuda_EBITDA"),
        ("Caja_MXN", "Deuda_CP_MXN", "Caja_Deuda_CP"),
        ("CxC_MXN", "Ventas_12M_MXN", "CxC_Ventas"),
        ("Inventarios_MXN", "Ventas_12M_MXN", "Inventario_Ventas"),
        ("Deuda_CP_MXN", "Ventas_12M_MXN", "Deuda_CP_Ventas"),
        ("Linea_Credito_Disponible_MXN", "Deuda_CP_MXN", "Cobertura_Credito_CP"),
        ("Brecha_Caja_90d_MXN", "Ventas_12M_MXN", "Brecha_Caja_Ventas"),
        ("EBITDA_MXN", "Ventas_12M_MXN", "Margen_EBITDA_Derivado"),
    ]
    for num, den, name in pairs:
        if num in df.columns and den in df.columns:
            denominator = df[den].replace(0, np.nan)
            df[name] = df[num] / denominator
            derived.append(name)

    if all(c in df.columns for c in ["DSO_dias", "DIO_dias", "DPO_dias"]):
        df["CCC_Derivado"] = df["DSO_dias"] + df["DIO_dias"] - df["DPO_dias"]
        derived.append("CCC_Derivado")
    return df, derived

def prepare_data(df):
    if TARGET not in df.columns:
        raise ValueError(f"No existe la variable objetivo '{TARGET}'.")
    df, derived = create_financial_features(df)
    before = len(df)
    df = df.dropna(axis=0).reset_index(drop=True)
    removed = before - len(df)

    y = pd.to_numeric(df[TARGET], errors="coerce")
    mask = y.notna()
    df = df.loc[mask].copy()
    y = y.loc[mask].astype(int)

    cols = [c for c in df.columns if c not in EXCLUDED]
    X = df[cols].copy()

    # Solo variables numéricas + Sector como categórica.
    for c in list(X.columns):
        if X[c].dtype == "object" and c != "Sector":
            X.drop(columns=[c], inplace=True)
    X = X.dropna(axis=1, how="all")

    return X, y, {"rows_removed": removed, "derived_features": derived}

def preprocess(X_train, X_test):
    categorical = [c for c in X_train.columns if X_train[c].dtype == "object"]
    numeric = [c for c in X_train.columns if c not in categorical]

    transformer = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    train_t = transformer.fit_transform(X_train)
    test_t = transformer.transform(X_test)
    names = list(transformer.get_feature_names_out())
    return transformer, train_t, test_t, names

def build_model(max_features):
    mf = None if max_features == "None" else max_features
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        criterion="gini",
        class_weight="balanced",
        max_features=mf,
        random_state=42,
        n_jobs=-1,
    )

def excel_bytes(predictions, importance, metrics, cv_scores, config):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        predictions.to_excel(writer, sheet_name="Predicciones", index=False)
        importance.to_excel(writer, sheet_name="Importancia_Variables", index=False)
        pd.DataFrame({
            "Metrica": ["Precision", "Accuracy", "Recall", "F1", "ROC-AUC",
                        "Precision_CV_Promedio", "Precision_CV_DesvStd"],
            "Valor": [metrics["precision"], metrics["accuracy"], metrics["recall"],
                      metrics["f1"], metrics["roc_auc"],
                      cv_scores.mean(), cv_scores.std()]
        }).to_excel(writer, sheet_name="Metricas", index=False)
        config.to_excel(writer, sheet_name="Configuracion", index=False)
    output.seek(0)
    return output.getvalue()

st.set_page_config(page_title="Random Forest — Tensión de Liquidez", layout="wide")
st.title("🌲 Random Forest — Predicción de Tensión de Liquidez")
st.caption("Clasificación binaria mediante Random Forest sobre variables financieras y de capital de trabajo.")

st.sidebar.header("Configuración")
uploaded = st.sidebar.file_uploader("Carga la base Excel", type=["xlsx", "xls"])
max_features = st.sidebar.selectbox("max_features", ["sqrt", "log2", "None"])
st.sidebar.write("**500 árboles | max_depth=10 | Gini | balanced**")
st.sidebar.write("**Train/Test = 80/20 | CV = 5-fold | Umbral = 0.50**")

if uploaded is None:
    st.info("Carga tu archivo Excel para ejecutar el modelo.")
    st.stop()

if st.sidebar.button("Ejecutar modelo", type="primary"):
    try:
        data = pd.read_excel(uploaded, sheet_name="Datos_Modelo")

        st.subheader("1. Base de datos")
        c1, c2, c3 = st.columns(3)
        c1.metric("Observaciones", f"{len(data):,}")
        c2.metric("Variables", f"{data.shape[1]:,}")
        c3.metric("Variable objetivo", TARGET)

        X, y, info = prepare_data(data)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )
        transformer, X_train_t, X_test_t, feature_names = preprocess(X_train, X_test)

        model = build_model(max_features)
        model.fit(X_train_t, y_train)

        prob = model.predict_proba(X_test_t)[:, 1]
        pred = (prob >= 0.50).astype(int)

        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        accuracy = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, prob)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            build_model(max_features), X_train_t, y_train,
            cv=cv, scoring="precision", n_jobs=-1
        )

        metrics = {
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "roc_auc": auc
        }

        importance = pd.DataFrame({
            "Variable": feature_names,
            "Importancia": model.feature_importances_
        }).sort_values("Importancia", ascending=False).reset_index(drop=True)

        predictions = pd.DataFrame({
            "Indice_Observacion_Test": X_test.index,
            "Probabilidad_Tension": prob,
            "Prediccion_Tension": pred,
            "Clasificacion": np.where(pred == 1, "Con tensión", "Sin tensión"),
            "Valor_Real": y_test.values,
        })

        st.subheader("2. Desempeño")
        a,b,c,d,e = st.columns(5)
        a.metric("Precision", f"{precision:.2%}")
        b.metric("Accuracy", f"{accuracy:.2%}")
        c.metric("Recall", f"{recall:.2%}")
        d.metric("F1", f"{f1:.2%}")
        e.metric("ROC-AUC", f"{auc:.2%}")

        a,b = st.columns(2)
        with a:
            st.markdown("**Matriz de confusión**")
            st.dataframe(
                pd.DataFrame(
                    confusion_matrix(y_test, pred),
                    index=["Real 0", "Real 1"],
                    columns=["Pred. 0", "Pred. 1"]
                ),
                use_container_width=True
            )
        with b:
            fpr,tpr,_ = roc_curve(y_test, prob)
            roc_df = pd.DataFrame({"FPR":fpr,"TPR":tpr})
            fig = px.line(roc_df, x="FPR", y="TPR", title="Curva ROC")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("3. Validación cruzada")
        a,b = st.columns(2)
        a.metric("Precision CV promedio", f"{cv_scores.mean():.2%}")
        b.metric("Desviación estándar", f"{cv_scores.std():.2%}")

        st.subheader("4. Importancia de variables")
        top_n = st.slider("Variables a mostrar", 5, min(30, len(importance)), min(15, len(importance)))
        fig = px.bar(
            importance.head(top_n).sort_values("Importancia"),
            x="Importancia", y="Variable", orientation="h",
            title="Importancia de variables"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("5. Predicciones")
        st.dataframe(predictions, use_container_width=True)

        config = pd.DataFrame({
            "Parametro": [
                "Modelo","Objetivo","n_estimators","max_depth","criterion",
                "class_weight","max_features","Train/Test","Cross-Validation",
                "Umbral","Metrica principal","Valores faltantes","Outliers"
            ],
            "Configuracion": [
                "RandomForestClassifier",TARGET,"500","10","gini","balanced",
                max_features,"80% / 20%","5-fold","0.50","Precision",
                "Eliminar registros","Sin tratamiento"
            ]
        })
        st.subheader("6. Configuración")
        st.dataframe(config, use_container_width=True)

        st.download_button(
            "⬇️ DESCARGAR RESULTADOS EN EXCEL",
            data=excel_bytes(predictions, importance, metrics, cv_scores, config),
            file_name="random_forest_tension_liquidez_resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as exc:
        st.error("Se produjo un error al ejecutar el modelo.")
        st.exception(exc)
else:
    st.success("Base cargada. Pulsa «Ejecutar modelo».")

