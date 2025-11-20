import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go

@st.cache_data
def load_and_prepare_data():
    df = pd.read_excel("proyectom.xlsx")
    
    df["eficiencia_estudio_pasado"] = df["Calificaciones pasadas"] / (df["Horas estudio pasadas "] + 1)
    df["intensidad_estudio_actual"] = df["Horas de estudio actuales "] / (df["Materias nuevas"] + 1)
    df["cambio_horas"] = df["Horas de estudio actuales "] - df["Horas estudio pasadas "]
    df["ratio_materias"] = df["Materias nuevas"] / (df["Materias pasadas "] + 1)
    df["tendencia_academica"] = df["Calificaciones pasadas"] * (df["Horas de estudio actuales "] / (df["Horas estudio pasadas "] + 1))
    
    df["potencial_mejora"] = (df["Horas de estudio actuales "] - df["Horas estudio pasadas "]) * df["Calificaciones pasadas"] / 10
    df["carga_academica"] = df["Materias nuevas"] * (df["Horas de estudio actuales "] + 1)
    df["historial_fuerte"] = (df["Calificaciones pasadas"] >= 9.0).astype(int)
    
    return df

df = load_and_prepare_data()

feature_cols = [
    "Materias pasadas ",
    "Materias nuevas",
    "Horas de estudio actuales ",
    "Horas estudio pasadas ",
    "Calificaciones pasadas",
    "eficiencia_estudio_pasado",
    "intensidad_estudio_actual",
    "cambio_horas",
    "ratio_materias",
    "tendencia_academica",
    "potencial_mejora",
    "carga_academica",
    "historial_fuerte"
]

X = df[feature_cols]

Y_grade = df["Calificaciones pasadas"]
scaler_reg = StandardScaler()
X_scaled_reg = scaler_reg.fit_transform(X)

model_regression = RandomForestRegressor(
    n_estimators=150, random_state=42, max_depth=6, min_samples_leaf=2
)
model_regression.fit(X_scaled_reg, Y_grade)

def create_high_performance_target(row):
    score = 0
    
    if row["Calificaciones pasadas"] >= 9.2:
        score += 3
    elif row["Calificaciones pasadas"] >= 8.8:
        score += 2
    elif row["Calificaciones pasadas"] >= 8.5:
        score += 1
    
    if row["cambio_horas"] > 2:
        score += 2
    elif row["cambio_horas"] > 0:
        score += 1
    
    if row["eficiencia_estudio_pasado"] > 1.5:
        score += 2
    elif row["eficiencia_estudio_pasado"] > 1.2:
        score += 1
    
    if row["Materias nuevas"] <= row["Materias pasadas "]:
        score += 1
    
    if row["intensidad_estudio_actual"] >= 1.0:
        score += 1
    
    return 1 if score >= 5 else 0

Y_class = df.apply(create_high_performance_target, axis=1)

scaler_class = StandardScaler()
X_scaled_class = scaler_class.fit_transform(X)

model_classification = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)
model_classification.fit(X_scaled_class, Y_class)

model_logistic = LogisticRegression(
    C=0.5,
    max_iter=1000,
    solver="liblinear",
    class_weight='balanced',
    random_state=42
)
model_logistic.fit(X_scaled_class, Y_class)

st.title("Predictor de Calificaciones")
st.markdown("Predice tu calificación esperada y probabilidad de alto rendimiento")

st.markdown("---")
st.subheader("Información Personal")

col_info1, col_info2 = st.columns(2)

with col_info1:
    gender = st.selectbox("Género", ["Masculino", "Femenino", "Otro"], key="gender")

with col_info2:
    semester = st.selectbox("Semestre actual", list(range(1, 10)), format_func=lambda x: f"{x}° semestre", key="semester")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Semestre Anterior")
    courses_past = st.number_input("Materias cursadas", min_value=1, max_value=15, value=7)
    hours_past = st.number_input("Horas de estudio semanales", min_value=1, max_value=30, value=5)
    grade_past = st.number_input("Calificación final", min_value=6.0, max_value=10.0, value=9.0, step=0.1)

with col2:
    st.subheader("Semestre Actual")
    courses_now = st.number_input("Materias cursando", min_value=1, max_value=15, value=8)
    hours_now = st.number_input("Horas de estudio semanales", min_value=1, max_value=30, value=5)

eficiencia = grade_past / (hours_past + 1)
intensidad = hours_now / (courses_now + 1)
cambio_h = hours_now - hours_past
ratio_mat = courses_now / (courses_past + 1)
tendencia = grade_past * (hours_now / (hours_past + 1))
potencial_mejora = (hours_now - hours_past) * grade_past / 10
carga_academica = courses_now * (hours_now + 1)
historial_fuerte = 1 if grade_past >= 9.0 else 0

if st.button("Predecir Rendimiento", type="primary"):
    new_data = pd.DataFrame({
        "Materias pasadas ": [courses_past],
        "Materias nuevas": [courses_now],
        "Horas de estudio actuales ": [hours_now],
        "Horas estudio pasadas ": [hours_past],
        "Calificaciones pasadas": [grade_past],
        "eficiencia_estudio_pasado": [eficiencia],
        "intensidad_estudio_actual": [intensidad],
        "cambio_horas": [cambio_h],
        "ratio_materias": [ratio_mat],
        "tendencia_academica": [tendencia],
        "potencial_mejora": [potencial_mejora],
        "carga_academica": [carga_academica],
        "historial_fuerte": [historial_fuerte]
    })
    
    new_data_scaled_reg = scaler_reg.transform(new_data)
    predicted_grade = model_regression.predict(new_data_scaled_reg)[0]
    predicted_grade = np.clip(predicted_grade, 6.0, 10.0)
    
    new_data_scaled_class = scaler_class.transform(new_data)
    prediction_class = model_classification.predict(new_data_scaled_class)[0]
    probability = model_classification.predict_proba(new_data_scaled_class)[0][1]
    
    probability_logistic = model_logistic.predict_proba(new_data_scaled_class)[0][1]

    st.markdown("---")
    st.subheader("Resultados de la Predicción")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("Calificación Esperada")
        st.markdown(f"# {predicted_grade:.2f}")
        change = predicted_grade - grade_past
        st.metric("Cambio vs semestre anterior", f"{change:+.2f}")

    with col_b:
        st.markdown("Probabilidad de Alto Rendimiento")
        st.markdown(f"# {probability*100:.1f}%")
        result_text = "ALTO" if prediction_class == 1 else "MODERADO"
        st.metric("Clasificación", result_text)

    st.markdown("---")
    st.markdown("Interpretación")
    
    if probability >= 0.6 and predicted_grade >= 9.2:
        st.success(f"""
        Excelente proyección:
        - Probabilidad de alto rendimiento: {probability*100:.1f}%
        - Calificación esperada: {predicted_grade:.2f}
        """)
    elif probability >= 0.4 and predicted_grade >= 8.8:
        st.info(f"""
        Buen desempeño:
        - Probabilidad de alto rendimiento: {probability*100:.1f}%
        - Calificación esperada: {predicted_grade:.2f}
        """)
    else:
        st.warning(f"""
        Área de mejora:
        - Probabilidad actual: {probability*100:.1f}%
        - Calificación proyectada: {predicted_grade:.2f}
        """)
