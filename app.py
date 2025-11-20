import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go

# ------------------------------
# Load and prepare data
# ------------------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_excel("proyectom.xlsx")
    
    # Feature engineering - variables normalizadas
    df["eficiencia_estudio_pasado"] = df["Calificaciones pasadas"] / (df["Horas estudio pasadas "] + 1)
    df["intensidad_estudio_actual"] = df["Horas de estudio actuales "] / (df["Materias nuevas"] + 1)
    df["cambio_horas"] = df["Horas de estudio actuales "] - df["Horas estudio pasadas "]
    df["ratio_materias"] = df["Materias nuevas"] / (df["Materias pasadas "] + 1)
    df["tendencia_academica"] = df["Calificaciones pasadas"] * (df["Horas de estudio actuales "] / (df["Horas estudio pasadas "] + 1))
    
    # ✅ NUEVAS FEATURES PREDICTIVAS
    df["potencial_mejora"] = (df["Horas de estudio actuales "] - df["Horas estudio pasadas "]) * df["Calificaciones pasadas"] / 10
    df["carga_academica"] = df["Materias nuevas"] * (df["Horas de estudio actuales "] + 1)
    df["historial_fuerte"] = (df["Calificaciones pasadas"] >= 9.0).astype(int)
    
    return df

df = load_and_prepare_data()

# Features
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

# --------------------------------------------------------
# MODELO 1: REGRESIÓN (CALIFICACIÓN EXACTA)
# --------------------------------------------------------
Y_grade = df["Calificaciones pasadas"]
scaler_reg = StandardScaler()
X_scaled_reg = scaler_reg.fit_transform(X)
model_regression = RandomForestRegressor(
    n_estimators=150, random_state=42, max_depth=6, min_samples_leaf=2
)
model_regression.fit(X_scaled_reg, Y_grade)

# --------------------------------------------------------
# MODELO 2: CLASIFICACIÓN MEJORADA con lógica predictiva
# --------------------------------------------------------
# ✅ Crear objetivo basado en COMBINACIÓN de factores favorables
def create_high_performance_target(row):
    """
    Determina si un estudiante tiene potencial de alto rendimiento
    basado en múltiples factores predictivos
    """
    score = 0
    
    # Factor 1: Calificación histórica fuerte
    if row["Calificaciones pasadas"] >= 9.2:
        score += 3
    elif row["Calificaciones pasadas"] >= 8.8:
        score += 2
    elif row["Calificaciones pasadas"] >= 8.5:
        score += 1
    
    # Factor 2: Incremento en horas de estudio
    if row["cambio_horas"] > 2:
        score += 2
    elif row["cambio_horas"] > 0:
        score += 1
    
    # Factor 3: Buena eficiencia de estudio
    if row["eficiencia_estudio_pasado"] > 1.5:
        score += 2
    elif row["eficiencia_estudio_pasado"] > 1.2:
        score += 1
    
    # Factor 4: Carga académica manejable
    if row["Materias nuevas"] <= row["Materias pasadas "]:
        score += 1
    
    # Factor 5: Intensidad adecuada
    if row["intensidad_estudio_actual"] >= 1.0:
        score += 1
    
    # ✅ Si tiene 5+ puntos, tiene alto potencial
    return 1 if score >= 5 else 0

# Aplicar la función para crear el target
Y_class = df.apply(create_high_performance_target, axis=1)

# Verificar distribución
positive_rate = Y_class.sum() / len(Y_class)
st.sidebar.info(f"📊 Distribución de datos:\n- Alto potencial: {positive_rate*100:.1f}%\n- Casos positivos: {Y_class.sum()}/{len(Y_class)}")

scaler_class = StandardScaler()
X_scaled_class = scaler_class.fit_transform(X)

# ✅ Usar Gradient Boosting que maneja mejor datos desbalanceados
model_classification = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)
model_classification.fit(X_scaled_class, Y_class)

# También entrenar Logistic Regression para comparación
model_logistic = LogisticRegression(
    C=0.5,
    max_iter=1000,
    solver="liblinear",
    class_weight='balanced',
    random_state=42
)
model_logistic.fit(X_scaled_class, Y_class)

# ------------------------------
# UI
# ------------------------------
st.title("🎓 Predictor de Calificaciones")
st.markdown("Predice tu calificación esperada y probabilidad de alto rendimiento")

st.markdown("---")
st.subheader("👤 Información Personal")

col_info1, col_info2 = st.columns(2)

with col_info1:
    gender = st.selectbox("Género", ["Masculino", "Femenino", "Otro"], key="gender")

with col_info2:
    semester = st.selectbox("Semestre actual", list(range(1, 10)), format_func=lambda x: f"{x}° semestre", key="semester")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Semestre Anterior")
    courses_past = st.number_input("Materias cursadas", min_value=1, max_value=15, value=7, key="cp")
    hours_past = st.number_input("Horas de estudio semanales", min_value=1, max_value=30, value=5, key="hp")
    grade_past = st.number_input("Calificación final", min_value=6.0, max_value=10.0, value=9.0, step=0.1, key="gp")

with col2:
    st.subheader("📖 Semestre Actual")
    courses_now = st.number_input("Materias cursando", min_value=1, max_value=15, value=8, key="cn")
    hours_now = st.number_input("Horas de estudio semanales", min_value=1, max_value=30, value=5, key="hn")

# ------------------------------
# Cálculo de features derivadas
# ------------------------------
eficiencia = grade_past / (hours_past + 1)
intensidad = hours_now / (courses_now + 1)
cambio_h = hours_now - hours_past
ratio_mat = courses_now / (courses_past + 1)
tendencia = grade_past * (hours_now / (hours_past + 1))
potencial_mejora = (hours_now - hours_past) * grade_past / 10
carga_academica = courses_now * (hours_now + 1)
historial_fuerte = 1 if grade_past >= 9.0 else 0

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔮 Predecir Rendimiento", type="primary"):
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
    
    # --- Predicción de REGRESIÓN ---
    new_data_scaled_reg = scaler_reg.transform(new_data)
    predicted_grade = model_regression.predict(new_data_scaled_reg)[0]
    predicted_grade = np.clip(predicted_grade, 6.0, 10.0)
    
    # --- Predicción CLASIFICACIÓN (Gradient Boosting) ---
    new_data_scaled_class = scaler_class.transform(new_data)
    prediction_class = model_classification.predict(new_data_scaled_class)[0]
    probability = model_classification.predict_proba(new_data_scaled_class)[0][1]
    
    # --- Predicción LOGÍSTICA (para comparar) ---
    probability_logistic = model_logistic.predict_proba(new_data_scaled_class)[0][1]
    
    st.markdown("---")
    st.subheader("📊 Resultados de la Predicción")
    
    st.info("📌 *Cómo funciona:\n- 🔴 **Regresión: Predice tu calificación exacta\n- 🟢 **Clasificación ML*: Analiza tu potencial de alto rendimiento (≥9.2) basado en múltiples factores")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🎯 Calificación Esperada")
        grade_color = "🟢" if predicted_grade >= 9.2 else "🟡" if predicted_grade >= 8.5 else "🔴"
        st.markdown(f"# {grade_color} {predicted_grade:.2f}")
        change = predicted_grade - grade_past
        st.metric("Cambio vs semestre anterior", f"{change:+.2f}", delta=f"{change:+.2f}")

    with col_b:
        st.markdown("### 📈 Potencial de Alto Rendimiento")
        prob_color = "🟢" if probability >= 0.6 else "🟡" if probability >= 0.4 else "🔴"
        st.markdown(f"# {prob_color} {probability*100:.1f}%")
        result_text = "✅ ALTO" if prediction_class == 1 else "⚠️ MODERADO"
        st.metric("Clasificación", result_text)
    
    # Comparación de modelos
    st.markdown("---")
    st.markdown("### 🔬 Comparación de Modelos")
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.metric(
            "🌳 Gradient Boosting",
            f"{probability*100:.1f}%",
            help="Modelo avanzado que analiza patrones complejos"
        )
    
    with col_m2:
        st.metric(
            "📊 Regresión Logística",
            f"{probability_logistic*100:.1f}%",
            help="Modelo estadístico tradicional"
        )
    
    # Métricas adicionales
    st.markdown("---")
    st.markdown("### 📊 Análisis de Factores")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Eficiencia",
            f"{eficiencia:.2f}",
            help="Cal. / horas",
            delta="Buena" if eficiencia > 1.5 else "Regular" if eficiencia > 1.2 else "Baja"
        )
    
    with col2:
        st.metric(
            "Intensidad",
            f"{intensidad:.2f}",
            help="Horas / materia",
            delta="Buena" if intensidad >= 1.0 else "Aumentar"
        )
    
    with col3:
        st.metric(
            "Cambio Horas",
            f"{cambio_h:+.0f}h",
            help="Diferencia vs anterior",
            delta="Positivo" if cambio_h > 0 else "Mantener" if cambio_h == 0 else "Atención"
        )
    
    with col4:
        st.metric(
            "Historial",
            "Fuerte" if historial_fuerte else "Regular",
            help="Calificación ≥ 9.0"
        )
    
    # Interpretación detallada
    st.markdown("---")
    st.markdown("### 💡 Interpretación")
    
    if probability >= 0.6 and predicted_grade >= 9.2:
        st.success(f"""
        *🌟 ¡Excelente proyección!*
        - Tu probabilidad de alto rendimiento es *{probability*100:.1f}%*
        - Se espera una calificación de *{predicted_grade:.2f}*
        - Mantén tus hábitos de estudio actuales
        """)
    elif probability >= 0.4 and predicted_grade >= 8.8:
        st.info(f"""
        *✅ Buen camino*
        - Tienes *{probability*100:.1f}%* de alcanzar alto rendimiento
        - Calificación esperada: *{predicted_grade:.2f}*
        - Solo necesitas *{9.2 - predicted_grade:.2f} puntos* más para 9.2
        - Considera aumentar 2-3 horas de estudio semanales
        """)
    else:
        st.warning(f"""
        *⚠️ Área de mejora*
        - Probabilidad actual: *{probability*100:.1f}%*
        - Calificación proyectada: *{predicted_grade:.2f}*
        - Necesitas *{9.2 - predicted_grade:.2f} puntos* para alto rendimiento
        
        *Recomendaciones:*
        """)
        
        if eficiencia < 1.5:
            st.write("• 📚 Mejorar eficiencia de estudio (técnicas de estudio activo)")
        if cambio_h <= 0 and grade_past < 9.0:
            st.write("• ⏰ Aumentar horas de estudio semanales")
        if intensidad < 1.0:
            st.write("• 📖 Dedicar más tiempo por materia")

# Estadísticas del dataset
with st.expander("📊 Ver estadísticas del dataset"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Estudiantes analizados", len(df))
    with col2:
        st.metric("Calificación promedio", f"{df['Calificaciones pasadas'].mean():.2f}")
    with col3:
        st.metric("Con potencial alto", f"{(Y_class.sum()/len(Y_class)*100):.1f}%")
    with col4:
        st.metric("Horas promedio", f"{df['Horas de estudio actuales '].mean():.1f}")
