import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---

@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_excel("proyectom.xlsx")
    except FileNotFoundError:
        st.error("⚠️ No se encontró el archivo 'proyectom.xlsx'. Asegúrate de subirlo.")
        return pd.DataFrame()

    # Ingeniería de características (Feature Engineering)
    df["eficiencia_estudio_pasado"] = df["Calificaciones pasadas"] / (df["Horas estudio pasadas "] + 1)
    df["intensidad_estudio_actual"] = df["Horas de estudio actuales "] / (df["Materias nuevas"] + 1)
    df["cambio_horas"] = df["Horas de estudio actuales "] - df["Horas estudio pasadas "]
    df["ratio_materias"] = df["Materias nuevas"] / (df["Materias pasadas "] + 1)
    df["tendencia_academica"] = df["Calificaciones pasadas"] * (df["Horas de estudio actuales "] / (df["Horas estudio pasadas "] + 1))
    df["potencial_mejora"] = (df["Horas de estudio actuales "] - df["Horas estudio pasadas "]) * df["Calificaciones pasadas"] / 10
    df["carga_academica"] = df["Materias nuevas"] * (df["Horas de estudio actuales "] + 1)
    df["historial_fuerte"] = (df["Calificaciones pasadas"] >= 9.0).astype(int)
    
    return df

# Definir función para el target de clasificación fuera para evitar errores de serialización
def create_high_performance_target(row):
    score = 0
    if row["Calificaciones pasadas"] >= 9.2: score += 3
    elif row["Calificaciones pasadas"] >= 8.8: score += 2
    elif row["Calificaciones pasadas"] >= 8.5: score += 1
    
    if row["cambio_horas"] > 2: score += 2
    elif row["cambio_horas"] > 0: score += 1
    
    if row["eficiencia_estudio_pasado"] > 1.5: score += 2
    elif row["eficiencia_estudio_pasado"] > 1.2: score += 1
    
    if row["Materias nuevas"] <= row["Materias pasadas "]: score += 1
    if row["intensidad_estudio_actual"] >= 1.0: score += 1
    
    return 1 if score >= 5 else 0

# --- 2. ENTRENAMIENTO DE MODELOS (CON CACHÉ) ---

@st.cache_resource
def train_models(df):
    # IMPORTANTE: Eliminamos "Calificaciones pasadas" de X para evitar Data Leakage
    feature_cols = [
        "Materias pasadas ",
        "Materias nuevas",
        "Horas de estudio actuales ",
        "Horas estudio pasadas ",
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
    
    # Target para regresión (Nota numérica)
    Y_grade = df["Calificaciones pasadas"]
    
    # Target para clasificación (Alto Rendimiento SI/NO)
    Y_class = df.apply(create_high_performance_target, axis=1)

    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Modelo de Regresión (Random Forest)
    model_reg = RandomForestRegressor(
        n_estimators=150, random_state=42, max_depth=6, min_samples_leaf=2
    )
    model_reg.fit(X_scaled, Y_grade)

    # Modelo de Clasificación (Gradient Boosting)
    model_class = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, 
        min_samples_split=5, min_samples_leaf=3, random_state=42
    )
    model_class.fit(X_scaled, Y_class)

    return scaler, model_reg, model_class, feature_cols

# --- 3. INTERFAZ DE USUARIO ---

st.title("🎓 Predictor de Rendimiento Académico")
st.markdown("Ingresa tus datos para estimar tu calificación esperada basándote en tus hábitos de estudio.")

# Cargar y entrenar
df = load_and_prepare_data()

if not df.empty:
    scaler, model_reg, model_class, feature_cols = train_models(df)

    st.markdown("---")
    
    # Inputs del usuario
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Semestre Anterior")
        courses_past = st.number_input("Materias cursadas", min_value=1, max_value=15, value=7)
        hours_past = st.number_input("Horas estudio (semestre pasado)", min_value=1, max_value=40, value=5)
        grade_past = st.number_input("Calificación final obtenida", min_value=0.0, max_value=10.0, value=8.5, step=0.1)

    with col2:
        st.subheader("Semestre Actual")
        courses_now = st.number_input("Materias a cursar", min_value=1, max_value=15, value=8)
        hours_now = st.number_input("Horas estudio planeadas (semanal)", min_value=1, max_value=40, value=6)

    # Cálculos en tiempo real de las variables derivadas
    eficiencia = grade_past / (hours_past + 1)
    intensidad = hours_now / (courses_now + 1)
    cambio_h = hours_now - hours_past
    ratio_mat = courses_now / (courses_past + 1)
    tendencia = grade_past * (hours_now / (hours_past + 1))
    potencial_mejora = (hours_now - hours_past) * grade_past / 10
    carga_academica = courses_now * (hours_now + 1)
    historial_fuerte = 1 if grade_past >= 9.0 else 0

    # Botón de Predicción
    if st.button("Calcular Predicción", type="primary"):
        
        # Crear DataFrame con una sola fila para la predicción
        new_data = pd.DataFrame({
            "Materias pasadas ": [courses_past],
            "Materias nuevas": [courses_now],
            "Horas de estudio actuales ": [hours_now],
            "Horas estudio pasadas ": [hours_past],
            "eficiencia_estudio_pasado": [eficiencia],
            "intensidad_estudio_actual": [intensidad],
            "cambio_horas": [cambio_h],
            "ratio_materias": [ratio_mat],
            "tendencia_academica": [tendencia],
            "potencial_mejora": [potencial_mejora],
            "carga_academica": [carga_academica],
            "historial_fuerte": [historial_fuerte]
        })
        
        # Importante: Asegurar que las columnas estén en el mismo orden que en el entrenamiento
        new_data = new_data[feature_cols]
        
        # Escalar y Predecir
        new_data_scaled = scaler.transform(new_data)
        
        predicted_grade = model_reg.predict(new_data_scaled)[0]
        predicted_grade = np.clip(predicted_grade, 0.0, 10.0) # Limitar entre 0 y 10
        
        prediction_class = model_class.predict(new_data_scaled)[0]
        probability = model_class.predict_proba(new_data_scaled)[0][1]

        # Mostrar Resultados
        st.markdown("---")
        st.subheader("📊 Resultados del Análisis")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("### Calificación Esperada")
            st.markdown(f"<h1 style='color: #4CAF50;'>{predicted_grade:.2f}</h1>", unsafe_allow_html=True)
            diff = predicted_grade - grade_past
            if diff > 0:
                st.success(f"📈 Proyección: +{diff:.2f} vs semestre anterior")
            elif diff < 0:
                st.warning(f"📉 Proyección: {diff:.2f} vs semestre anterior")
            else:
                st.info("Mismo rendimiento esperado")

        with res_col2:
            st.markdown("### Probabilidad de Alto Rendimiento")
            st.progress(int(probability * 100))
            st.markdown(f"**{probability*100:.1f}%** de probabilidad de superar expectativas.")
            
            if prediction_class == 1:
                st.success("🌟 Clasificación: ALTO POTENCIAL")
            else:
                st.info("🔹 Clasificación: RENDIMIENTO ESTÁNDAR")

        # Interpretación final
        st.markdown("#### 💡 Interpretación")
        if probability >= 0.65:
            st.write("Tus hábitos actuales y el aumento en la intensidad de estudio sugieren un semestre muy exitoso.")
        elif probability >= 0.4:
            st.write("Tienes una base sólida, pero podrías beneficiarte de optimizar tus horas de estudio por materia.")
        else:
            st.write("La carga académica parece alta para las horas planeadas. Considera aumentar tus horas de estudio o priorizar materias clave.")

else:
    st.warning("Esperando archivo de datos...")
