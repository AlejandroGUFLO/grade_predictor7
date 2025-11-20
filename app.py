import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
import altair as alt

# --- 1. CONFIGURACIÓN VISUAL (CSS) ---
st.set_page_config(page_title="Predictor Académico", layout="centered")

st.markdown("""
<style>
    /* Estilos para las tarjetas */
    .stat-card {
        background-color: #F8F9FE;
        border: 1px solid #E1E4E8;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stat-title {
        color: #1F2937;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
        font-size: 1.1em;
    }
    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #eee;
        font-size: 0.9em;
        color: #555;
    }
    .stat-row:last-child {
        border-bottom: none;
    }
    .stat-val {
        font-weight: 600;
        color: #3B82F6;
    }
    
    /* Estilos para el resultado final */
    .result-card-success {
        background-color: #ECFDF5;
        border: 1px solid #34D399;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-card-warning {
        background-color: #FFFBEB;
        border: 1px solid #F59E0B;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #EFF6FF;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .big-number {
        font-size: 24px;
        font-weight: bold;
        color: #4F46E5;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. FUNCIONES DE DATOS Y MODELO ---

@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_excel("proyectom.xlsx")
    except FileNotFoundError:
        return pd.DataFrame()

    df["eficiencia_estudio_pasado"] = df["Calificaciones pasadas"] / (df["Horas estudio pasadas "] + 1)
    df["intensidad_estudio_actual"] = df["Horas de estudio actuales "] / (df["Materias nuevas"] + 1)
    df["cambio_horas"] = df["Horas de estudio actuales "] - df["Horas estudio pasadas "]
    df["ratio_materias"] = df["Materias nuevas"] / (df["Materias pasadas "] + 1)
    df["tendencia_academica"] = df["Calificaciones pasadas"] * (df["Horas de estudio actuales "] / (df["Horas estudio pasadas "] + 1))
    df["potencial_mejora"] = (df["Horas de estudio actuales "] - df["Horas estudio pasadas "]) * df["Calificaciones pasadas"] / 10
    df["carga_academica"] = df["Materias nuevas"] * (df["Horas de estudio actuales "] + 1)
    df["historial_fuerte"] = (df["Calificaciones pasadas"] >= 9.0).astype(int)
    return df

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

@st.cache_resource
def train_models(df):
    feature_cols = [
        "Materias pasadas ", "Materias nuevas", "Horas de estudio actuales ",
        "Horas estudio pasadas ", "eficiencia_estudio_pasado", 
        "intensidad_estudio_actual", "cambio_horas", "ratio_materias",
        "tendencia_academica", "potencial_mejora", "carga_academica", "historial_fuerte"
    ]
    
    X = df[feature_cols]
    Y_grade = df["Calificaciones pasadas"]
    Y_class = df.apply(create_high_performance_target, axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_reg = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=6)
    model_reg.fit(X_scaled, Y_grade)

    model_class = GradientBoostingClassifier(random_state=42)
    model_class.fit(X_scaled, Y_class)
    
    # Calculamos la precisión del modelo para mostrarla en el gráfico
    accuracy_score = model_class.score(X_scaled, Y_class)

    return scaler, model_reg, model_class, feature_cols, accuracy_score

# --- 3. HELPER PARA MOSTRAR ESTADÍSTICAS (Imagen 1) ---
def render_stats_card(title, series):
    desc = series.describe()
    stats_html = f"""
    <div class="stat-card">
        <div class="stat-title">{title}</div>
        <div class="stat-row"><span>N:</span><span class="stat-val">{int(desc['count'])}</span></div>
        <div class="stat-row"><span>Media:</span><span class="stat-val">{desc['mean']:.2f}</span></div>
        <div class="stat-row"><span>Mediana:</span><span class="stat-val">{desc['50%']:.2f}</span></div>
        <div class="stat-row"><span>Desv. Est.:</span><span class="stat-val">{desc['std']:.2f}</span></div>
        <div class="stat-row"><span>Mínimo:</span><span class="stat-val">{desc['min']:.2f}</span></div>
        <div class="stat-row"><span>Máximo:</span><span class="stat-val">{desc['max']:.2f}</span></div>
        <div class="stat-row"><span>Q1 (25%):</span><span class="stat-val">{desc['25%']:.2f}</span></div>
        <div class="stat-row"><span>Q3 (75%):</span><span class="stat-val">{desc['75%']:.2f}</span></div>
    </div>
    """
    st.markdown(stats_html, unsafe_allow_html=True)

# --- 4. INTERFAZ PRINCIPAL ---

st.title("📊 Análisis y Predicción Académica")

df = load_and_prepare_data()

if df.empty:
    st.warning("Por favor sube el archivo 'proyectom.xlsx' para continuar.")
else:
    scaler, model_reg, model_class, feature_cols, accuracy = train_models(df)

    # --- SECCIÓN: ESTADÍSTICAS DEL MODELO (Como Imagen 1) ---
    with st.expander("📈 Ver Estadísticas del Dataset", expanded=False):
        st.subheader("Estadísticas Descriptivas")
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            render_stats_card("Materias Semestre Pasado", df["Materias pasadas "])
        
        with col_stat2:
            render_stats_card("Materias Semestre Actual", df["Materias nuevas"])

    st.markdown("---")
    st.subheader("Simulador de Rendimiento")

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        courses_past = st.number_input("Materias cursadas (anterior)", 1, 15, 7)
        hours_past = st.number_input("Horas estudio (anterior)", 1, 40, 5)
        grade_past = st.number_input("Calificación anterior", 0.0, 10.0, 8.5, step=0.1)
    with col2:
        courses_now = st.number_input("Materias actuales", 1, 15, 8)
        hours_now = st.number_input("Horas estudio (actual)", 1, 40, 6)

    # Variables derivadas
    eficiencia = grade_past / (hours_past + 1)
    intensidad = hours_now / (courses_now + 1)
    cambio_h = hours_now - hours_past
    ratio_mat = courses_now / (courses_past + 1)
    tendencia = grade_past * (hours_now / (hours_past + 1))
    potencial_mejora = (hours_now - hours_past) * grade_past / 10
    carga_academica = courses_now * (hours_now + 1)
    historial_fuerte = 1 if grade_past >= 9.0 else 0

    if st.button("Calcular Predicción", type="primary"):
        # Preparar datos
        new_data = pd.DataFrame([{
            "Materias pasadas ": courses_past, "Materias nuevas": courses_now,
            "Horas de estudio actuales ": hours_now, "Horas estudio pasadas ": hours_past,
            "eficiencia_estudio_pasado": eficiencia, "intensidad_estudio_actual": intensidad,
            "cambio_horas": cambio_h, "ratio_materias": ratio_mat,
            "tendencia_academica": tendencia, "potencial_mejora": potencial_mejora,
            "carga_academica": carga_academica, "historial_fuerte": historial_fuerte
        }])[feature_cols]

        # Predicciones
        X_input = scaler.transform(new_data)
        pred_grade = np.clip(model_reg.predict(X_input)[0], 0, 10)
        prob_high = model_class.predict_proba(X_input)[0][1]
        
        # --- SECCIÓN: RESULTADOS (Como Imagen 2) ---
        st.markdown("### Resultados de la Predicción")
        
        # 1. Tarjeta Grande de Resultado
        if prob_high >= 0.5:
            st.markdown(f"""
            <div class="result-card-success">
                <h2 style="color: #065F46; margin:0;">🎖️ Excelente Proyección</h2>
                <p style="color: #065F46;">Calificación esperada: <b>{pred_grade:.2f}</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-warning">
                <h2 style="color: #92400E; margin:0;">⚠️ Atención Requerida</h2>
                <p style="color: #92400E;">Calificación esperada: <b>{pred_grade:.2f}</b></p>
            </div>
            """, unsafe_allow_html=True)

        # 2. Métricas de Porcentaje
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size: 12px; color: #6B7280;">Probabilidad de Alto Rendimiento</div>
                <div class="big-number" style="color: #4F46E5;">{prob_high*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m_col2:
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size: 12px; color: #6B7280;">Precisión del Modelo (Dataset)</div>
                <div class="big-number" style="color: #4F46E5;">{accuracy*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # 3. Gráfico de Barras (Análisis Visual)
        st.write("Análisis Visual")
        
        chart_data = pd.DataFrame({
            'Métrica': ['Probabilidad', 'Precisión'],
            'Valor': [prob_high * 100, accuracy * 100]
        })

        # Gráfico de Altair para imitar las barras azules sólidas
        chart = alt.Chart(chart_data).mark_bar(color='#4F46E5', size=50).encode(
            x=alt.X('Métrica', axis=None),
            y=alt.Y('Valor', scale=alt.Scale(domain=[0, 100])),
            tooltip=['Métrica', 'Valor']
        ).properties(height=200)

        st.altair_chart(chart, use_container_width=True)
