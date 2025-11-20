import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go

# ===============================
# 1. CARGAR Y PREPARAR DATOS
# ===============================
@st.cache_data
def load_and_prepare_data():
    """Carga el archivo Excel y realiza ingeniería de características"""
    df = pd.read_excel("proyectom.xlsx")
    
    # Limpiar nombres de columnas (remover espacios extra)
    df.columns = df.columns.str.strip()
    
    # Remover filas con valores nulos en columnas críticas
    critical_cols = ["Calificaciones pasadas", "Horas estudio pasadas", "Horas de estudio actuales", 
                     "Materias pasadas", "Materias nuevas"]
    df = df.dropna(subset=critical_cols)
    
    # Variable objetivo: Alto rendimiento (≥9.2)
    df["HighPerformance"] = (df["Calificaciones pasadas"] >= 9.2).astype(int)
    
    # Ingeniería de características
    df["eficiencia_estudio_pasado"] = df["Calificaciones pasadas"] / (df["Horas estudio pasadas"] + 1)
    df["intensidad_estudio_actual"] = df["Horas de estudio actuales"] / (df["Materias nuevas"] + 1)
    df["cambio_horas"] = df["Horas de estudio actuales"] - df["Horas estudio pasadas"]
    df["ratio_materias"] = df["Materias nuevas"] / (df["Materias pasadas"] + 1)
    df["tendencia_academica"] = df["Calificaciones pasadas"] * (df["Horas de estudio actuales"] / (df["Horas estudio pasadas"] + 1))
    
    return df

# Cargar datos
df = load_and_prepare_data()

# ===============================
# 2. DEFINIR FEATURES
# ===============================
feature_cols = [
    "Materias pasadas",
    "Materias nuevas",
    "Horas de estudio actuales",
    "Horas estudio pasadas",
    "Calificaciones pasadas",
    "eficiencia_estudio_pasado",
    "intensidad_estudio_actual",
    "cambio_horas",
    "ratio_materias",
    "tendencia_academica"
]

X = df[feature_cols].copy()

# ===============================
# 3. ENTRENAR MODELO DE REGRESIÓN
# ===============================
Y_grade = df["Calificaciones pasadas"]
scaler_reg = StandardScaler()
X_scaled_reg = scaler_reg.fit_transform(X)

model_regression = RandomForestRegressor(
    n_estimators=200, 
    random_state=42, 
    max_depth=8, 
    min_samples_leaf=2,
    min_samples_split=3
)
model_regression.fit(X_scaled_reg, Y_grade)

# ===============================
# 4. ENTRENAR MODELO DE REGRESIÓN LOGÍSTICA
# ===============================
Y_class = df["HighPerformance"]
scaler_class = StandardScaler()
X_scaled_class = scaler_class.fit_transform(X)

model_classification = LogisticRegression(
    C=0.1,  # ← REDUCIDO (más regularización = menos sobreajuste)
    max_iter=1000,
    solver="lbfgs",
    random_state=42,
    class_weight='balanced',
    penalty='l2'  # ← AGREGADO (regularización L2)
)
model_classification.fit(X_scaled_class, Y_class)

# ===============================
# 5. FUNCIÓN DE VALIDACIÓN
# ===============================
def validate_prediction(predicted_grade):
    """Asegurar que la predicción esté en rango válido (6-10)"""
    return np.clip(predicted_grade, 6.0, 10.0)

# ===============================
# 6. INTERFAZ STREAMLIT
# ===============================
st.title("🎓 Predictor de Calificaciones")
st.markdown("*Predice tu calificación esperada y probabilidad de alto rendimiento*")

# ===============================
# 7. SECCIÓN: INFORMACIÓN PERSONAL
# ===============================
st.markdown("---")
st.subheader("👤 Información Personal")

col_info1, col_info2 = st.columns(2)

with col_info1:
    gender = st.selectbox("Género", ["Masculino", "Femenino", "Otro"], key="gender")

with col_info2:
    semester = st.selectbox("Semestre actual", list(range(1, 10)), format_func=lambda x: f"{x}° semestre", key="semester")

# ===============================
# 8. SECCIÓN: DATOS DE ENTRADA
# ===============================
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

# ===============================
# 9. CALCULAR FEATURES DERIVADAS
# ===============================
eficiencia = grade_past / (hours_past + 1)
intensidad = hours_now / (courses_now + 1)
cambio_h = hours_now - hours_past
ratio_mat = courses_now / (courses_past + 1)
tendencia = grade_past * (hours_now / (hours_past + 1))

# ===============================
# 10. REALIZAR PREDICCIÓN
# ===============================
if st.button("🔮 Predecir Rendimiento", type="primary"):
    # Crear DataFrame con nuevos datos
    new_data = pd.DataFrame({
        "Materias pasadas": [courses_past],
        "Materias nuevas": [courses_now],
        "Horas de estudio actuales": [hours_now],
        "Horas estudio pasadas": [hours_past],
        "Calificaciones pasadas": [grade_past],
        "eficiencia_estudio_pasado": [eficiencia],
        "intensidad_estudio_actual": [intensidad],
        "cambio_horas": [cambio_h],
        "ratio_materias": [ratio_mat],
        "tendencia_academica": [tendencia]
    })
    
    # Predicción de calificación (Regresión)
    new_data_scaled_reg = scaler_reg.transform(new_data)
    predicted_grade = model_regression.predict(new_data_scaled_reg)[0]
    predicted_grade = validate_prediction(predicted_grade)
    
    # Predicción de clasificación (Regresión Logística)
    new_data_scaled_class = scaler_class.transform(new_data)
    prediction_class = model_classification.predict(new_data_scaled_class)[0]
    probability = model_classification.predict_proba(new_data_scaled_class)[0][1]
    
    # ===============================
    # 11. MOSTRAR RESULTADOS
    # ===============================
    st.markdown("---")
    st.subheader("📊 Resultados de la Predicción")
    
    st.info("📌 **Cómo funciona:**\n- 🔴 **Regresión (izquierda)**: Predice tu calificación exacta (número entre 6-10)\n- 🟢 **Regresión Logística (derecha)**: Predice probabilidad de obtener ≥9.2 (SÍ/NO)")
    
    # Dos columnas para las dos predicciones
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 🎯 Calificación Esperada")
        grade_color = "🟢" if predicted_grade >= 9.2 else "🟡" if predicted_grade >= 8.5 else "🔴"
        st.markdown(f"# {grade_color} {predicted_grade:.2f}")
        change = predicted_grade - grade_past
        st.metric(
            "Cambio vs semestre anterior",
            f"{change:+.2f} puntos",
            delta=f"{change:+.2f}"
        )
    
    with col_b:
        st.markdown("### 📈 Alto Rendimiento (≥9.2)")
        prob_color = "🟢" if probability >= 0.7 else "🟡" if probability >= 0.4 else "🔴"
        st.markdown(f"# {prob_color} {probability*100:.1f}%")
        result_text = "✅ SÍ" if prediction_class == 1 else "⚠️ NO"
        st.metric(
            "Predicción",
            result_text,
            delta=f"{probability*100:.1f}%"
        )
    
    # Métricas adicionales
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Eficiencia de Estudio",
            f"{eficiencia:.2f}",
            help="Calificación / hora de estudio"
        )
    
    with col2:
        st.metric(
            "Intensidad Actual",
            f"{intensidad:.2f}",
            help="Horas / materia"
        )
    
    with col3:
        st.metric(
            "Cambio en Horas",
            f"{cambio_h:+.0f}h",
            help="Diferencia vs semestre anterior"
        )
    
    # ===============================
    # 12. GRÁFICOS PRINCIPALES
    # ===============================
    col_gauge1, col_gauge2 = st.columns(2)
    
    with col_gauge1:
    # Gráfico tipo velocímetro para calificación
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_grade,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Calificación Esperada", 'font': {'size': 20}},
        delta={'reference': grade_past, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        number={'font': {'size': 50, 'color': 'darkblue'}},  # ✅ AUMENTADO y con color
        gauge={
            'axis': {'range': [6, 10], 'tickwidth': 2, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue", 'thickness': 0.75},
            'steps': [
                {'range': [6, 7], 'color': "#ffcccc"},
                {'range': [7, 8], 'color': "#fff4cc"},
                {'range': [8, 9], 'color': "#cce5ff"},
                {'range': [9, 10], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.85,
                'value': 9.2
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    
    with col_gauge2:
        # Gráfico de barras para probabilidades (Regresión Logística)
        prob_bajo_graph = 1 - probability
        categories = ["✅ Alto\nRendimiento\n(≥9.2)", "⚠️ No Alcanza\nAlto\nRendimiento"]
        probs = [probability * 100, prob_bajo_graph * 100]
        colors_probs = ["#2ecc71", "#e74c3c"]
        
        fig_prob = go.Figure(data=[
            go.Bar(
                x=categories,
                y=probs,
                marker=dict(color=colors_probs),
                text=[f"{p:.1f}%" for p in probs],
                textposition='auto',
                textfont=dict(size=16, color='white'),
                hovertemplate="<b>%{x}</b><br>Probabilidad: %{y:.1f}%<extra></extra>"
            )
        ])
        
        fig_prob.update_layout(
            title="Predicción por Regresión Logística",
            yaxis_title="Probabilidad (%)",
            yaxis=dict(range=[0, 105]),
            height=350,
            showlegend=False,
            hovermode='x',
            xaxis=dict(tickfont=dict(size=10))
        )
        
        st.plotly_chart(fig_prob, use_container_width=True)
    
    # ===============================
    # 13. ANÁLISIS DETALLADO
    # ===============================
    grade_change = predicted_grade - grade_past
    
    st.markdown("---")
    st.subheader("📊 Análisis Detallado")
    
    # Tabla de probabilidades
    prob_bajo_table = 1 - probability
    analysis_data = {
        "Categoría": ["✅ Alto Rendimiento (≥9.2)", "⚠️ No Alcanza Alto Rendimiento (<9.2)"],
        "Probabilidad": [f"{probability*100:.1f}%", f"{prob_bajo_table*100:.1f}%"],
        "Interpretación": [
            "Posibilidad de alcanzar la meta" if probability >= 0.5 else "Difícil pero posible",
            "Probabilidad complementaria" if prob_bajo_table >= 0.5 else "Muy probable éxito"
        ]
    }
    
    df_analysis = pd.DataFrame(analysis_data)
    st.dataframe(df_analysis, use_container_width=True, hide_index=True)
    
    # Interpretación general
    st.markdown("**Interpretación:**")
    if probability >= 0.7:
        st.success(f"✅ Tienes una **alta probabilidad ({probability*100:.1f}%)** de alcanzar alto rendimiento (≥9.2)")
    elif probability >= 0.5:
        st.info(f"🟡 Tienes una **probabilidad moderada ({probability*100:.1f}%)** de alcanzar alto rendimiento (≥9.2)")
    else:
        st.warning(f"🔴 Tienes una **baja probabilidad ({probability*100:.1f}%)** de alcanzar alto rendimiento (≥9.2)")
    
    if grade_change > 0.3:
        st.success(f"📈 **¡Excelente!** Se espera una mejora de **{grade_change:.2f} puntos**")
    elif grade_change < -0.3:
        st.error(f"📉 **Atención:** Se espera una baja de **{abs(grade_change):.2f} puntos**")
    else:
        st.info(f"📊 **Estable:** Calificación similar al semestre anterior ({grade_change:+.2f})")
    
    # ===============================
    # 14. RECOMENDACIONES
    # ===============================
    st.markdown("---")
    st.subheader("💡 Recomendaciones Personalizadas")
    
    if predicted_grade < 9.0:
        st.warning("**Sugerencias para mejorar tu calificación:**")
        
        if eficiencia < 1.5:
            st.write("• 📚 **Eficiencia baja:** Tu aprovechamiento es bajo. Mejora con:")
            st.write("  - Método Pomodoro (25 min estudio + 5 min descanso)")
            st.write("  - Estudio activo (resúmenes, mapas mentales)")
            st.write("  - Eliminar distracciones durante el estudio")
        
        if intensidad < 1.5:
            st.write(f"• ⏰ **Poco tiempo por materia:** Solo dedicas {intensidad:.1f} horas/materia")
            st.write("  - Aumenta el tiempo dedicado a cada materia")
            st.write("  - Enfócate en las materias más difíciles")
        
        if hours_now < hours_past and grade_past >= 9.0:
            st.write(f"• ⚠️ **Reducción de horas:** Pasaste de {hours_past}h a {hours_now}h semanales")
            st.write("  - Considera volver a tu carga anterior de horas")
        
        if grade_past < 8.5:
            st.write("• 🎯 **Historial bajo:** Busca apoyo adicional:")
            st.write("  - Grupos de estudio con compañeros")
            st.write("  - Tutorías o asesorías especializadas")
            st.write("  - Recursos en línea (Khan Academy, Coursera, etc.)")
    
    elif predicted_grade >= 9.2:
        st.success("**🌟 ¡Excelente proyección!**")
        st.write("• ✅ Mantén tus hábitos de estudio actuales")
        st.write("• 💪 Tu eficiencia de estudio es muy buena")
        st.write("• 🤝 Considera ayudar a compañeros con dificultades")
        st.write("• 📚 Podrías tomar una materia adicional si lo deseas")
    
    else:
        st.info("**✅ Buen camino - Estás cerca del alto rendimiento**")
        st.write(f"• 🎯 Solo necesitas **{9.2 - predicted_grade:.2f} puntos** más para llegar a 9.2")
        st.write("• ⏰ Aumentar 2-3 horas de estudio semanales podría ser suficiente")
        st.write("• 📖 Enfócate en técnicas de estudio más efectivas")
    
    # ===============================
    # 15. SIMULADOR
    # ===============================
    st.markdown("---")
    st.subheader("🔄 Simulador: Impacto de las Horas de Estudio")
    
    hours_scenarios = []
    grades_scenarios = []
    probs_scenarios = []
    
    for h in range(1, 21):
        sim_eficiencia = grade_past / (hours_past + 1)
        sim_intensidad = h / (courses_now + 1)
        sim_cambio = h - hours_past
        sim_tendencia = grade_past * (h / (hours_past + 1))
        
        sim_data = pd.DataFrame({
            "Materias pasadas": [courses_past],
            "Materias nuevas": [courses_now],
            "Horas de estudio actuales": [h],
            "Horas estudio pasadas": [hours_past],
            "Calificaciones pasadas": [grade_past],
            "eficiencia_estudio_pasado": [sim_eficiencia],
            "intensidad_estudio_actual": [sim_intensidad],
            "cambio_horas": [sim_cambio],
            "ratio_materias": [ratio_mat],
            "tendencia_academica": [sim_tendencia]
        })
        
        sim_scaled_reg = scaler_reg.transform(sim_data)
        sim_grade = model_regression.predict(sim_scaled_reg)[0]
        sim_grade = validate_prediction(sim_grade)
        
        sim_scaled_class = scaler_class.transform(sim_data)
        sim_prob = model_classification.predict_proba(sim_scaled_class)[0][1]
        
        hours_scenarios.append(h)
        grades_scenarios.append(sim_grade)
        probs_scenarios.append(sim_prob * 100)
    
    fig2 = go.Figure()
    
    # Línea de calificación esperada
    fig2.add_trace(go.Scatter(
        x=hours_scenarios,
        y=grades_scenarios,
        mode='lines+markers',
        name='Calificación esperada',
        line=dict(color='steelblue', width=3),
        marker=dict(size=6),
        yaxis='y1'
    ))
    
    # Marcar punto actual
    fig2.add_trace(go.Scatter(
        x=[hours_now],
        y=[predicted_grade],
        mode='markers',
        name='Tu situación actual',
        marker=dict(size=15, color='red', symbol='star'),
        yaxis='y1'
    ))
    
    # Línea de referencia 9.2
    fig2.add_hline(y=9.2, line_dash="dash", line_color="green", 
                   annotation_text="Alto rendimiento (9.2)", yref='y1')
    
    fig2.update_layout(
        title="¿Cómo afectan las horas de estudio a tu calificación?",
        xaxis_title="Horas de estudio semanales",
        yaxis_title="Calificación esperada",
        yaxis=dict(range=[6, 10]),
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Encontrar horas óptimas
    optimal_idx = np.argmax(grades_scenarios)
    optimal_hours = hours_scenarios[optimal_idx]
    max_grade = grades_scenarios[optimal_idx]
    
    st.info(f"💡 **Punto óptimo:** Con **{optimal_hours} horas** semanales podrías alcanzar **{max_grade:.2f}**")
    
    # ===============================
# 16. IMPORTANCIA DE VARIABLES (MEJORADO)
# ===============================
st.markdown("---")
st.subheader("📈 ¿Qué Afecta Más a tu Calificación?")

st.markdown("**Análisis basado en Regresión Logística:**\nEstos factores influyen en tu probabilidad de alcanzar alto rendimiento (≥9.2)")

feature_names_readable = {
    "Materias pasadas": "Materias semestre anterior",
    "Materias nuevas": "Materias actuales",
    "Horas de estudio actuales": "Horas de estudio actuales",
    "Horas estudio pasadas": "Horas semestre anterior",
    "Calificaciones pasadas": "Calificación anterior",
    "eficiencia_estudio_pasado": "Eficiencia de estudio",
    "intensidad_estudio_actual": "Intensidad (horas/materia)",
    "cambio_horas": "Cambio en horas",
    "ratio_materias": "Cambio en materias",
    "tendencia_academica": "Tendencia académica"
}

# Obtener coeficientes en valor absoluto
coef_importance = np.abs(model_classification.coef_[0])

# ✅ MEJORA: Normalizar por desviación estándar de cada feature
feature_std = X_scaled_class.std(axis=0)
coef_normalized = coef_importance / (feature_std + 1e-8)

# ✅ MEJORA: Aplicar escala logarítmica para reducir el dominio de un factor
coef_log = np.log1p(coef_normalized)

feature_importance = pd.DataFrame({
    'Factor': [feature_names_readable[col] for col in feature_cols],
    'Importancia': coef_log
}).sort_values('Importancia', ascending=False)

# Normalizar a porcentaje
feature_importance['Porcentaje'] = (feature_importance['Importancia'] / feature_importance['Importancia'].sum() * 100)

fig3 = go.Figure(go.Bar(
    x=feature_importance['Porcentaje'],
    y=feature_importance['Factor'],
    orientation='h',
    marker=dict(
        color=feature_importance['Porcentaje'],
        colorscale='Greens',
        showscale=False
    ),
    text=feature_importance['Porcentaje'].round(1).astype(str) + '%',
    textposition='auto',
))
fig3.update_layout(
    title="Importancia relativa - Regresión Logística (Probabilidad de Alto Rendimiento)",
    xaxis_title="Importancia (%)",
    height=400,
    showlegend=False
)

st.plotly_chart(fig3, use_container_width=True)

st.caption("💡 Los factores más arriba son los que más influyen en tu probabilidad de alcanzar ≥9.2")

# ===============================
# 17. ESTADÍSTICAS DEL DATASET
# ===============================
with st.expander("📊 Ver estadísticas del dataset"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Estudiantes analizados", len(df))
    with col2:
        st.metric("Calificación promedio", f"{df['Calificaciones pasadas'].mean():.2f}")
    with col3:
        st.metric("Alto rendimiento", f"{(Y_class.sum()/len(Y_class)*100):.1f}%")
    with col4:
        st.metric("Horas promedio", f"{df['Horas de estudio actuales'].mean():.1f}")
