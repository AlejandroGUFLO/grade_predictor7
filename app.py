import streamlit as st
import numpy as np
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Predicción de Rendimiento Académico", layout="centered")

st.title("🎓 Predicción de Rendimiento Académico")
st.markdown("Introduce tus valores para estimar tu probabilidad de obtener **≥ 9.2**.")

horas = st.slider("Horas de estudio por semana", 0, 40, 10)
sueno = st.slider("Horas de sueño por noche", 3, 12, 7)
estres = st.slider("Nivel de estrés (0 = bajo, 10 = muy alto)", 0, 10, 4)
motivacion = st.slider("Motivación (0 = nada motivado, 10 = muy motivado)", 0, 10, 7)

X_input = np.array([[horas, sueno, estres, motivacion]])

prob = model.predict_proba(X_input)[0][1]
prediccion = "ALTO rendimiento (≥9.2)" if prob >= 0.5 else "BAJO rendimiento"

st.subheader("📊 Resultado")
st.write(f"**Probabilidad de obtener ≥ 9.2:** {prob:.2%}")
st.write(f"**Clasificación:** {prediccion}")

if prob >= 0.5:
    st.success("¡Vas por buen camino! 🔥")
else:
    st.error("Tu rendimiento está por debajo del objetivo. Puedes mejorar.")
