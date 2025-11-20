import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.graph_objects as go
import plotly.express as px

# ===============================================
# TASK A: LOAD STUDENT DATA (CONFIDENTIAL)
# ===============================================
@st.cache_data
def load_and_prepare_data():
    """Load historical student data without personal information"""
    df = pd.read_excel("proyectom.xlsx")
    
    # Clean column names - remove extra spaces
    df.columns = df.columns.str.strip()
    
    # Map old column names to clean names
    column_mapping = {}
    for col in df.columns:
        clean_col = col.strip()
        if 'materias' in col.lower() and 'pasadas' in col.lower():
            column_mapping[col] = 'Materias pasadas'
        elif 'materias' in col.lower() and 'nuevas' in col.lower():
            column_mapping[col] = 'Materias nuevas'
        elif 'horas' in col.lower() and 'actuales' in col.lower():
            column_mapping[col] = 'Horas de estudio actuales'
        elif 'horas' in col.lower() and 'pasadas' in col.lower():
            column_mapping[col] = 'Horas estudio pasadas'
        elif 'calificaciones' in col.lower():
            column_mapping[col] = 'Calificaciones pasadas'
    
    # Rename columns
    df.rename(columns=column_mapping, inplace=True)
    
    # Feature engineering
    df["eficiencia_estudio_pasado"] = df["Calificaciones pasadas"] / (df["Horas estudio pasadas"] + 1)
    df["intensidad_estudio_actual"] = df["Horas de estudio actuales"] / (df["Materias nuevas"] + 1)
    df["cambio_horas"] = df["Horas de estudio actuales"] - df["Horas estudio pasadas"]
    df["ratio_materias"] = df["Materias nuevas"] / (df["Materias pasadas"] + 1)
    df["tendencia_academica"] = df["Calificaciones pasadas"] * (df["Horas de estudio actuales"] / (df["Horas estudio pasadas"] + 1))
    
    # Target variable: High Performance (≥9.2)
    df["HighPerformance"] = (df["Calificaciones pasadas"] >= 9.2).astype(int)
    
    return df

df = load_and_prepare_data()

# Define features
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
    "tendencia_academica"
]

X = df[feature_cols]
Y_grade = df["Calificaciones pasadas"]
Y_class = df["HighPerformance"]

# ===============================================
# REGRESSION MODEL (for exact grade prediction)
# ===============================================
scaler_reg = StandardScaler()
X_scaled_reg = scaler_reg.fit_transform(X)
model_regression = RandomForestRegressor(
    n_estimators=150, random_state=42, max_depth=6, min_samples_leaf=2
)
model_regression.fit(X_scaled_reg, Y_grade)

# ===============================================
# TASK C: TRAIN LOGISTIC REGRESSION MODEL
# ===============================================
scaler_class = StandardScaler()
X_scaled_class = scaler_class.fit_transform(X)

model_logistic = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="lbfgs",
    random_state=42
)
model_logistic.fit(X_scaled_class, Y_class)

# Model evaluation
y_pred = model_logistic.predict(X_scaled_class)
accuracy = accuracy_score(Y_class, y_pred)
conf_matrix = confusion_matrix(Y_class, y_pred)

# ===============================================
# STREAMLIT APP INTERFACE
# ===============================================
st.set_page_config(page_title="Grade Predictor", page_icon="🎓", layout="wide")

# Sidebar - Navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Go to:", [
    "🏠 Home & Prediction",
    "📊 Task B: Descriptive Report",
    "🤖 Task C & D: Model Parameters"
])

# ===============================================
# PAGE 1: HOME & PREDICTION
# ===============================================
if page == "🏠 Home & Prediction":
    st.title("🎓 Student Grade Prediction System")
    st.markdown("*Predict student performance using Logistic Regression*")
    
    st.markdown("---")
    st.info("""
    ### 📌 About This System
    - **Purpose**: Predict if a student will achieve high performance (≥9.2 grade)
    - **Model**: Logistic Regression (as required)
    - **Data**: Historical student performance (confidential, no names)
    - **Accuracy**: {:.1f}%
    """.format(accuracy * 100))
    
    st.markdown("---")
    st.subheader("👤 Student Information")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
    
    with col_info2:
        semester = st.selectbox("Current Semester", list(range(1, 10)), 
                               format_func=lambda x: f"{x}° Semester", key="semester")
    
    st.markdown("---")
    st.subheader("📝 Enter Academic Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📚 Previous Semester")
        courses_past = st.number_input("Courses taken", min_value=1, max_value=15, value=7, key="cp")
        hours_past = st.number_input("Weekly study hours", min_value=1, max_value=30, value=5, key="hp")
        grade_past = st.number_input("Final grade", min_value=6.0, max_value=10.0, value=9.0, step=0.1, key="gp")
    
    with col2:
        st.markdown("#### 📖 Current Semester")
        courses_now = st.number_input("Current courses", min_value=1, max_value=15, value=8, key="cn")
        hours_now = st.number_input("Weekly study hours", min_value=1, max_value=30, value=5, key="hn")
    
    # Calculate derived features
    eficiencia = grade_past / (hours_past + 1)
    intensidad = hours_now / (courses_now + 1)
    cambio_h = hours_now - hours_past
    ratio_mat = courses_now / (courses_past + 1)
    tendencia = grade_past * (hours_now / (hours_past + 1))
    
    # ===============================================
    # PREDICTION BUTTON
    # ===============================================
    if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
        # Prepare new data
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
            "tendencia_academica": [tendencia]
        })
        
        # REGRESSION prediction (exact grade)
        new_data_scaled_reg = scaler_reg.transform(new_data)
        predicted_grade = model_regression.predict(new_data_scaled_reg)[0]
        predicted_grade = np.clip(predicted_grade, 6.0, 10.0)
        
        # LOGISTIC REGRESSION prediction (high performance probability)
        new_data_scaled_class = scaler_class.transform(new_data)
        prediction_class = model_logistic.predict(new_data_scaled_class)[0]
        probability = model_logistic.predict_proba(new_data_scaled_class)[0][1]
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Display results
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric(
                "Expected Grade",
                f"{predicted_grade:.2f}",
                delta=f"{predicted_grade - grade_past:+.2f}",
                help="Predicted grade for current semester"
            )
        
        with col_b:
            st.metric(
                "High Performance Probability",
                f"{probability*100:.1f}%",
                help="Probability of achieving ≥9.2 (Logistic Regression)"
            )
        
        with col_c:
            result_text = "✅ YES" if prediction_class == 1 else "⚠️ NO"
            st.metric(
                "Classification",
                result_text,
                help="Binary prediction: High Performance (≥9.2)"
            )
        
        # Visual representation
        st.markdown("---")
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            # Gauge chart for grade
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_grade,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Expected Grade", 'font': {'size': 20}},
                delta={'reference': grade_past},
                gauge={
                    'axis': {'range': [6, 10]},
                    'bar': {'color': "darkblue"},
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
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col_g2:
            # Bar chart for probability
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=["High Performance\n(≥9.2)", "Below High\nPerformance"],
                    y=[probability * 100, (1-probability) * 100],
                    marker=dict(color=['#2ecc71', '#e74c3c']),
                    text=[f"{probability*100:.1f}%", f"{(1-probability)*100:.1f}%"],
                    textposition='auto',
                    textfont=dict(size=16, color='white')
                )
            ])
            fig_bar.update_layout(
                title="Logistic Regression Prediction",
                yaxis_title="Probability (%)",
                yaxis=dict(range=[0, 105]),
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Interpretation
        st.markdown("---")
        st.markdown("### 💡 Interpretation")
        
        if probability >= 0.7:
            st.success(f"✅ High probability ({probability*100:.1f}%) of achieving high performance (≥9.2)")
        elif probability >= 0.5:
            st.info(f"🟡 Moderate probability ({probability*100:.1f}%) of achieving high performance")
        else:
            st.warning(f"🔴 Low probability ({probability*100:.1f}%) of achieving high performance")
            st.markdown("**Recommendations:**")
            if eficiencia < 1.5:
                st.write("• 📚 Improve study efficiency")
            if cambio_h <= 0:
                st.write("• ⏰ Increase weekly study hours")
            if intensidad < 1.0:
                st.write("• 📖 Dedicate more time per subject")

# ===============================================
# PAGE 2: DESCRIPTIVE REPORT (TASK B)
# ===============================================
elif page == "📊 Task B: Descriptive Report":
    st.title("📊 Descriptive Statistical Report")
    st.markdown("### Historical Student Data Analysis")
    
    st.markdown("---")
    
    # Summary statistics
    st.subheader("1️⃣ Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Average Grade", f"{df['Calificaciones pasadas'].mean():.2f}")
    with col3:
        high_perf_pct = (Y_class.sum()/len(Y_class)*100)
        st.metric("High Performance Rate", f"{high_perf_pct:.1f}%")
    with col4:
        st.metric("Avg. Study Hours", f"{df['Horas de estudio actuales '].mean():.1f}")
    
    st.markdown("---")
    
    # Grade distribution
    st.subheader("2️⃣ Grade Distribution")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        fig_hist = px.histogram(
            df, 
            x="Calificaciones pasadas",
            nbins=20,
            title="Distribution of Past Grades",
            labels={"Calificaciones pasadas": "Grade"},
            color_discrete_sequence=['#3498db']
        )
        fig_hist.add_vline(x=9.2, line_dash="dash", line_color="red", 
                          annotation_text="High Performance Threshold")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_d2:
        # Box plot
        fig_box = px.box(
            df,
            y="Calificaciones pasadas",
            title="Grade Distribution Statistics",
            labels={"Calificaciones pasadas": "Grade"}
        )
        fig_box.add_hline(y=9.2, line_dash="dash", line_color="red")
        st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation analysis
    st.subheader("3️⃣ Feature Statistics")
    
    stats_df = df[feature_cols].describe().T
    stats_df = stats_df[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    stats_df.columns = ['Mean', 'Std Dev', 'Min', 'Q1', 'Median', 'Q3', 'Max']
    st.dataframe(stats_df.round(2), use_container_width=True)
    
    st.markdown("---")
    
    # Study hours vs grades
    st.subheader("4️⃣ Relationship: Study Hours vs Grades")
    
    fig_scatter = px.scatter(
        df,
        x="Horas de estudio actuales ",
        y="Calificaciones pasadas",
        color="HighPerformance",
        title="Study Hours vs Past Grades",
        labels={
            "Horas de estudio actuales ": "Current Study Hours",
            "Calificaciones pasadas": "Past Grade",
            "HighPerformance": "High Performance"
        },
        color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
        trendline="ols"
    )
    fig_scatter.add_hline(y=9.2, line_dash="dash", line_color="red")
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    # Class distribution
    st.subheader("5️⃣ High Performance Distribution")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        high_perf_counts = Y_class.value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Below High Performance', 'High Performance (≥9.2)'],
            values=high_perf_counts.values,
            marker=dict(colors=['#e74c3c', '#2ecc71']),
            textinfo='label+percent',
            textfont=dict(size=14)
        )])
        fig_pie.update_layout(title="Student Performance Classification")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_p2:
        st.markdown("#### Summary Statistics:")
        st.write(f"- **Total Students**: {len(df)}")
        st.write(f"- **High Performance**: {Y_class.sum()} ({high_perf_pct:.1f}%)")
        st.write(f"- **Below High Performance**: {len(df) - Y_class.sum()} ({100-high_perf_pct:.1f}%)")
        st.write(f"- **Average Grade**: {df['Calificaciones pasadas'].mean():.2f}")
        st.write(f"- **Grade Std Dev**: {df['Calificaciones pasadas'].std():.2f}")
        st.write(f"- **Highest Grade**: {df['Calificaciones pasadas'].max():.2f}")
        st.write(f"- **Lowest Grade**: {df['Calificaciones pasadas'].min():.2f}")

# ===============================================
# PAGE 3: MODEL PARAMETERS (TASK C & D)
# ===============================================
elif page == "🤖 Task C & D: Model Parameters":
    st.title("🤖 Logistic Regression Model Parameters")
    st.markdown("### Task C: Model Training & Task D: Parameters for Prediction")
    
    st.markdown("---")
    
    # Model performance
    st.subheader("📈 Model Performance Metrics")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
    with col_m2:
        precision = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[0,1]) if (conf_matrix[1,1] + conf_matrix[0,1]) > 0 else 0
        st.metric("Precision", f"{precision*100:.2f}%")
    with col_m3:
        recall = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0]) if (conf_matrix[1,1] + conf_matrix[1,0]) > 0 else 0
        st.metric("Recall", f"{recall*100:.2f}%")
    
    st.markdown("---")
    
    # Confusion matrix
    st.subheader("🔢 Confusion Matrix")
    
    col_c1, col_c2 = st.columns([1, 1])
    
    with col_c1:
        fig_cm = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=['Predicted: No', 'Predicted: Yes'],
            y=['Actual: No', 'Actual: Yes'],
            colorscale='Blues',
            text=conf_matrix,
            texttemplate='%{text}',
            textfont={"size": 20},
            showscale=True
        ))
        fig_cm.update_layout(
            title="Confusion Matrix",
            height=400
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col_c2:
        st.markdown("#### Confusion Matrix Interpretation:")
        st.write(f"- **True Negatives (TN)**: {conf_matrix[0,0]}")
        st.write(f"- **False Positives (FP)**: {conf_matrix[0,1]}")
        st.write(f"- **False Negatives (FN)**: {conf_matrix[1,0]}")
        st.write(f"- **True Positives (TP)**: {conf_matrix[1,1]}")
        st.write("")
        st.info(f"""
        **Model correctly predicts:**
        - {conf_matrix[0,0] + conf_matrix[1,1]} out of {len(Y_class)} cases
        - Accuracy: {accuracy*100:.2f}%
        """)
    
    st.markdown("---")
    
    # TASK D: Model coefficients (parameters)
    st.subheader("📊 Task D: Model Parameters (Coefficients)")
    
    st.info("""
    ### 🔑 How to Use These Parameters for New Predictions
    
    The **Logistic Regression equation** is:
    
    **P(High Performance) = 1 / (1 + e^(-z))**
    
    Where: **z = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ**
    
    - **β₀** = Intercept (bias term)
    - **βᵢ** = Coefficient for feature i
    - **Xᵢ** = Standardized value of feature i
    """)
    
    # Intercept
    st.markdown("#### 1️⃣ Intercept (β₀)")
    intercept = model_logistic.intercept_[0]
    st.code(f"β₀ (Intercept) = {intercept:.6f}", language="python")
    
    # Coefficients
    st.markdown("#### 2️⃣ Feature Coefficients (β₁, β₂, ..., βₙ)")
    
    coef_df = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient (β)': model_logistic.coef_[0],
        'Standardized Mean (μ)': scaler_class.mean_,
        'Standardized Std (σ)': scaler_class.scale_
    })
    
    st.dataframe(coef_df.round(6), use_container_width=True)
    
    # Download coefficients
    csv = coef_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Model Parameters (CSV)",
        data=csv,
        file_name="logistic_regression_parameters.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Visual representation of coefficients
    st.subheader("📈 Coefficient Importance")
    
    feature_names_readable = {
        "Materias pasadas ": "Past Courses",
        "Materias nuevas": "Current Courses",
        "Horas de estudio actuales ": "Current Study Hours",
        "Horas estudio pasadas ": "Past Study Hours",
        "Calificaciones pasadas": "Past Grade",
        "eficiencia_estudio_pasado": "Study Efficiency",
        "intensidad_estudio_actual": "Study Intensity",
        "cambio_horas": "Change in Hours",
        "ratio_materias": "Course Ratio",
        "tendencia_academica": "Academic Trend"
    }
    
    coef_importance = pd.DataFrame({
        'Feature': [feature_names_readable.get(col, col) for col in feature_cols],
        'Coefficient': model_logistic.coef_[0]
    }).sort_values('Coefficient', ascending=True)
    
    fig_coef = go.Figure(go.Bar(
        x=coef_importance['Coefficient'],
        y=coef_importance['Feature'],
        orientation='h',
        marker=dict(
            color=coef_importance['Coefficient'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Coefficient Value")
        ),
        text=coef_importance['Coefficient'].round(3),
        textposition='auto'
    ))
    
    fig_coef.update_layout(
        title="Logistic Regression Coefficients",
        xaxis_title="Coefficient Value (β)",
        yaxis_title="Feature",
        height=500
    )
    
    st.plotly_chart(fig_coef, use_container_width=True)
    
    st.markdown("---")
    
    # Example calculation
    st.subheader("🧮 Example: How to Calculate Probability Manually")
    
    with st.expander("Click to see manual calculation example"):
        st.markdown("""
        ### Step-by-step calculation for a new student:
        
        **Given data:**
        - Past Courses (X₁) = 7
        - Current Courses (X₂) = 8
        - Current Study Hours (X₃) = 10
        - Past Study Hours (X₄) = 5
        - Past Grade (X₅) = 9.0
        
        **Step 1:** Calculate derived features (efficiency, intensity, etc.)
        
        **Step 2:** Standardize each feature:
        ```
        X_standardized = (X - μ) / σ
        ```
        
        **Step 3:** Calculate z:
        ```
        z = β₀ + β₁X₁_std + β₂X₂_std + ... + βₙXₙ_std
        ```
        
        **Step 4:** Calculate probability:
        ```
        P(High Performance) = 1 / (1 + e^(-z))
        ```
        
        **Step 5:** Classify:
        - If P ≥ 0.5 → High Performance
        - If P < 0.5 → Below High Performance
        """)
        
        st.code(f"""
# Python code example:
import numpy as np

# Model parameters
intercept = {intercept:.6f}
coefficients = {list(model_logistic.coef_[0].round(6))}
means = {list(scaler_class.mean_.round(6))}
stds = {list(scaler_class.scale_.round(6))}

# New student data (raw values)
new_data = [7, 8, 10, 5, 9.0, ...]  # All 10 features

# Standardize
X_std = [(x - mu) / sigma for x, mu, sigma in zip(new_data, means, stds)]

# Calculate z
z = intercept + sum([coef * x for coef, x in zip(coefficients, X_std)])

# Calculate probability
probability = 1 / (1 + np.exp(-z))

print(f"Probability of High Performance: {{probability*100:.2f}}%")
        """, language="python")
    
    st.markdown("---")
    
    # Model equation
    st.subheader("📝 Complete Model Equation")
    
    st.latex(r"""
    P(\text{High Performance}) = \frac{1}{1 + e^{-z}}
    """)
    
    st.latex(r"""
    \text{where } z = \beta_0 + \sum_{i=1}^{n} \beta_i \cdot \frac{X_i - \mu_i}{\sigma_i}
    """)
    
    st.markdown(f"""
    **With the trained parameters:**
    - β₀ = {intercept:.6f}
    - β₁ to β₁₀ = See table above
    - μ (means) and σ (standard deviations) for standardization = See table above
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎓 Student Grade Prediction System | Logistic Regression Model</p>
    <p>All student data is confidential and anonymized</p>
</div>
""", unsafe_allow_html=True)
