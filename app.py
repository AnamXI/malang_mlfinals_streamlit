import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

model = joblib.load('segmentation_model.joblib')
le_gender = joblib.load('le_gender.joblib')
le_category = joblib.load('le_category.joblib')
features = joblib.load('feature_names.joblib')
importances = joblib.load('feature_importances.joblib')
means = joblib.load('feature_means.joblib')

st.set_page_config(page_title="Spending Score Predictor", layout="wide")
st.title("🪄 Customer Value Projection ")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Demographics")
    age = st.number_input("Age", 18, 100, 38)
    gender = st.selectbox("Gender", options=le_gender.classes_)
    income = st.number_input("Annual Income ($)", min_value=0, value=99000)

with col2:
    st.subheader("Behavior")
    category = st.selectbox("Preferred Category", options=le_category.classes_)
    membership = st.slider("Membership Years", 0, 15, 3)
    frequency = st.slider("Annual Purchase Frequency", 1, 50, 24)

st.divider()

if st.button("Generate Projection", type="primary"):
    input_row = pd.DataFrame([[
        age, le_gender.transform([gender])[0], income, 
        membership, frequency, le_category.transform([category])[0]
    ]], columns=features)
    
    score = model.predict(input_row)[0]
    st.metric(label="Projected Spending Score", value=f"{score:.1f} / 100")

    # Basic Dynamic Explanation for the UI (SURFACE LEVEL ONLY!!!! Just for project demo)
    reasons = []
    if income > means['income']: reasons.append("above-average income")
    if frequency > means['purchase_frequency']: reasons.append("high shopping frequency")
    if membership > means['membership_years']: reasons.append("long-term loyalty")
    
    if reasons:
        explanation = f"The model boosted this score primarily due to the customer's {', '.join(reasons)}."
    else:
        explanation = "The model identified this customer as a baseline profile with standard engagement metrics."

    # For the explanation box
    st.markdown(
        f"""
        <div style="
            background-color: rgba(229, 57, 53, 0.1); 
            padding: 20px; 
            border-radius: 10px; 
            border: 1px solid rgba(229, 57, 53, 0.3);
            border-left: 5px solid #e53935;
            backdrop-filter: blur(5px);
        ">
            <h4 style="color: #e53935; margin-top: 0; display: flex; align-items: center;">
                <span style="margin-right: 10px;">🥽</span> Analysis of Model's Decision:
            </h4>
            <p style="color: #f3f3f3; font-size: 1.1em; margin-bottom: 0;">
                {explanation}
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Prioritization Visualization
st.subheader("🌲 Model Decision Logic 🌲")
importance_df = pd.DataFrame({
    'Feature': [f.replace('_', ' ').title() for f in features],
    'Importance (%)': importances * 100
}).sort_values(by='Importance (%)', ascending=True)

fig = px.bar(
    importance_df, 
    x='Importance (%)', 
    y='Feature', 
    orientation='h',
    template='plotly_dark' 
)

fig.update_traces(marker_color='#009688')
fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))

st.plotly_chart(fig, use_container_width=True)