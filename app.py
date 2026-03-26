import streamlit as st
import pandas as pd
import joblib

# Load trained components
model = joblib.load('segmentation_model.joblib')
le_gender = joblib.load('le_gender.joblib')
le_category = joblib.load('le_category.joblib')
features = joblib.load('feature_names.joblib')
importances = joblib.load('feature_importances.joblib')

st.set_page_config(page_title="Customer Insights AI", layout="wide")

st.title("🎯 Customer Value Projection")
st.write("Input profile data to project the likely **Spending Score**.")

# Layout with two columns
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
        age,
        le_gender.transform([gender])[0],
        income,
        membership,
        frequency,
        le_category.transform([category])[0]
    ]], columns=features)
    
    score = model.predict(input_row)[0]
    st.metric(label="Projected Spending Score", value=f"{score:.1f} / 100")
    
    if score >= 70:
        st.success("Target: High-Value Customer")
    elif score >= 40:
        st.info("Target: Standard Customer")
    else:
        st.warning("Target: Low-Engagement Customer")

# --- Transparency/Priority Section ---
st.subheader("🤖 Model Decision Logic")
st.write("This chart shows which properties the AI values most when making its projection.")

importance_df = pd.DataFrame({
    'Feature': [f.replace('_', ' ').title() for f in features],
    'Importance (%)': importances * 100
}).sort_values(by='Importance (%)', ascending=True)

st.bar_chart(importance_df.set_index('Feature'))

st.caption("Note: If a higher income results in a lower score, the AI has detected that higher income earners in this dataset tend to have lower engagement patterns compared to other groups.")