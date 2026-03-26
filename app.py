import streamlit as st
import pandas as pd
import joblib

model = joblib.load('segmentation_model.joblib')
le_gender = joblib.load('le_gender.joblib')
le_category = joblib.load('le_category.joblib')
features = joblib.load('feature_names.joblib')
importances = joblib.load('feature_importances.joblib')
means = joblib.load('feature_means.joblib')

st.set_page_config(page_title="Spending Score Predictor", layout="wide")
st.title("🥽 Customer Value Projection 🪄")

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

    # Basic Dynamic Explanations (General Reasons Only)
    st.info("📊 **Basic Analysis of Model's Decision:**")
    
    reasons = []
    if income > means['income']: reasons.append("above-average income")
    if frequency > means['purchase_frequency']: reasons.append("high shopping frequency")
    if membership > means['membership_years']: reasons.append("long-term loyalty")
    
    if reasons:
        st.write(f"The model boosted this score primarily due to the customer's {', '.join(reasons)}.")
    else:
        st.write("The model identified this customer as a baseline profile with standard engagement metrics.")

# Prioritization Visualization
st.subheader("🌲 Model Decision Logic 🌲")
importance_df = pd.DataFrame({
    'Feature': [f.replace('_', ' ').title() for f in features],
    'Importance (%)': importances * 100
}).sort_values(by='Importance (%)', ascending=True)

st.bar_chart(importance_df.set_index('Feature'))