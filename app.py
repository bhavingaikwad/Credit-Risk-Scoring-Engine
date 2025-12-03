import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import sqlite3

# --- PAGE CONFIG ---
st.set_page_config(page_title="Credit Risk Scorer", page_icon="🏦", layout="wide")

# --- 1. LOAD THE SAVED MODEL ---
@st.cache_resource
def load_model():
    with open('data/credit_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- 2. HEADER ---
st.title("🏦 Credit Risk Scoring Engine")
st.markdown("Prototype for **M&T Bank** | Predicting probability of loan default.")

# --- 3. SIDEBAR (User Inputs) ---
st.sidebar.header("📝 Loan Application")
st.sidebar.markdown("Enter applicant details below:")

# Important: These inputs must match the order your model expects!
# [Age, Credit amount, Duration]
age = st.sidebar.number_input("Applicant Age", min_value=18, max_value=100, value=30)
amount = st.sidebar.number_input("Credit Amount ($)", min_value=500, max_value=20000, value=5000)
duration = st.sidebar.slider("Loan Duration (Months)", min_value=6, max_value=72, value=24)

# --- 4. PREDICTION LOGIC ---
if st.sidebar.button("Calculate Risk Score"):
    # Create a DataFrame for the model
    input_data = pd.DataFrame([[age, amount, duration]], 
                              columns=['Age', 'Credit amount', 'Duration'])
    
    # Get Prediction
    prediction = model.predict(input_data)[0] # 0 or 1
    probability = model.predict_proba(input_data)[0][1] # % chance of default
    
    # Display Results
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Assessment")
        if prediction == 1: # Bad Loan
            st.error(f"⚠️ **HIGH RISK**: Application Recommended for Rejection")
            st.metric("Default Probability", f"{probability:.1%}", delta="-High Risk", delta_color="inverse")
        else: # Good Loan
            st.success(f"✅ **LOW RISK**: Application Recommended for Approval")
            st.metric("Default Probability", f"{probability:.1%}", delta="Safe", delta_color="normal")

# --- 5. ANALYST DASHBOARD (The "Data Analyst" View) ---
st.divider()
st.subheader("📊 Portfolio Analysis (Data Analyst View)")

# Load data from SQL for charts
conn = sqlite3.connect('data/credit_risk.db')
df_sql = pd.read_sql("SELECT * FROM loans", conn)
conn.close()

# Create 2 Columns for Charts
chart1, chart2 = st.columns(2)

with chart1:
    st.markdown("**Loan Amounts by Purpose**")
    # Interactive Bar Chart
    fig = px.box(df_sql, x="Purpose", y="Credit amount", color="Purpose")
    st.plotly_chart(fig, use_container_width=True)

with chart2:
    st.markdown("**Risk Distribution by Age**")
    # Histogram
    fig2 = px.histogram(df_sql, x="Age", color="Risk", barmode="overlay")
    st.plotly_chart(fig2, use_container_width=True)