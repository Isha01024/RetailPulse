# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Import modular backend pipelines
from src.data_processor import DataWarehousePipeline
from src.segmentor import CustomerSegmentor
from src.forecaster import DemandForecaster
from src.churn_predictor import ChurnPredictorEngine
from src.data_synthesizer import generate_mock_data

st.set_page_config(page_title="RetailPulse Engine", layout="wide", page_icon="📊")

# Safety check: Ensure raw files are available before spinning up UI layers
if not os.path.exists('data/raw_sales.csv'):
    generate_mock_data()

@st.cache_data
def load_cached_warehouse_data():
    dw = DataWarehousePipeline()
    return dw.load_and_clean()

# Ingest current operational datasets
sales, customers, inventory = load_cached_warehouse_data()

# Main Application Title & Layout Settings
st.title("📊 RetailPulse: Enterprise AI Customer Analytics & Demand Forecasting Platform")
st.markdown("---")

# Left Sidebar Navigation Panel
st.sidebar.title("Navigation Hub")
sidebar_nav = st.sidebar.radio(
    "Select Interface Portal:", 
    ["Executive Command Center", "Customer Segments (RFM)", "Demand Forecasts & Inventory", "Churn Proactive Radar"]
)

# ---------------------------------------------------------------------------------
# PAGE 1: EXECUTIVE COMMAND CENTER
# ---------------------------------------------------------------------------------
if sidebar_nav == "Executive Command Center":
    st.subheader("💡 Core Organizational KPI Framework")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Gross Revenue Architecture", f"${sales['total_amount'].sum():,.2f}")
    with col2:
        st.metric("Transaction Volume Enrolled", f"{len(sales):,}")
    with col3:
        st.metric("Tracked Active Consumer Base", f"{customers['customer_id'].nunique():,}")
    with col4:
        st.metric("Monitored Inventory Classes", f"{inventory['product_category'].nunique()}")
        
    st.markdown("### 📈 Temporal Revenue Distribution Patterns")
    # Clean string conversion to avoid Period index layout errors inside plotly charts
    monthly_revenue = sales.groupby(sales['date'].dt.to_period('M'))['total_amount'].sum().reset_index()
    monthly_revenue['date'] = monthly_revenue['date'].astype(str)
    
    fig = px.line(monthly_revenue, x='date', y='total_amount', title="Monthly Revenue Growth Trends", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------------
# PAGE 2: CUSTOMER SEGMENTS (RFM)
# ---------------------------------------------------------------------------------
elif sidebar_nav == "Customer Segments (RFM)":
    st.subheader("🎯 Behavioral Cohort Clustering Profiles")
    
    segmentor = CustomerSegmentor()
    rfm_output = segmentor.compute_rfm_segments(sales)
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.scatter(
            rfm_output, x='Recency', y='Monetary', color='Segment Name',
            title="Recency vs Monetary Value Cluster Layout",
            labels={'Recency': 'Recency (Days Since Last Purchase)', 'Monetary': 'Total Monetary Spend ($)'}
        )
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.pie(rfm_output, names='Segment Name', values='Monetary', title='Contribution to Total Revenue Share by Cohort')
        st.plotly_chart(fig2, use_container_width=True)
        
    st.markdown("### Detailed Customer RFM Cohort Log")
    st.dataframe(rfm_output[['Recency', 'Frequency', 'Monetary', 'Segment Name']].head(50), use_container_width=True)

# ---------------------------------------------------------------------------------
# PAGE 3: DEMAND FORECASTS & INVENTORY OPTIMIZATION
# ---------------------------------------------------------------------------------
elif sidebar_nav == "Demand Forecasts & Inventory":
    st.subheader("🔮 Predictive Machine Learning Demand & Advanced Procurement Optimizer")
    
    selected_cat = st.selectbox("Choose Targeted Vertical for Evaluation:", inventory['product_category'].unique())
    
    # Run the background forecasting pipeline
    with st.spinner("Generating 30-day ahead predictive analytics..."):
        forecast_results = DemandForecaster.forecast_category_demand(sales, selected_cat)
    
    fig_fc = px.line(forecast_results, x='ds', y='yhat', title=f"Predicted 30-Day Outbound Volume for {selected_cat}", markers=True)
    st.plotly_chart(fig_fc, use_container_width=True)
    
    # Run operations optimization logic
    cat_inv = inventory[inventory['product_category'] == selected_cat].iloc[0]
    optimization_metrics = DemandForecaster.optimize_inventory(
        forecast_results['yhat'], 
        cat_inv['current_stock_level'], 
        cat_inv['reorder_point'], 
        cat_inv['lead_time_days']
    )
    
    st.markdown("### 📦 Strategic Inventory Optimization Recommendations")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("30-Day Aggregated Demand Projection", optimization_metrics["Projected Demand (30D)"])
    with c2:
        st.metric("Target Suggested Order Quantity", optimization_metrics["Suggested Procurement"])
    with c3:
        st.metric("Current Operational Status Level", str(optimization_metrics["Stock Evaluation Status"]))

# ---------------------------------------------------------------------------------
# PAGE 4: CHURN PROACTIVE RADAR
# ---------------------------------------------------------------------------------
elif sidebar_nav == "Churn Proactive Radar":
    st.subheader("🚨 Customer Defection Vulnerability Monitoring Matrix")
    
    predictor = ChurnPredictorEngine()
    
    with st.spinner("Training predictive XGBoost model pipeline..."):
        train_log = predictor.train_pipeline(sales)
    st.success(train_log)
    
    risk_matrix = predictor.extract_risk_profiles(sales)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### High Risk Profiles Action Ledger")
        st.dataframe(risk_matrix.head(100), use_container_width=True)
    with col2:
        st.markdown("#### Risk Distribution Breakdown")
        fig_risk = px.histogram(risk_matrix, x='Churn Risk Probability', nbins=20, title='Customer Distribution by Churn Risk Probability')
        st.plotly_chart(fig_risk, use_container_width=True)
