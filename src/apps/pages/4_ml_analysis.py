# src/pages/07_ml_analysis.py
"""
Machine Learning Analysis Page
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.models.ml_features import FeatureEngineer
from src.models.ml_regimes import RegimeDetector
from src.models.ml_predictions import SpreadPredictor
from src.models.database import CDSDatabase

def run():
    st.title("🤖 ML Analysis & Predictions")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Analysis Settings")
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Market Regime", "Spread Prediction", "Statistical Relationships"]
        )
        
        index_name = st.selectbox(
            "Index",
            ["EU_IG", "EU_XO", "US_IG", "US_HY"]
        )
        
        tenor = st.selectbox("Tenor", ["3Y", "5Y", "7Y", "10Y"])
    
    # Load data
    db = CDSDatabase("data/raw/cds_indices_raw.db")
    spread_data = db.query_historical_spreads(index_name, tenor, '2020-01-01')
    
    if spread_data.empty:
        st.error("No data available")
        return
    
    # Feature engineering
    feature_eng = FeatureEngineer()
    features = feature_eng.create_spread_features(spread_data)
    
    # Analysis based on type
    if analysis_type == "Market Regime":
        show_regime_analysis(features, spread_data)
    
    elif analysis_type == "Spread Prediction":
        show_prediction_analysis(features, spread_data)
    
    elif analysis_type == "Statistical Relationships":
        show_statistical_analysis(spread_data, db)

def show_regime_analysis(features, spread_data):
    """Display regime detection results"""
    st.header("Market Regime Detection")
    
    # Detect regimes
    detector = RegimeDetector()
    regimes = detector.fit_predict(features.dropna())
    
    # Current regime
    current_regime = regimes.iloc[-1]
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Regime", current_regime)
    
    # Regime statistics
    regime_stats = detector.get_regime_statistics(features, regimes)
    st.dataframe(regime_stats)
    
    # Visualization
    fig = go.Figure()
    
    # Add spread line
    fig.add_trace(go.Scatter(
        x=spread_data['date'],
        y=spread_data['spread_bps'],
        name='Spread',
        line=dict(color='blue')
    ))
    
    # Color background by regime
    # (Implementation of regime coloring)
    
    st.plotly_chart(fig, use_container_width=True)

def show_prediction_analysis(features, spread_data):
    """Display prediction results"""
    st.header("Spread Direction Prediction")
    
    # Train predictor
    predictor = SpreadPredictor()
    labels = predictor.create_labels(spread_data['spread_bps'])
    
    # Prepare training data
    X = features.dropna()
    y = labels[X.index].dropna()
    X = X[y.index]
    
    # Train model
    results = predictor.train(X, y)
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CV Accuracy", f"{results['cv_score_mean']:.2%}")
    with col2:
        st.metric("Std Dev", f"{results['cv_score_std']:.2%}")
    
    # Feature importance
    st.subheader("Feature Importance")
    st.bar_chart(results['feature_importance'].head(10))
    
    # Current prediction
    if len(X) > 0:
        current_features = X.iloc[[-1]]
        probs = predictor.predict_proba(current_features)
        
        st.subheader("Current Prediction")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("P(Tighten)", f"{probs['prob_tighten'].iloc[0]:.2%}")
        with col2:
            st.metric("P(Flat)", f"{probs['prob_flat'].iloc[0]:.2%}")
        with col3:
            st.metric("P(Widen)", f"{probs['prob_widen'].iloc[0]:.2%}")