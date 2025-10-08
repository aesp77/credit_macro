# src/apps/pages/04_ml_analysis.py
"""
Machine Learning Analysis Page
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

# Fix the import paths
current_dir = os.path.dirname(os.path.abspath(__file__))  # pages directory
apps_dir = os.path.dirname(current_dir)  # apps directory
src_dir = os.path.dirname(apps_dir)  # src directory
root_dir = os.path.dirname(src_dir)  # root directory

sys.path.insert(0, src_dir)

# Now import from models
from models.ml_features import FeatureEngineer
from models.ml_regimes import RegimeDetector
from models.ml_predictions import SpreadPredictor
from models.database import CDSDatabase

def run():
    st.title("ML Analysis & Predictions")
    
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
    
    # Load data - use absolute path from root directory
    db_path = os.path.join(root_dir, "data", "raw", "cds_indices_raw.db")
    db = CDSDatabase(db_path)
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
    
    # Prepare training data - ensure alignment
    X = features.dropna()
    
    # Find common indices between X and labels
    common_idx = X.index.intersection(labels.index)
    
    if len(common_idx) == 0:
        st.error("No common indices between features and labels")
        return
    
    # Align both to common indices
    X = X.loc[common_idx]
    y = labels.loc[common_idx]
    
    # Remove any remaining NaNs
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    if len(X) < 50:  # Need minimum samples for training
        st.warning(f"Insufficient data for training: {len(X)} samples")
        return
    
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
            
def show_statistical_analysis(spread_data, db):
    """Display statistical relationships"""
    import numpy as np  # Add import here if not at top of file
    st.header("Statistical Relationships")
    
    # Basic statistics
    st.subheader("Spread Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{spread_data['spread_bps'].mean():.1f} bps")
    with col2:
        st.metric("Std Dev", f"{spread_data['spread_bps'].std():.1f} bps")
    with col3:
        st.metric("Min", f"{spread_data['spread_bps'].min():.1f} bps")
    with col4:
        st.metric("Max", f"{spread_data['spread_bps'].max():.1f} bps")
    
    # Calculate rolling statistics
    st.subheader("Rolling Statistics")
    
    window = st.slider("Rolling Window (days)", 5, 60, 20)
    
    spread_data_sorted = spread_data.sort_values('date')
    spread_data_sorted['rolling_mean'] = spread_data_sorted['spread_bps'].rolling(window).mean()
    spread_data_sorted['rolling_std'] = spread_data_sorted['spread_bps'].rolling(window).std()
    
    # Plot rolling statistics
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=spread_data_sorted['date'],
        y=spread_data_sorted['spread_bps'],
        name='Spread',
        line=dict(color='blue', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=spread_data_sorted['date'],
        y=spread_data_sorted['rolling_mean'],
        name=f'{window}D MA',
        line=dict(color='red', width=2)
    ))
    
    # Add confidence bands
    upper_band = spread_data_sorted['rolling_mean'] + 2 * spread_data_sorted['rolling_std']
    lower_band = spread_data_sorted['rolling_mean'] - 2 * spread_data_sorted['rolling_std']
    
    fig.add_trace(go.Scatter(
        x=spread_data_sorted['date'],
        y=upper_band,
        name='Upper Band (2σ)',
        line=dict(color='gray', dash='dash'),
        opacity=0.5
    ))
    
    fig.add_trace(go.Scatter(
        x=spread_data_sorted['date'],
        y=lower_band,
        name='Lower Band (2σ)',
        line=dict(color='gray', dash='dash'),
        opacity=0.5
    ))
    
    fig.update_layout(
        title="Spread with Rolling Statistics",
        xaxis_title="Date",
        yaxis_title="Spread (bps)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution analysis
    st.subheader("Spread Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=spread_data['spread_bps'],
            nbinsx=30,
            name='Distribution'
        ))
        fig_hist.update_layout(
            title="Spread Distribution",
            xaxis_title="Spread (bps)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Q-Q plot
        try:
            from scipy import stats
            # Get the Q-Q plot data
            probplot_data = stats.probplot(spread_data['spread_bps'], dist="norm")
            theoretical_quantiles = probplot_data[0][0]
            sample_quantiles = probplot_data[0][1]
            
            # Calculate the reference line (45-degree line for perfect normal distribution)
            # Using the slope and intercept from probplot
            slope = probplot_data[1][0]
            intercept = probplot_data[1][1]
            
            # Create reference line points
            line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
            line_y = slope * line_x + intercept
            
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color='blue', size=5)
            ))
            fig_qq.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode='lines',
                name='Normal Reference',
                line=dict(color='red', dash='dash')
            ))
            fig_qq.update_layout(
                title="Q-Q Plot",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                showlegend=True
            )
            st.plotly_chart(fig_qq, use_container_width=True)
            
            # Q-Q Plot Interpretation
            with st.expander("Q-Q Plot Interpretation"):
                # Calculate deviations at the tails
                tail_threshold = len(theoretical_quantiles) // 10  # Look at 10% tails
                
                # Left tail (lower values)
                left_tail_deviation = np.mean(sample_quantiles[:tail_threshold] - 
                                             (slope * theoretical_quantiles[:tail_threshold] + intercept))
                
                # Right tail (higher values)  
                right_tail_deviation = np.mean(sample_quantiles[-tail_threshold:] - 
                                              (slope * theoretical_quantiles[-tail_threshold:] + intercept))
                
                st.markdown("**Distribution Characteristics:**")
                
                # Overall shape interpretation
                if abs(left_tail_deviation) < 2 and abs(right_tail_deviation) < 2:
                    st.info("**Near-Normal Distribution**: Data points closely follow the reference line, suggesting the spread distribution is approximately normal.")
                else:
                    deviations = []
                    
                    # Left tail interpretation
                    if left_tail_deviation < -2:
                        deviations.append("• **Thin left tail**: Lower spreads occur less frequently than expected under normality")
                    elif left_tail_deviation > 2:
                        deviations.append("• **Fat left tail**: Lower spreads occur more frequently than expected (potential for extreme tightening)")
                    
                    # Right tail interpretation
                    if right_tail_deviation < -2:
                        deviations.append("• **Thin right tail**: Higher spreads occur less frequently than expected under normality")
                    elif right_tail_deviation > 2:
                        deviations.append("• **Fat right tail**: Higher spreads occur more frequently than expected (potential for extreme widening)")
                    
                    if deviations:
                        st.warning("**Non-Normal Distribution Detected:**\n" + "\n".join(deviations))
                
                # S-shape or other patterns
                middle_points = len(theoretical_quantiles) // 2
                middle_deviation = np.mean(sample_quantiles[middle_points-tail_threshold:middle_points+tail_threshold] - 
                                          (slope * theoretical_quantiles[middle_points-tail_threshold:middle_points+tail_threshold] + intercept))
                
                if abs(middle_deviation) > 2:
                    if middle_deviation > 0:
                        st.info("**Positive skew detected**: Distribution has a longer right tail")
                    else:
                        st.info("**Negative skew detected**: Distribution has a longer left tail")
                
                # Risk implications
                st.markdown("**Risk Implications:**")
                if right_tail_deviation > 3:
                    st.error("**Tail Risk Alert**: Fat right tail suggests higher probability of extreme spread widening events than normal distribution would predict. Consider tail risk hedging.")
                elif right_tail_deviation > 2:
                    st.warning("⚡ **Moderate Tail Risk**: Some evidence of tail risk in spread widening scenarios.")
                else:
                    st.success("✅ **Limited Tail Risk**: Right tail behavior is consistent with or thinner than normal distribution.")
        except ImportError:
            st.info("Install scipy for Q-Q plot: pip install scipy")
    
    
    
if __name__ == "__main__":
    run()