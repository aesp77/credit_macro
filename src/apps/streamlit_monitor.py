# Streamlit CDS Monitor Interface
# Monitor and analyze CDS strategies

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from migrated modules
from src.models.trs import TRSDatabaseBuilder, calculate_cds_total_return
from src.models.strategy_calc import calculate_steepener_pnl, calculate_generic_strategy_metrics

# Page configuration
st.set_page_config(
    page_title="CDS Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database paths
TRS_DB_PATH = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\processed\cds_trs.db"

def load_trs_database():
    """Load TRS database connection"""
    try:
        conn = sqlite3.connect(TRS_DB_PATH, check_same_thread=False)
        return conn
    except:
        st.error(f"Cannot connect to TRS database at {TRS_DB_PATH}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_series():
    """Get list of available indices and tenors"""
    conn = load_trs_database()
    if not conn:
        return [], []
    
    cursor = conn.cursor()
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'trs_%'").fetchall()
    
    indices = set()
    tenors = set()
    
    for table in tables:
        parts = table[0].replace('trs_', '').split('_')
        if len(parts) >= 2:
            index = '_'.join(parts[:-1]).upper()
            tenor = parts[-1].upper()
            indices.add(index)
            tenors.add(tenor)
    
    return sorted(list(indices)), sorted(list(tenors))

def load_trs_data(index: str, tenor: str, start_date: str = None, end_date: str = None):
    """Load TRS data for specific index/tenor"""
    conn = load_trs_database()
    if not conn:
        return pd.DataFrame()
    
    table_name = f"trs_{index.lower()}_{tenor.lower()}"
    
    query = f"SELECT * FROM {table_name}"
    conditions = []
    
    if start_date:
        conditions.append(f"date >= '{start_date}'")
    if end_date:
        conditions.append(f"date <= '{end_date}'")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY date"
    
    try:
        df = pd.read_sql_query(query, conn)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except:
        return pd.DataFrame()

def calculate_strategy_pnl(legs: list, start_date: str, end_date: str):
    """
    Calculate strategy P&L from multiple legs
    
    legs: list of dicts with keys: index, tenor, side, weight
    """
    combined_df = None
    
    for i, leg in enumerate(legs):
        trs_data = load_trs_data(leg['index'], leg['tenor'], start_date, end_date)
        
        if trs_data.empty:
            st.warning(f"No data for {leg['index']} {leg['tenor']}")
            continue
        
        # Select appropriate TRS column based on side
        trs_col = 'long_tr' if leg['side'] == 'Long' else 'short_tr'
        pnl_col = 'long_pnl' if leg['side'] == 'Long' else 'short_pnl'
        
        # Normalize to 100 at start
        initial_value = trs_data[trs_col].iloc[0]
        trs_data[f'norm_tr_{i}'] = (trs_data[trs_col] / initial_value) * 100
        
        # Apply weight
        trs_data[f'weighted_tr_{i}'] = (trs_data[f'norm_tr_{i}'] - 100) * leg['weight'] / 100
        
        # Also get weighted P&L
        initial_pnl = trs_data[pnl_col].iloc[0]
        trs_data[f'weighted_pnl_{i}'] = (trs_data[pnl_col] - initial_pnl) * leg['weight'] / 100
        
        # Store leg info
        trs_data[f'spread_{i}'] = trs_data['spread_bps']
        
        if combined_df is None:
            combined_df = trs_data[['date', f'weighted_tr_{i}', f'weighted_pnl_{i}', f'spread_{i}']].copy()
        else:
            combined_df = pd.merge(combined_df, 
                                 trs_data[['date', f'weighted_tr_{i}', f'weighted_pnl_{i}', f'spread_{i}']], 
                                 on='date')
    
    if combined_df is not None:
        # Calculate combined strategy performance
        weighted_tr_cols = [col for col in combined_df.columns if 'weighted_tr' in col]
        weighted_pnl_cols = [col for col in combined_df.columns if 'weighted_pnl' in col]
        
        combined_df['strategy_return'] = combined_df[weighted_tr_cols].sum(axis=1)
        combined_df['strategy_tr'] = 100 + combined_df['strategy_return']
        combined_df['strategy_pnl'] = combined_df[weighted_pnl_cols].sum(axis=1)
        combined_df['daily_pnl'] = combined_df['strategy_pnl'].diff()
        
    return combined_df

# MAIN APP
def main():
    st.title("CDS Strategy Monitor")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Strategy Configuration")
        
        # Get available indices and tenors
        indices, tenors = get_available_series()
        
        if not indices:
            st.error("No data available. Please build TRS database first.")
            return
        
        # Strategy type selection
        strategy_type = st.selectbox(
            "Strategy Type",
            ["Curve Trade (5s10s)", "Compression Trade", "Custom Multi-Leg"]
        )
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", 
                                      value=datetime.now() - timedelta(days=365))
        with col2:
            end_date = st.date_input("End Date", 
                                    value=datetime.now())
        
        st.divider()
        
        # Configure legs based on strategy type
        legs = []
        
        if strategy_type == "Curve Trade (5s10s)":
            index = st.selectbox("Index", indices)
            trade_type = st.radio("Trade Type", ["Steepener", "Flattener"])
            weight_method = st.selectbox("Weighting", ["DV01 Neutral", "Equal", "Beta"])
            
            # DV01 neutral approximation
            if weight_method == "DV01 Neutral":
                weight_5y = 100
                weight_10y = 66.7  # 10Y has ~1.5x DV01 of 5Y
            elif weight_method == "Beta":
                # Would calculate from historical data
                weight_5y = 100
                weight_10y = 50  # Placeholder
            else:
                weight_5y = 100
                weight_10y = 100
            
            if trade_type == "Steepener":
                legs = [
                    {'index': index, 'tenor': '5Y', 'side': 'Short', 'weight': weight_5y},
                    {'index': index, 'tenor': '10Y', 'side': 'Long', 'weight': weight_10y}
                ]
            else:  # Flattener
                legs = [
                    {'index': index, 'tenor': '5Y', 'side': 'Long', 'weight': weight_5y},
                    {'index': index, 'tenor': '10Y', 'side': 'Short', 'weight': weight_10y}
                ]
        
        elif strategy_type == "Compression Trade":
            # Get indices by type
            xo_indices = [idx for idx in indices if 'XO' in idx]
            hy_indices = [idx for idx in indices if 'HY' in idx]
            ig_indices = [idx for idx in indices if 'IG' in idx]
            
            high_spread_indices = xo_indices + hy_indices
            low_spread_indices = ig_indices
            
            if not high_spread_indices or not low_spread_indices:
                st.error("Insufficient indices for compression trade")
                return
            
            long_index = st.selectbox("Long Index (High Spread)", high_spread_indices)
            short_index = st.selectbox("Short Index (Low Spread)", low_spread_indices)
            tenor = st.selectbox("Tenor", ['5Y'])
            weight_method = st.selectbox("Weighting", ["Beta", "DV01 Neutral", "Equal"])
            
            # Weight calculation
            if weight_method == "Beta":
                weight_short = 400  # Default 4x beta assumption
            elif weight_method == "DV01 Neutral":
                weight_short = 100  # Simplified
            else:
                weight_short = 100
            
            legs = [
                {'index': long_index, 'tenor': tenor, 'side': 'Long', 'weight': 100},
                {'index': short_index, 'tenor': tenor, 'side': 'Short', 'weight': weight_short}
            ]
        
        else:  # Custom Multi-Leg
            n_legs = st.number_input("Number of Legs", min_value=1, max_value=4, value=2)
            
            # Add weighting method selection FIRST
            weight_method = st.selectbox("Weighting Method", ["Equal", "DV01 Neutral", "Beta", "Custom"])
            
            leg_configs = []
            for i in range(n_legs):
                st.subheader(f"Leg {i+1}")
                col1, col2 = st.columns(2)
                with col1:
                    index = st.selectbox(f"Index", indices, key=f"idx_{i}")
                    tenor = st.selectbox(f"Tenor", tenors, key=f"tenor_{i}")
                with col2:
                    side = st.selectbox(f"Side", ["Long", "Short"], key=f"side_{i}")
                    
                    # Only show manual weight input if Custom weighting
                    if weight_method == "Custom":
                        weight = st.number_input(f"Weight %", value=100.0, key=f"weight_{i}")
                    else:
                        weight = None  # Will be calculated
                
                leg_configs.append({
                    'index': index,
                    'tenor': tenor,
                    'side': side,
                    'manual_weight': weight
                })
            
            # Calculate weights based on method
            if weight_method == "Equal":
                for config in leg_configs:
                    config['weight'] = 100.0
            
            elif weight_method == "DV01 Neutral":
                # Use tenor-based DV01 approximations
                tenor_dv01_map = {'3Y': 0.6, '5Y': 1.0, '7Y': 1.35, '10Y': 1.5}
                base_dv01 = tenor_dv01_map.get(leg_configs[0]['tenor'], 1.0)
                
                for i, config in enumerate(leg_configs):
                    if i == 0:
                        config['weight'] = 100.0
                    else:
                        leg_dv01 = tenor_dv01_map.get(config['tenor'], 1.0)
                        config['weight'] = 100.0 * (base_dv01 / leg_dv01)
            
            elif weight_method == "Beta":
                # Beta adjustment - inverse relationship for volatility
                for i, config in enumerate(leg_configs):
                    if i == 0:
                        config['weight'] = 100.0
                    else:
                        # Higher volatility indices need LESS weight
                        # XO/HY are ~4x more volatile than IG, so use 1/4 weight
                        if ('XO' in config['index'] or 'HY' in config['index']) and ('IG' in leg_configs[0]['index']):
                            config['weight'] = 25.0  # 1/4x for high vol short vs low vol long
                        elif ('IG' in config['index']) and ('XO' in leg_configs[0]['index'] or 'HY' in leg_configs[0]['index']):
                            config['weight'] = 400.0  # 4x for low vol short vs high vol long
                        else:
                            config['weight'] = 100.0
            
            else:  # Custom
                for config in leg_configs:
                    config['weight'] = config.get('manual_weight', 100.0)
            
            # Build legs list
            legs = []
            for config in leg_configs:
                legs.append({
                    'index': config['index'],
                    'tenor': config['tenor'],
                    'side': config['side'],
                    'weight': config['weight']
                })
            
            # Display calculated weights
            st.subheader("Calculated Weights")
            weights_df = pd.DataFrame([
                {
                    'Leg': i+1,
                    'Index': leg['index'],
                    'Tenor': leg['tenor'],
                    'Side': leg['side'],
                    'Weight': f"{leg['weight']:.1f}%"
                }
                for i, leg in enumerate(legs)
            ])
            st.dataframe(weights_df, hide_index=True)
        
        st.divider()
        
        # Run strategy button
        run_strategy = st.button("Calculate Strategy", type="primary")
    
    # Main content area
    if run_strategy and legs:
        # Calculate strategy P&L
        strategy_data = calculate_strategy_pnl(legs, 
                                              start_date.strftime('%Y-%m-%d'),
                                              end_date.strftime('%Y-%m-%d'))
        
        if strategy_data is not None and not strategy_data.empty:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_pnl = strategy_data['strategy_pnl'].iloc[-1]
            total_return = strategy_data['strategy_return'].iloc[-1]
            
            # Calculate Sharpe
            if len(strategy_data) > 1:
                daily_returns = strategy_data['strategy_pnl'].diff() / 10_000_000
                sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
            else:
                sharpe = 0
            
            # Max drawdown
            running_max = strategy_data['strategy_pnl'].cummax()
            drawdown = strategy_data['strategy_pnl'] - running_max
            max_dd = drawdown.min()
            
            with col1:
                st.metric("Total P&L", f"${total_pnl:,.0f}", 
                         f"{total_return:+.2f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            with col3:
                st.metric("Max Drawdown", f"${max_dd:,.0f}")
            with col4:
                win_rate = (strategy_data['daily_pnl'] > 0).mean()
                st.metric("Win Rate", f"{win_rate:.1%}")
            
            # Charts
            tab1, tab2, tab3 = st.tabs(["Performance", "Spreads", "Analysis"])
            
            with tab1:
                # Performance chart
                fig = make_subplots(rows=2, cols=1, 
                                  subplot_titles=("Cumulative P&L", "Daily P&L"),
                                  row_heights=[0.7, 0.3])
                
                # Cumulative P&L
                fig.add_trace(
                    go.Scatter(x=strategy_data['date'], 
                             y=strategy_data['strategy_pnl'],
                             name='Cumulative P&L',
                             line=dict(color='purple', width=2)),
                    row=1, col=1
                )
                
                # Daily P&L bars
                colors = ['green' if x > 0 else 'red' for x in strategy_data['daily_pnl'].fillna(0)]
                fig.add_trace(
                    go.Bar(x=strategy_data['date'], 
                          y=strategy_data['daily_pnl'],
                          name='Daily P&L',
                          marker_color=colors,
                          showlegend=False),
                    row=2, col=1
                )
                
                fig.update_layout(height=700, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Spread levels
                fig = go.Figure()
                
                spread_cols = [col for col in strategy_data.columns if col.startswith('spread_')]
                for i, col in enumerate(spread_cols):
                    fig.add_trace(go.Scatter(
                        x=strategy_data['date'],
                        y=strategy_data[col],
                        name=f"{legs[i]['index']} {legs[i]['tenor']}",
                        mode='lines'
                    ))
                
                fig.update_layout(
                    title="Spread Levels",
                    yaxis_title="Spread (bps)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Summary table
                st.subheader("Strategy Summary")
                
                # Create summary dataframe
                summary_df = pd.DataFrame({
                    'Metric': ['Total Return (%)', 'Total P&L ($)', 'Sharpe Ratio', 
                              'Max Drawdown ($)', 'Win Rate (%)', 'Trading Days'],
                    'Value': [f"{total_return:.2f}", f"{total_pnl:,.0f}", f"{sharpe:.2f}",
                             f"{max_dd:,.0f}", f"{win_rate*100:.1f}", len(strategy_data)]
                })
                st.dataframe(summary_df, hide_index=True)
                
                st.subheader("Leg Configuration")
                leg_df = pd.DataFrame(legs)
                st.dataframe(leg_df, hide_index=True)
            
            # Download data
            st.divider()
            csv = strategy_data.to_csv(index=False)
            st.download_button(
                label="Download Strategy Data",
                data=csv,
                file_name=f"strategy_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.error("No data available for selected parameters")
    
    else:
        # Show instructions
        st.info("""
        Configure your strategy in the sidebar:
        
        **Curve Trade**: 5s10s steepener or flattener
        **Compression Trade**: Long high-spread vs short low-spread index  
        **Custom Multi-Leg**: Build any combination of legs
        
        Select weighting method and date range, then click Calculate Strategy.
        """)

if __name__ == "__main__":
    main()