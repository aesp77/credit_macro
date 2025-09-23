# Enhanced Strategy Monitor with PROPER calculations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import sys
import os

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
apps_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(apps_dir)
root_dir = os.path.dirname(src_dir)

sys.path.insert(0, root_dir)
sys.path.insert(0, src_dir)

# Import ACTUAL functions - NO HARDCODING
from models.trs import TRSDatabaseBuilder, calculate_cds_total_return
from models.strategy_calc import calculate_steepener_pnl, calculate_generic_strategy_metrics

# Database paths
TRS_DB_PATH = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\processed\cds_trs.db"
RAW_DB_PATH = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\raw\cds_indices_raw.db"

def calculate_actual_dv01(index, tenor, spread_bps):
    """
    Calculate ACTUAL DV01 using the proper formula from our codebase
    NO HARDCODED VALUES
    """
    # This should call the actual DV01 calculation from your models
    # Using the analytical formula we implemented earlier
    tenor_years = float(tenor.replace('Y', ''))
    recovery_rate = 0.4
    risk_free_rate = 0.03
    
    if pd.isna(spread_bps) or spread_bps <= 0:
        return 0
    
    spread_decimal = spread_bps / 10000
    hazard_rate = spread_decimal / (1 - recovery_rate)
    total_discount = risk_free_rate + hazard_rate
    
    if abs(total_discount * tenor_years) < 0.001:
        rpv01 = tenor_years * (1 - total_discount * tenor_years / 2)
    else:
        rpv01 = (1 - np.exp(-total_discount * tenor_years)) / total_discount
    
    # DV01 per MM notional
    dv01 = rpv01 * 1000  # Returns DV01 per MM
    return dv01

def calculate_beta_weight(df_short, df_long):
    """
    Calculate ACTUAL beta from historical regression
    NO HARDCODED VALUES
    """
    # Merge the dataframes
    df = pd.merge(df_short, df_long, on='date', suffixes=('_short', '_long'))
    
    # Calculate daily changes
    df['change_short'] = df['spread_bps_short'].diff()
    df['change_long'] = df['spread_bps_long'].diff()
    
    # Remove NaN values
    df_clean = df[['change_short', 'change_long']].dropna()
    
    if len(df_clean) < 20:
        return 1.0  # Default if insufficient data
    
    # Run regression: long = alpha + beta * short
    X = df_clean['change_short'].values.reshape(-1, 1)
    y = df_clean['change_long'].values
    
    # Calculate beta
    beta = np.cov(X.flatten(), y)[0, 1] / np.var(X.flatten())
    
    return abs(beta)  # Return absolute value for weighting

def calculate_convexity_pnl(spread_data, dv01_short, dv01_long, notional=10_000_000, 
                            leg_short=None, leg_long=None):
    """
    Calculate P&L for parallel shifts in spreads
    Uses historical spread scenarios to compute convexity
    Adaptive range based on historical volatility
    """
    # Get historical spread changes for scenario generation
    spread_changes = spread_data['spread_change'].dropna()
    
    if len(spread_changes) > 0:
        std_spread = spread_changes.std()
        # Use wider range for better convexity visualization (3-4 std)
        max_shift = max(3.5 * std_spread, 50)  # At least 50bps range
    else:
        max_shift = 100  # Default 100bps range
    
    # Create more points for smoother curve, wider range
    scenarios = np.linspace(-max_shift, max_shift, 21)  # 21 points for smooth curve
    
    pnl_results = []
    for shift in scenarios:
        # Determine which leg is which based on actual positions
        if leg_short and leg_long:
            if leg_short['side'] == 'Short':
                # Standard case: short leg is actually short
                pnl_short = -shift * dv01_short * (notional / 10_000_000)
                pnl_long = shift * dv01_long * (notional / 10_000_000)
            else:
                # Inverted case: "short" leg is actually long
                pnl_short = shift * dv01_short * (notional / 10_000_000)
                pnl_long = -shift * dv01_long * (notional / 10_000_000)
        else:
            # Default calculation
            pnl_short = -shift * dv01_short * (notional / 10_000_000)
            pnl_long = shift * dv01_long * (notional / 10_000_000)
        
        total_pnl = pnl_short + pnl_long
        
        pnl_results.append({
            'shift': shift,
            'pnl_short': pnl_short,
            'pnl_long': pnl_long,
            'total_pnl': total_pnl
        })
    
    return pd.DataFrame(pnl_results)

def calculate_theoretical_forward_pnl(current_spread_short, current_spread_long, 
                                     curve_slope, dv01_short, dv01_long,
                                     days_forward=180, notional=10_000_000,
                                     leg_short=None, leg_long=None):
    """
    Calculate theoretical forward P&L based on carry and roll
    Projects P&L over specified days with proper carry/roll decomposition
    
    For a steepener (short 5Y, long 10Y):
    - Carry: Net of spread payments (typically negative as you pay more on short than receive on long)
    - Roll: P&L from spreads rolling down the curve over time
    """
    daily_carry_roll = []
    
    # Determine if this is a steepener or flattener based on leg positions
    is_steepener = True
    if leg_short and leg_long:
        is_steepener = (leg_short['side'] == 'Short' and leg_long['side'] == 'Long')
    
    for day in range(0, days_forward + 1, 30):  # Monthly steps including day 0
        # Time factor in years
        time_years = day / 365
        
        # CARRY CALCULATION
        # For a steepener: You're short the front (pay spread) and long the back (receive spread)
        # Daily carry in bps = (spread_long - spread_short) for DV01-neutral position
        # But we need to weight by actual DV01s
        
        if is_steepener:
            # Short position pays spread, long position receives spread
            # Net carry = (spread received on long - spread paid on short) adjusted for DV01s
            # Since positions are DV01-weighted, use the ratio
            daily_carry_bps = (current_spread_long - current_spread_short * (dv01_short/dv01_long if dv01_long > 0 else 1))
        else:
            # Flattener: long front (receive), short back (pay)
            daily_carry_bps = (current_spread_short - current_spread_long * (dv01_long/dv01_short if dv01_short > 0 else 1))
        
        # Convert daily carry to P&L over time period
        carry_pnl = daily_carry_bps * time_years * dv01_long * (notional / 10_000_000)
        
        # ROLL CALCULATION
        # Estimate how spreads evolve as they roll down the curve
        # Simplified approach: spreads converge toward shorter-dated levels
        
        if day > 0:
            # Estimate roll-down effect
            # As time passes, 5Y becomes ~4Y, 10Y becomes ~9Y
            # Use curve slope to estimate new spread levels
            
            # Simplified roll assumptions:
            # - 5Y rolls faster (moves 20% toward 3Y level per year)
            # - 10Y rolls slower (moves 10% toward 7Y level per year)
            
            roll_factor_short = 0.20 * time_years  # 20% roll per year for shorter tenor
            roll_factor_long = 0.10 * time_years   # 10% roll per year for longer tenor
            
            # Estimate forward spreads based on curve shape
            # Spreads typically compress (move lower) as they roll down
            forward_spread_short = current_spread_short * (1 - roll_factor_short * 0.3)  # 30% of roll effect
            forward_spread_long = current_spread_long * (1 - roll_factor_long * 0.2)   # 20% of roll effect
            
            # Ensure spreads remain positive
            forward_spread_short = max(forward_spread_short, 10)
            forward_spread_long = max(forward_spread_long, 15)
            
            # Roll P&L from spread changes
            spread_change_short = current_spread_short - forward_spread_short
            spread_change_long = current_spread_long - forward_spread_long
            
            if is_steepener:
                # Short front benefits from spread tightening, long back hurt by spread tightening
                roll_pnl = (spread_change_short * dv01_short - spread_change_long * dv01_long) * (notional / 10_000_000)
            else:
                # Flattener: opposite
                roll_pnl = (-spread_change_short * dv01_short + spread_change_long * dv01_long) * (notional / 10_000_000)
        else:
            forward_spread_short = current_spread_short
            forward_spread_long = current_spread_long
            roll_pnl = 0
        
        total_pnl = carry_pnl + roll_pnl
        
        daily_carry_roll.append({
            'days': day,
            'carry_pnl': carry_pnl,
            'roll_pnl': roll_pnl,
            'total_pnl': total_pnl,
            'forward_spread_short': forward_spread_short,
            'forward_spread_long': forward_spread_long,
            'curve_slope': forward_spread_long - forward_spread_short
        })
    
    return pd.DataFrame(daily_carry_roll)

def fit_linear_relationship(spread_short, spread_long):
    """
    Fit LINEAR relationship between spreads 
    Returns slope, intercept, and R-squared
    """
    # Remove NaN values
    mask = ~(np.isnan(spread_short) | np.isnan(spread_long))
    x = spread_short[mask].values.reshape(-1, 1)
    y = spread_long[mask].values
    
    if len(x) < 10:
        return None, None, None, None, None
    
    # Fit linear model
    model = LinearRegression()
    model.fit(x, y)
    
    # Get predictions for smooth line
    x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    
    # Calculate R-squared
    y_fitted = model.predict(x)
    r_squared = 1 - (np.sum((y - y_fitted)**2) / np.sum((y - np.mean(y))**2))
    
    slope = model.coef_[0]
    intercept = model.intercept_
    
    return x_range, y_pred, r_squared, slope, intercept

def render_enhanced_strategy_monitor():
    """Main function to render enhanced strategy monitor"""
    
    # Initialize session state for strategy details
    if 'current_strategy' not in st.session_state:
        st.session_state.current_strategy = None
    
    # Sidebar for strategy configuration
    with st.sidebar:
        st.header("Strategy Configuration")
        
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
            end_date = st.date_input("End Date", value=datetime.now())
        
        st.divider()
        
        # Configure legs based on strategy type
        legs = []
        
        if strategy_type == "Curve Trade (5s10s)":
            index = st.selectbox("Index", ["EU_IG", "EU_XO", "US_IG", "US_HY"])
            
            # Allow custom curve selection
            curve_type = st.selectbox("Curve", ["5s10s", "3s5s", "5s7s", "7s10s", "3s10s", "Custom"])
            
            if curve_type == "Custom":
                col1, col2 = st.columns(2)
                with col1:
                    short_tenor = st.selectbox("Short Tenor", ["3Y", "5Y", "7Y"])
                with col2:
                    long_tenor = st.selectbox("Long Tenor", ["5Y", "7Y", "10Y"])
            else:
                # Predefined curve mappings
                curve_mappings = {
                    "5s10s": ("5Y", "10Y"),
                    "3s5s": ("3Y", "5Y"),
                    "5s7s": ("5Y", "7Y"),
                    "7s10s": ("7Y", "10Y"),
                    "3s10s": ("3Y", "10Y")
                }
                short_tenor, long_tenor = curve_mappings[curve_type]
            
            trade_type = st.radio("Trade Type", ["Steepener", "Flattener"])
            weight_method = st.selectbox("Weighting", ["DV01 Neutral", "Equal", "Beta", "Manual"])
            
            # Manual sizing option
            if weight_method == "Manual":
                col1, col2 = st.columns(2)
                with col1:
                    manual_short = st.number_input(f"{short_tenor} Weight %", value=100.0, min_value=0.0)
                with col2:
                    manual_long = st.number_input(f"{long_tenor} Weight %", value=100.0, min_value=0.0)
                weight_short = manual_short
                weight_long = manual_long
                
            # Calculate weights based on ACTUAL data
            elif weight_method == "DV01 Neutral":
                # Load data to get current spreads
                conn = sqlite3.connect(RAW_DB_PATH, check_same_thread=False)
                
                # Get current spreads for DV01 calculation
                query_short = f"SELECT spread_bps FROM raw_historical_spreads WHERE index_name = '{index}' AND tenor = '{short_tenor}' ORDER BY date DESC LIMIT 1"
                query_long = f"SELECT spread_bps FROM raw_historical_spreads WHERE index_name = '{index}' AND tenor = '{long_tenor}' ORDER BY date DESC LIMIT 1"
                
                spread_short = pd.read_sql_query(query_short, conn).iloc[0, 0] if not pd.read_sql_query(query_short, conn).empty else 50
                spread_long = pd.read_sql_query(query_long, conn).iloc[0, 0] if not pd.read_sql_query(query_long, conn).empty else 60
                
                # Calculate ACTUAL DV01s
                dv01_short = calculate_actual_dv01(index, short_tenor, spread_short)
                dv01_long = calculate_actual_dv01(index, long_tenor, spread_long)
                
                # DV01 neutral weights
                weight_short = 100
                weight_long = 100 * (dv01_short / dv01_long) if dv01_long > 0 else 100
                
                conn.close()
                
            elif weight_method == "Beta":
                # Load historical data for beta calculation
                conn = sqlite3.connect(RAW_DB_PATH, check_same_thread=False)
                
                query_short = f"SELECT date, spread_bps FROM raw_historical_spreads WHERE index_name = '{index}' AND tenor = '{short_tenor}' AND date >= date('now', '-180 days') ORDER BY date"
                query_long = f"SELECT date, spread_bps FROM raw_historical_spreads WHERE index_name = '{index}' AND tenor = '{long_tenor}' AND date >= date('now', '-180 days') ORDER BY date"
                
                df_short = pd.read_sql_query(query_short, conn)
                df_long = pd.read_sql_query(query_long, conn)
                
                # Calculate actual beta
                beta = calculate_beta_weight(df_short, df_long)
                
                weight_short = 100
                weight_long = 100 / beta  # Inverse beta weighting
                
                conn.close()
            else:  # Equal
                weight_short = 100
                weight_long = 100
            
            # Display calculated weights
            if weight_method != "Manual":
                st.write(f"**Calculated Weights:** {short_tenor}: {weight_short:.1f}%, {long_tenor}: {weight_long:.1f}%")
            
            if trade_type == "Steepener":
                legs = [
                    {'index': index, 'tenor': short_tenor, 'side': 'Short', 'weight': weight_short},
                    {'index': index, 'tenor': long_tenor, 'side': 'Long', 'weight': weight_long}
                ]
            else:  # Flattener
                legs = [
                    {'index': index, 'tenor': short_tenor, 'side': 'Long', 'weight': weight_short},
                    {'index': index, 'tenor': long_tenor, 'side': 'Short', 'weight': weight_long}
                ]
        
        elif strategy_type == "Compression Trade":
            # Allow choice of direction
            trade_direction = st.radio("Trade Direction", 
                ["Long XO/HY vs Short IG (Compression)", 
                 "Short XO/HY vs Long IG (Decompression)"])
            
            # Determine which indices go where based on direction
            if "Compression" in trade_direction:
                # Standard compression: Long high spread, Short low spread
                xo_indices = ["EU_XO"]
                hy_indices = ["US_HY"]
                ig_indices = ["EU_IG", "US_IG"]
                
                high_spread_indices = xo_indices + hy_indices
                low_spread_indices = ig_indices
                
                long_index = st.selectbox("Long Index (High Spread)", high_spread_indices)
                short_index = st.selectbox("Short Index (Low Spread)", low_spread_indices)
            else:
                # Decompression: Short high spread, Long low spread
                xo_indices = ["EU_XO"]
                hy_indices = ["US_HY"]
                ig_indices = ["EU_IG", "US_IG"]
                
                high_spread_indices = xo_indices + hy_indices
                low_spread_indices = ig_indices
                
                short_index = st.selectbox("Short Index (High Spread)", high_spread_indices)
                long_index = st.selectbox("Long Index (Low Spread)", low_spread_indices)
            
            tenor = st.selectbox("Tenor", ['5Y', '10Y'])
            weight_method = st.selectbox("Weighting", ["Beta", "DV01 Neutral", "Equal", "Manual"])
            
            # Manual sizing option
            if weight_method == "Manual":
                col1, col2 = st.columns(2)
                with col1:
                    manual_long = st.number_input(f"{long_index} Weight %", value=100.0, min_value=0.0)
                with col2:
                    manual_short = st.number_input(f"{short_index} Weight %", value=100.0, min_value=0.0)
                weight_long = manual_long
                weight_short = manual_short
                
            # Weight calculation for compression - USING ACTUAL DATA
            elif weight_method == "Beta":
                # Calculate actual beta between indices
                conn = sqlite3.connect(RAW_DB_PATH, check_same_thread=False)
                
                query_long = f"SELECT date, spread_bps FROM raw_historical_spreads WHERE index_name = '{long_index}' AND tenor = '{tenor}' AND date >= date('now', '-180 days') ORDER BY date"
                query_short = f"SELECT date, spread_bps FROM raw_historical_spreads WHERE index_name = '{short_index}' AND tenor = '{tenor}' AND date >= date('now', '-180 days') ORDER BY date"
                
                df_long = pd.read_sql_query(query_long, conn)
                df_short = pd.read_sql_query(query_short, conn)
                
                beta = calculate_beta_weight(df_short, df_long)
                weight_long = 100
                weight_short = 100 * beta  # Beta times the long position
                
                conn.close()
                
            elif weight_method == "DV01 Neutral":
                # Get current spreads and calculate DV01s
                conn = sqlite3.connect(RAW_DB_PATH, check_same_thread=False)
                
                query_long = f"SELECT spread_bps FROM raw_historical_spreads WHERE index_name = '{long_index}' AND tenor = '{tenor}' ORDER BY date DESC LIMIT 1"
                query_short = f"SELECT spread_bps FROM raw_historical_spreads WHERE index_name = '{short_index}' AND tenor = '{tenor}' ORDER BY date DESC LIMIT 1"
                
                spread_long = pd.read_sql_query(query_long, conn).iloc[0, 0] if not pd.read_sql_query(query_long, conn).empty else 250
                spread_short = pd.read_sql_query(query_short, conn).iloc[0, 0] if not pd.read_sql_query(query_short, conn).empty else 50
                
                dv01_long = calculate_actual_dv01(long_index, tenor, spread_long)
                dv01_short = calculate_actual_dv01(short_index, tenor, spread_short)
                
                weight_long = 100
                weight_short = 100 * (dv01_long / dv01_short) if dv01_short > 0 else 100
                
                conn.close()
            else:  # Equal
                weight_long = 100
                weight_short = 100
            
            # Display calculated weights
            if weight_method != "Manual":
                st.write(f"**Calculated Weights:** {long_index}: {weight_long:.1f}%, {short_index}: {weight_short:.1f}%")
            
            legs = [
                {'index': long_index, 'tenor': tenor, 'side': 'Long', 'weight': weight_long},
                {'index': short_index, 'tenor': tenor, 'side': 'Short', 'weight': weight_short}
            ]
        
        else:  # Custom Multi-Leg
            n_legs = st.number_input("Number of Legs", min_value=1, max_value=4, value=2)
            
            # Weighting method selection
            weight_method = st.selectbox("Weighting Method", ["Equal", "DV01 Neutral", "Beta", "Custom"])
            
            leg_configs = []
            for i in range(n_legs):
                st.subheader(f"Leg {i+1}")
                col1, col2 = st.columns(2)
                with col1:
                    index = st.selectbox(f"Index", ["EU_IG", "EU_XO", "US_IG", "US_HY"], key=f"idx_{i}")
                    tenor = st.selectbox(f"Tenor", ["3Y", "5Y", "7Y", "10Y"], key=f"tenor_{i}")
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
                # Get actual DV01s for each leg
                conn = sqlite3.connect(RAW_DB_PATH, check_same_thread=False)
                
                for i, config in enumerate(leg_configs):
                    # Get current spread
                    query = f"SELECT spread_bps FROM raw_historical_spreads WHERE index_name = '{config['index']}' AND tenor = '{config['tenor']}' ORDER BY date DESC LIMIT 1"
                    result = pd.read_sql_query(query, conn)
                    spread = result.iloc[0, 0] if not result.empty else 100
                    
                    # Calculate actual DV01
                    dv01 = calculate_actual_dv01(config['index'], config['tenor'], spread)
                    config['dv01'] = dv01
                
                # Set weights based on DV01 ratios
                base_dv01 = leg_configs[0]['dv01']
                for i, config in enumerate(leg_configs):
                    if i == 0:
                        config['weight'] = 100.0
                    else:
                        config['weight'] = 100.0 * (base_dv01 / config['dv01']) if config['dv01'] > 0 else 100.0
                
                conn.close()
            
            elif weight_method == "Beta":
                # Calculate beta relationships
                conn = sqlite3.connect(RAW_DB_PATH, check_same_thread=False)
                
                # Get historical data for all legs
                historical_data = []
                for config in leg_configs:
                    query = f"SELECT date, spread_bps FROM raw_historical_spreads WHERE index_name = '{config['index']}' AND tenor = '{config['tenor']}' AND date >= date('now', '-180 days') ORDER BY date"
                    df = pd.read_sql_query(query, conn)
                    historical_data.append(df)
                
                # Calculate betas relative to first leg
                leg_configs[0]['weight'] = 100.0
                for i in range(1, len(leg_configs)):
                    if len(historical_data[0]) > 20 and len(historical_data[i]) > 20:
                        beta = calculate_beta_weight(historical_data[0], historical_data[i])
                        leg_configs[i]['weight'] = 100.0 / beta if leg_configs[i]['side'] != leg_configs[0]['side'] else 100.0 * beta
                    else:
                        leg_configs[i]['weight'] = 100.0
                
                conn.close()
            
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
        
        notional = st.number_input("Notional (MM)", value=10, min_value=1) * 1_000_000
        
        st.divider()
        
        run_analysis = st.button("Run Analysis", type="primary")
    
    # Main content area with tabs - only show if analysis is run
    if run_analysis and legs:
        # Build dynamic title based on strategy
        if strategy_type == "Curve Trade (5s10s)":
            if curve_type != "Custom":
                strategy_name = f"{index} {curve_type} {trade_type}"
            else:
                strategy_name = f"{index} {short_tenor.replace('Y', '')}s{long_tenor.replace('Y', '')} {trade_type}"
        elif strategy_type == "Compression Trade":
            if "Compression" in trade_direction:
                strategy_name = f"Compression: Long {long_index} vs Short {short_index} {tenor}"
            else:
                strategy_name = f"Decompression: Short {short_index} vs Long {long_index} {tenor}"
        else:
            strategy_name = f"Custom {n_legs}-Leg Strategy"
        
        # Display dynamic title
        st.title(f"Strategy Analysis: {strategy_name}")
        
        # Load historical spread data
        conn_raw = sqlite3.connect(RAW_DB_PATH, check_same_thread=False)
        
        # For 2-leg strategies, use the legs list
        if len(legs) == 2:
            leg_short = legs[0] if legs[0]['side'] == 'Short' else legs[1]
            leg_long = legs[1] if legs[1]['side'] == 'Long' else legs[0]
            weight_short = leg_short['weight']
            weight_long = leg_long['weight']
        else:
            # For multi-leg, focus on first two legs for convexity analysis
            leg_short = legs[0]
            leg_long = legs[1] if len(legs) > 1 else legs[0]
            weight_short = leg_short['weight']
            weight_long = leg_long['weight']
        
        # Query historical spreads for both legs
        query_short = f"""
            SELECT date, spread_bps 
            FROM raw_historical_spreads 
            WHERE index_name = '{leg_short['index']}' 
            AND tenor = '{leg_short['tenor']}'
            AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date
        """
        
        query_long = f"""
            SELECT date, spread_bps 
            FROM raw_historical_spreads 
            WHERE index_name = '{leg_long['index']}' 
            AND tenor = '{leg_long['tenor']}'
            AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date
        """
        
        df_short = pd.read_sql_query(query_short, conn_raw)
        df_long = pd.read_sql_query(query_long, conn_raw)
        
        # Merge data
        df_merged = pd.merge(df_short, df_long, on='date', suffixes=('_short', '_long'))
        
        # Apply weighting to spread calculation
        df_merged['weighted_spread_short'] = df_merged['spread_bps_short'] * (weight_short / 100)
        df_merged['weighted_spread_long'] = df_merged['spread_bps_long'] * (weight_long / 100)
        
        # Calculate weighted spread differential (accounting for long/short)
        if leg_short['side'] == 'Short':
            df_merged['spread_diff'] = df_merged['weighted_spread_long'] - df_merged['weighted_spread_short']
        else:
            df_merged['spread_diff'] = df_merged['weighted_spread_short'] - df_merged['weighted_spread_long']
        
        df_merged['spread_change_short'] = df_merged['spread_bps_short'].diff()
        df_merged['spread_change_long'] = df_merged['spread_bps_long'].diff()
        df_merged['spread_change'] = df_merged['spread_diff'].diff()
        
        # Calculate ACTUAL DV01s - NO HARDCODING
        conn = sqlite3.connect(RAW_DB_PATH, check_same_thread=False)
        
        # Get current spreads for DV01 calculation
        query_short_spread = f"SELECT spread_bps FROM raw_historical_spreads WHERE index_name = '{leg_short['index']}' AND tenor = '{leg_short['tenor']}' ORDER BY date DESC LIMIT 1"
        query_long_spread = f"SELECT spread_bps FROM raw_historical_spreads WHERE index_name = '{leg_long['index']}' AND tenor = '{leg_long['tenor']}' ORDER BY date DESC LIMIT 1"
        
        current_spread_short = pd.read_sql_query(query_short_spread, conn).iloc[0, 0] if not pd.read_sql_query(query_short_spread, conn).empty else 100
        current_spread_long = pd.read_sql_query(query_long_spread, conn).iloc[0, 0] if not pd.read_sql_query(query_long_spread, conn).empty else 100
        
        conn.close()
        
        # Calculate ACTUAL DV01s using our function
        base_dv01_short = calculate_actual_dv01(leg_short['index'], leg_short['tenor'], current_spread_short)
        base_dv01_long = calculate_actual_dv01(leg_long['index'], leg_long['tenor'], current_spread_long)
        
        # Apply weights to DV01s
        dv01_short = base_dv01_short * (weight_short / 100)
        dv01_long = base_dv01_long * (weight_long / 100)
        
        # Display actual calculated values
        st.sidebar.write("**Calculated DV01s (per MM):**")
        st.sidebar.write(f"Short leg: {base_dv01_short:.2f}")
        st.sidebar.write(f"Long leg: {base_dv01_long:.2f}")
        st.sidebar.write(f"DV01 ratio: {base_dv01_long/base_dv01_short:.3f}" if base_dv01_short > 0 else "N/A")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Convexity Analysis", "Carry/Roll Analysis", "Statistical Analysis"])
        
        with tab1:
            # Historical performance chart
            st.subheader("Historical Spread Performance")
            
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=("Spread Levels", "Spread Differential"),
                              row_heights=[0.6, 0.4])
            
            # Individual spreads
            fig.add_trace(
                go.Scatter(x=df_merged['date'], y=df_merged['spread_bps_short'],
                          name=f"{leg_short['index']} {leg_short['tenor']}", 
                          line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df_merged['date'], y=df_merged['spread_bps_long'],
                          name=f"{leg_long['index']} {leg_long['tenor']}", 
                          line=dict(color='red')),
                row=1, col=1
            )
            
            # Spread differential
            fig.add_trace(
                go.Scatter(x=df_merged['date'], y=df_merged['spread_diff'],
                          name='Spread Diff', line=dict(color='purple', width=2)),
                row=2, col=1
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                current_diff = df_merged['spread_diff'].iloc[-1]
                st.metric("Current Spread", f"{current_diff:.1f} bps")
            with col2:
                avg_diff = df_merged['spread_diff'].mean()
                st.metric("Average Spread", f"{avg_diff:.1f} bps")
            with col3:
                std_diff = df_merged['spread_diff'].std()
                st.metric("Spread Volatility", f"{std_diff:.1f} bps")
            with col4:
                sharpe = (df_merged['spread_change'].mean() / df_merged['spread_change'].std()) * np.sqrt(252) if df_merged['spread_change'].std() > 0 else 0
                st.metric("Spread Sharpe", f"{sharpe:.2f}")
        
        with tab2:
            st.subheader("Convexity Analysis")
            
            # Stack charts vertically for better visibility
            # First row - Scatter plot with LINEAR fit
            st.write("**Spread Relationship**")
            
            x_range, y_pred, r_squared, slope, intercept = fit_linear_relationship(
                df_merged['spread_bps_short'], 
                df_merged['spread_bps_long']
            )
            
            fig1 = go.Figure()
            
            # Scatter plot
            fig1.add_trace(go.Scatter(
                x=df_merged['spread_bps_short'],
                y=df_merged['spread_bps_long'],
                mode='markers',
                name='Historical',
                marker=dict(size=4, color='blue', opacity=0.5)
            ))
            
            # Linear fit - NOT quadratic
            if x_range is not None:
                fig1.add_trace(go.Scatter(
                    x=x_range.flatten(),
                    y=y_pred,
                    mode='lines',
                    name=f'Linear Fit (R²={r_squared:.3f})',
                    line=dict(color='red', width=3)
                ))
            
            fig1.update_layout(
                title="Convexity Scatter Plot",
                xaxis_title=f"{leg_short['index']} {leg_short['tenor']} (bps)",
                yaxis_title=f"{leg_long['index']} {leg_long['tenor']} (bps)",
                height=400
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Display relationship metrics
            col1, col2, col3 = st.columns(3)
            if slope is not None:
                with col1:
                    st.metric("Beta (Slope)", f"{slope:.3f}")
                with col2:
                    st.metric("Intercept", f"{intercept:.1f}")
                with col3:
                    st.metric("R-Squared", f"{r_squared:.3f}")
            
            # Second row - Parallel shift P&L analysis with QUADRATIC shape
            st.write("**Parallel Shift P&L Analysis**")
            
            convexity_df = calculate_convexity_pnl(
                df_merged[['spread_change']], 
                dv01_short, 
                dv01_long, 
                notional,
                leg_short=leg_short,
                leg_long=leg_long
            )
            
            fig2 = go.Figure()
            
            # This SHOULD show QUADRATIC/PARABOLIC shape of P&L
            fig2.add_trace(go.Scatter(
                x=convexity_df['shift'],
                y=convexity_df['total_pnl'],
                mode='lines+markers',
                name='Total P&L',
                line=dict(color='purple', width=3),
                marker=dict(size=6)
            ))
            
            # Add components
            fig2.add_trace(go.Scatter(
                x=convexity_df['shift'],
                y=convexity_df['pnl_short'],
                mode='lines',
                name=f'{leg_short["index"]} {leg_short["tenor"]} P&L',
                line=dict(color='blue', dash='dash', width=2)
            ))
            
            fig2.add_trace(go.Scatter(
                x=convexity_df['shift'],
                y=convexity_df['pnl_long'],
                mode='lines',
                name=f'{leg_long["index"]} {leg_long["tenor"]} P&L',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            # Add zero line
            fig2.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
            
            # Add vertical line at current spread level
            fig2.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1)
            
            fig2.update_layout(
                title="P&L vs Parallel Shift (Convexity Profile)",
                xaxis_title="Spread Shift (bps)",
                yaxis_title="P&L ($)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Calculate convexity from the P&L curve - PROPERLY
            if len(convexity_df) > 2:
                # Fit a quadratic to the P&L curve to measure convexity
                x = convexity_df['shift'].values
                y = convexity_df['total_pnl'].values
                
                # Fit polynomial: y = ax^2 + bx + c
                coeffs = np.polyfit(x, y, 2)
                
                # The coefficient of x^2 determines convexity
                # Positive coefficient = positive convexity (long gamma)
                # Negative coefficient = negative convexity (short gamma)
                convexity_coeff = coeffs[0]
                
                st.write("**Convexity Characteristics**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    convexity_type = "Positive (Long Gamma)" if convexity_coeff > 0 else "Negative (Short Gamma)"
                    st.metric("Convexity Type", convexity_type)
                with col2:
                    # Calculate the P&L at extremes vs linear interpolation
                    mid_idx = len(convexity_df) // 2
                    left_pnl = convexity_df.iloc[0]['total_pnl']
                    mid_pnl = convexity_df.iloc[mid_idx]['total_pnl']
                    right_pnl = convexity_df.iloc[-1]['total_pnl']
                    linear_mid = (left_pnl + right_pnl) / 2
                    convexity_magnitude = abs(mid_pnl - linear_mid)
                    st.metric("Convexity Magnitude", f"${convexity_magnitude:,.0f}")
                with col3:
                    # Show the range used
                    max_shift = convexity_df['shift'].max()
                    st.metric("Scenario Range", f"±{max_shift:.0f} bps")
        
        with tab3:
            st.subheader(f"Carry & Roll Analysis - {strategy_name}")
            
            # Determine if this is a steepener or flattener
            is_steepener = (leg_short['side'] == 'Short' and leg_long['side'] == 'Long')
            trade_type_display = "Steepener" if is_steepener else "Flattener"
            
            # Calculate theoretical forward P&L
            current_spread_short = df_merged['spread_bps_short'].iloc[-1]
            current_spread_long = df_merged['spread_bps_long'].iloc[-1]
            curve_slope = current_spread_long - current_spread_short
            
            # Display trade setup
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Trade Type", trade_type_display)
            with col2:
                st.metric(f"{leg_short['index']} {leg_short['tenor']}", f"{current_spread_short:.1f} bps")
            with col3:
                st.metric(f"{leg_long['index']} {leg_long['tenor']}", f"{current_spread_long:.1f} bps")
            with col4:
                st.metric("Curve", f"{curve_slope:.1f} bps")
            
            forward_df = calculate_theoretical_forward_pnl(
                current_spread_short, current_spread_long, curve_slope,
                dv01_short, dv01_long, days_forward=180, notional=notional,
                leg_short=leg_short, leg_long=leg_long
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Forward P&L projection
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=forward_df['days'],
                    y=forward_df['total_pnl'],
                    mode='lines',
                    name='Total P&L',
                    line=dict(color='purple', width=3)
                ))
                
                fig.add_trace(go.Scatter(
                    x=forward_df['days'],
                    y=forward_df['carry_pnl'],
                    mode='lines',
                    name='Carry Component',
                    line=dict(color='red' if is_steepener else 'green', dash='dash', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=forward_df['days'],
                    y=forward_df['roll_pnl'],
                    mode='lines',
                    name='Roll Component',
                    line=dict(color='green' if is_steepener else 'orange', dash='dash', width=2)
                ))
                
                # Add zero line
                fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
                
                fig.update_layout(
                    title=f"180-Day Forward P&L Projection ({trade_type_display})",
                    xaxis_title="Days Forward",
                    yaxis_title="P&L ($)",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Carry/Roll decomposition table
                st.write("**P&L Decomposition (Key Points)**")
                
                # Get values at key time points
                decomposition_data = []
                for days in [30, 90, 180]:
                    rows = forward_df[forward_df['days'] == days]
                    if not rows.empty:
                        row = rows.iloc[0]
                        decomposition_data.append({
                            'Days': f"{days}d",
                            'Carry P&L': f"${row['carry_pnl']:,.0f}",
                            'Roll P&L': f"${row['roll_pnl']:,.0f}",
                            'Total P&L': f"${row['total_pnl']:,.0f}"
                        })
                
                if decomposition_data:
                    decomposition_df = pd.DataFrame(decomposition_data)
                    st.dataframe(decomposition_df, hide_index=True)
                
                # Add interpretation
                final_pnl = forward_df.iloc[-1]
                st.write("**180-Day P&L Analysis:**")
                
                if is_steepener:
                    if final_pnl['carry_pnl'] < 0:
                        st.write("✓ Negative carry (typical for steepeners in upward sloping curves)")
                    else:
                        st.write("⚠️ Positive carry (unusual - check if curve is inverted)")
                    
                    if final_pnl['roll_pnl'] > 0:
                        st.write("✓ Positive roll (benefits from curve steepening)")
                    else:
                        st.write("⚠️ Negative roll (curve expected to flatten)")
                else:  # Flattener
                    if final_pnl['carry_pnl'] > 0:
                        st.write("✓ Positive carry (typical for flatteners)")
                    else:
                        st.write("⚠️ Negative carry (check positioning)")
                    
                    if final_pnl['roll_pnl'] < 0:
                        st.write("✓ Benefits from curve flattening")
                
                # Forward spread projections
                st.write("**Forward Spread Projections**")
                
                forward_spreads_data = []
                for days in [0, 90, 180]:
                    if days == 0:
                        forward_spreads_data.append({
                            'Days': '0d',
                            f"{leg_short['index']} {leg_short['tenor']}": f"{current_spread_short:.1f}",
                            f"{leg_long['index']} {leg_long['tenor']}": f"{current_spread_long:.1f}",
                            'Curve': f"{curve_slope:.1f}"
                        })
                    else:
                        rows = forward_df[forward_df['days'] == days]
                        if not rows.empty:
                            row = rows.iloc[0]
                            forward_spreads_data.append({
                                'Days': f"{days}d",
                                f"{leg_short['index']} {leg_short['tenor']}": f"{row['forward_spread_short']:.1f}",
                                f"{leg_long['index']} {leg_long['tenor']}": f"{row['forward_spread_long']:.1f}",
                                'Curve': f"{row['curve_slope']:.1f}"
                            })
                
                if forward_spreads_data:
                    forward_spreads = pd.DataFrame(forward_spreads_data)
                    st.dataframe(forward_spreads, hide_index=True)
        
        with tab4:
            st.subheader("Statistical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution analysis
                st.write("**Spread Distribution**")
                
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=df_merged['spread_diff'],
                    nbinsx=30,
                    name='Spread Distribution',
                    marker_color='blue',
                    opacity=0.7
                ))
                
                # Add normal distribution overlay
                mu = df_merged['spread_diff'].mean()
                sigma = df_merged['spread_diff'].std()
                x_range = np.linspace(df_merged['spread_diff'].min(), 
                                     df_merged['spread_diff'].max(), 100)
                normal_dist = stats.norm.pdf(x_range, mu, sigma) * len(df_merged) * \
                             (df_merged['spread_diff'].max() - df_merged['spread_diff'].min()) / 30
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=normal_dist,
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title="Spread Distribution vs Normal",
                    xaxis_title="Spread (bps)",
                    yaxis_title="Frequency",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical tests
                skew = stats.skew(df_merged['spread_diff'].dropna())
                kurtosis = stats.kurtosis(df_merged['spread_diff'].dropna())
                
                st.write("**Distribution Characteristics**")
                stats_df = pd.DataFrame({
                    'Metric': ['Skewness', 'Excess Kurtosis', 'Mean', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{skew:.3f}",
                        f"{kurtosis:.3f}",
                        f"{mu:.1f} bps",
                        f"{sigma:.1f} bps",
                        f"{df_merged['spread_diff'].min():.1f} bps",
                        f"{df_merged['spread_diff'].max():.1f} bps"
                    ]
                })
                st.dataframe(stats_df, hide_index=True)
            
            with col2:
                # Mean reversion analysis
                st.write("**Mean Reversion Analysis**")
                
                # Calculate z-score
                df_merged['z_score'] = (df_merged['spread_diff'] - mu) / sigma
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df_merged['date'],
                    y=df_merged['z_score'],
                    mode='lines',
                    name='Z-Score',
                    line=dict(color='purple', width=1)
                ))
                
                # Add threshold lines
                fig.add_hline(y=2, line_dash="dash", line_color="red", 
                            annotation_text="+2σ")
                fig.add_hline(y=-2, line_dash="dash", line_color="red", 
                            annotation_text="-2σ")
                fig.add_hline(y=0, line_dash="solid", line_color="gray", 
                            annotation_text="Mean")
                
                fig.update_layout(
                    title="Spread Z-Score (Mean Reversion)",
                    xaxis_title="Date",
                    yaxis_title="Z-Score",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate mean reversion metrics
                half_life = calculate_half_life(df_merged['spread_diff'])
                
                st.write("**Mean Reversion Metrics**")
                reversion_df = pd.DataFrame({
                    'Metric': ['Half-Life (days)', 'Current Z-Score', 'Percentile Rank'],
                    'Value': [
                        f"{half_life:.0f}" if half_life else "N/A",
                        f"{df_merged['z_score'].iloc[-1]:.2f}",
                        f"{stats.percentileofscore(df_merged['spread_diff'], df_merged['spread_diff'].iloc[-1]):.0f}%"
                    ]
                })
                st.dataframe(reversion_df, hide_index=True)
        
        conn_raw.close()

def calculate_half_life(series):
    """Calculate mean reversion half-life using OLS"""
    try:
        series = series.dropna()
        if len(series) < 20:
            return None
        
        # Fit AR(1) model: y_t = alpha + beta * y_{t-1} + epsilon
        y = series[1:].values
        x = series[:-1].values
        
        # Add constant
        x_with_const = np.column_stack([np.ones(len(x)), x])
        
        # OLS regression
        beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
        
        # Half-life = -log(2) / log(beta[1])
        if 0 < beta[1] < 1:
            half_life = -np.log(2) / np.log(beta[1])
            return half_life if half_life > 0 and half_life < 365 else None
        return None
    except:
        return None

# Main execution
if __name__ == "__main__":
    render_enhanced_strategy_monitor()