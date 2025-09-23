# 3_Series_Monitor.py - Place in src/apps/pages/
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import plotly.graph_objects as go
import sys
import os

# Add paths for imports if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
apps_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(apps_dir)
root_dir = os.path.dirname(src_dir)
sys.path.insert(0, root_dir)

# Page config is handled by main app
st.header("Series Monitor - Excel Style")

# Database path
RAW_DB_PATH = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\raw\cds_indices_raw.db"

def calculate_cds_dv01(spread_bps, tenor_years, notional_mm=10, risk_free_rate=0.03, recovery_rate=0.4):
    """
    Calculate CDS DV01 analytically using RPV01 formula
    
    DV01 = Notional × RPV01 × 0.0001
    RPV01 = Risky PV01 = ∫(0,T) S(t) × D(t) dt
    
    Where:
    - S(t) = Survival probability = exp(-λt)
    - D(t) = Discount factor = exp(-rt)
    - λ = hazard rate = spread/(1-recovery)
    
    Parameters:
    - spread_bps: Current CDS spread in basis points
    - tenor_years: Maturity in years
    - notional_mm: Notional in millions (default 10)
    - risk_free_rate: Risk-free rate for discounting (default 3%)
    - recovery_rate: Recovery rate assumption (default 40%)
    
    Returns:
    - DV01 in dollars per basis point
    """
    if pd.isna(spread_bps) or spread_bps <= 0:
        # If no spread, can't calculate DV01
        return 0
    
    # Convert spread to hazard rate
    spread_decimal = spread_bps / 10000
    hazard_rate = spread_decimal / (1 - recovery_rate)
    
    # Combined discount rate (risk-free + credit)
    total_discount = risk_free_rate + hazard_rate
    
    # Calculate RPV01 (Risky PV01)
    if abs(total_discount * tenor_years) < 0.001:
        # Use Taylor expansion for numerical stability with small values
        rpv01 = tenor_years * (1 - total_discount * tenor_years / 2 + 
                               (total_discount * tenor_years)**2 / 6)
    else:
        # Full analytical formula for RPV01
        rpv01 = (1 - np.exp(-total_discount * tenor_years)) / total_discount
    
    # DV01 = RPV01 × Notional × 0.0001 (for 1bp move)
    dv01 = rpv01 * notional_mm * 1_000_000 * 0.0001
    
    # Return in thousands for display
    return dv01 / 1000

def get_bloomberg_dv01(ticker, tenor, spread_bps):
    """
    Get DV01 from Bloomberg or calculate analytically
    
    Uses SW_EQV_BPV field with BEST ticker for executable quotes
    Falls back to analytical calculation if Bloomberg unavailable
    """
    try:
        # Import Bloomberg connector if available
        from data.bloomberg_connector import BloombergConnector
        connector = BloombergConnector()
        
        # Use BEST for executable quotes (not CBIN)
        bbg_ticker = f"{ticker} BEST INDEX"
        
        # Get DV01 from Bloomberg using SW_EQV_BPV field
        data = connector.blp.bdp(bbg_ticker, flds="SW_EQV_BPV")
        
        if not data.empty and 'SW_EQV_BPV' in data.columns:
            dv01 = data['SW_EQV_BPV'].iloc[0]
            if pd.notna(dv01) and dv01 != 0:
                # Bloomberg returns DV01 for $10MM notional
                # Return absolute value in thousands
                return abs(dv01) / 1000
    except ImportError:
        # Bloomberg not available
        pass
    except Exception as e:
        # Bloomberg query failed
        print(f"Bloomberg DV01 query failed: {e}")
    
    # Fallback to analytical calculation
    tenor_years = float(tenor.replace('Y', ''))
    return calculate_cds_dv01(spread_bps, tenor_years, notional_mm=10)

def build_monitor_table(conn, index_name, current_series):
    """Build the monitor data table matching Excel layout"""
    
    tenors = ['3Y', '5Y', '7Y', '10Y']
    series_to_show = [current_series, current_series - 1, current_series - 2, current_series - 3, current_series - 4]
    
    # Create DataFrame structure similar to Excel
    monitor_df = pd.DataFrame()
    
    # Get spreads for each series
    for series_num in series_to_show:
        series_data = {}
        series_label = f"S{series_num}"
        
        for tenor in tenors:
            query = f"""
                SELECT spread_bps
                FROM raw_historical_spreads
                WHERE index_name = '{index_name}'
                AND tenor = '{tenor}'
                AND series_number = {series_num}
                ORDER BY date DESC
                LIMIT 1
            """
            result = pd.read_sql_query(query, conn)
            
            if not result.empty:
                series_data[tenor] = round(result['spread_bps'].iloc[0], 2)
            else:
                series_data[tenor] = None
        
        monitor_df[series_label] = pd.Series(series_data)
    
    # Calculate DV01 for each tenor and series
    for tenor in tenors:
        dv01_row = {}
        tenor_years = float(tenor.replace('Y', ''))
        
        for col in monitor_df.columns:
            if tenor in monitor_df.index:
                spread = monitor_df.loc[tenor, col]
                if pd.notna(spread):
                    # Build Bloomberg ticker for this series
                    series_num = int(col.replace('S', ''))
                    if index_name == 'EU_IG':
                        ticker = f"ITRX EUR CDSI S{series_num} {tenor}"
                    elif index_name == 'EU_XO':
                        ticker = f"ITRX EUR XOVER S{series_num} {tenor}"
                    elif index_name == 'US_IG':
                        ticker = f"CDX IG CDSI S{series_num} {tenor}"
                    elif index_name == 'US_HY':
                        ticker = f"CDX HY CDSI S{series_num} {tenor}"
                    else:
                        ticker = f"{index_name} S{series_num} {tenor}"
                    
                    # Get DV01 (Bloomberg or calculated)
                    dv01 = get_bloomberg_dv01(ticker, tenor, spread)
                    dv01_row[col] = round(dv01, 2)
                else:
                    dv01_row[col] = None
        
        monitor_df.loc[f'DV01_{tenor}'] = pd.Series(dv01_row)
    
    return monitor_df

def calculate_curve_metrics(df):
    """Calculate curve slope metrics"""
    metrics = {}
    
    # Get latest series column (first column)
    if df.shape[1] > 0:
        latest_col = df.columns[0]
        
        # 3s5s
        if '3Y' in df.index and '5Y' in df.index:
            val_3y = df.loc['3Y', latest_col]
            val_5y = df.loc['5Y', latest_col]
            if pd.notna(val_3y) and pd.notna(val_5y):
                metrics['3s5s'] = round(val_5y - val_3y, 2)
        
        # 5s7s
        if '5Y' in df.index and '7Y' in df.index:
            val_5y = df.loc['5Y', latest_col]
            val_7y = df.loc['7Y', latest_col]
            if pd.notna(val_5y) and pd.notna(val_7y):
                metrics['5s7s'] = round(val_7y - val_5y, 2)
        
        # 7s10s
        if '7Y' in df.index and '10Y' in df.index:
            val_7y = df.loc['7Y', latest_col]
            val_10y = df.loc['10Y', latest_col]
            if pd.notna(val_7y) and pd.notna(val_10y):
                metrics['7s10s'] = round(val_10y - val_7y, 2)
        
        # 5s10s
        if '5Y' in df.index and '10Y' in df.index:
            val_5y = df.loc['5Y', latest_col]
            val_10y = df.loc['10Y', latest_col]
            if pd.notna(val_5y) and pd.notna(val_10y):
                metrics['5s10s'] = round(val_10y - val_5y, 2)
        
        # Roll (current vs previous series)
        if df.shape[1] > 1:
            current_col = df.columns[0]
            prev_col = df.columns[1]
            
            roll_3s5s = None
            if '3Y' in df.index and '5Y' in df.index:
                curr_3s5s = df.loc['5Y', current_col] - df.loc['3Y', current_col] if pd.notna(df.loc['5Y', current_col]) and pd.notna(df.loc['3Y', current_col]) else None
                prev_3s5s = df.loc['5Y', prev_col] - df.loc['3Y', prev_col] if pd.notna(df.loc['5Y', prev_col]) and pd.notna(df.loc['3Y', prev_col]) else None
                if curr_3s5s is not None and prev_3s5s is not None:
                    metrics['Roll_3s5s'] = round(curr_3s5s - prev_3s5s, 2)
            
            # Similar for other rolls
            if '5Y' in df.index and '10Y' in df.index:
                curr_5s10s = df.loc['10Y', current_col] - df.loc['5Y', current_col] if pd.notna(df.loc['10Y', current_col]) and pd.notna(df.loc['5Y', current_col]) else None
                prev_5s10s = df.loc['10Y', prev_col] - df.loc['5Y', prev_col] if pd.notna(df.loc['10Y', prev_col]) and pd.notna(df.loc['5Y', prev_col]) else None
                if curr_5s10s is not None and prev_5s10s is not None:
                    metrics['Roll_5s10s'] = round(curr_5s10s - prev_5s10s, 2)
    
    return metrics

def calculate_forward_spreads(monitor_df):
    """
    Calculate forward spreads using DV01 weighting
    
    Formula: Forward = ((Spread_Short + Curve_Slope) * DV01_Long - Spread_Short * DV01_Short) / (DV01_Long - DV01_Short)
    
    For 5Y5Y forward: Uses 5Y and 10Y points
    For 3Y2Y forward: Uses 3Y and 5Y points
    """
    forward_results = {}
    
    # Get the current series column (first column)
    if monitor_df.shape[1] > 0:
        current_col = monitor_df.columns[0]
        
        # Calculate 5Y5Y forward (5Y forward starting in 5Y)
        if all(x in monitor_df.index for x in ['5Y', '10Y', 'DV01_5Y', 'DV01_10Y']):
            spread_5y = monitor_df.loc['5Y', current_col]
            spread_10y = monitor_df.loc['10Y', current_col]
            dv01_5y = monitor_df.loc['DV01_5Y', current_col]
            dv01_10y = monitor_df.loc['DV01_10Y', current_col]
            
            if all(pd.notna(x) for x in [spread_5y, spread_10y, dv01_5y, dv01_10y]) and (dv01_10y - dv01_5y) != 0:
                curve_5s10s = spread_10y - spread_5y
                forward_5y5y = ((spread_5y + curve_5s10s) * dv01_10y - spread_5y * dv01_5y) / (dv01_10y - dv01_5y)
                forward_results['5Y5Y'] = round(forward_5y5y, 2)
        
        # Calculate 3Y2Y forward (2Y forward starting in 3Y)
        if all(x in monitor_df.index for x in ['3Y', '5Y', 'DV01_3Y', 'DV01_5Y']):
            spread_3y = monitor_df.loc['3Y', current_col]
            spread_5y = monitor_df.loc['5Y', current_col]
            dv01_3y = monitor_df.loc['DV01_3Y', current_col]
            dv01_5y = monitor_df.loc['DV01_5Y', current_col]
            
            if all(pd.notna(x) for x in [spread_3y, spread_5y, dv01_3y, dv01_5y]) and (dv01_5y - dv01_3y) != 0:
                curve_3s5s = spread_5y - spread_3y
                forward_3y2y = ((spread_3y + curve_3s5s) * dv01_5y - spread_3y * dv01_3y) / (dv01_5y - dv01_3y)
                forward_results['3Y2Y'] = round(forward_3y2y, 2)
        
        # Calculate 7Y3Y forward (3Y forward starting in 7Y)
        if all(x in monitor_df.index for x in ['7Y', '10Y', 'DV01_7Y', 'DV01_10Y']):
            spread_7y = monitor_df.loc['7Y', current_col]
            spread_10y = monitor_df.loc['10Y', current_col]
            dv01_7y = monitor_df.loc['DV01_7Y', current_col]
            dv01_10y = monitor_df.loc['DV01_10Y', current_col]
            
            if all(pd.notna(x) for x in [spread_7y, spread_10y, dv01_7y, dv01_10y]) and (dv01_10y - dv01_7y) != 0:
                curve_7s10s = spread_10y - spread_7y
                forward_7y3y = ((spread_7y + curve_7s10s) * dv01_10y - spread_7y * dv01_7y) / (dv01_10y - dv01_7y)
                forward_results['7Y3Y'] = round(forward_7y3y, 2)
    
    return forward_results
    """
    Calculate forward spreads using DV01 weighting
    
    Formula: Forward = ((Spread_Short + Curve_Slope) * DV01_Long - Spread_Short * DV01_Short) / (DV01_Long - DV01_Short)
    
    For 5Y5Y forward: Uses 5Y and 10Y points
    For 3Y2Y forward: Uses 3Y and 5Y points
    """
    forward_results = {}
    
    # Get the current series column (first column)
    if monitor_df.shape[1] > 0:
        current_col = monitor_df.columns[0]
        
        # Calculate 5Y5Y forward (5Y forward starting in 5Y)
        if all(x in monitor_df.index for x in ['5Y', '10Y', 'DV01_5Y', 'DV01_10Y']):
            spread_5y = monitor_df.loc['5Y', current_col]
            spread_10y = monitor_df.loc['10Y', current_col]
            dv01_5y = monitor_df.loc['DV01_5Y', current_col]
            dv01_10y = monitor_df.loc['DV01_10Y', current_col]
            
            if all(pd.notna(x) for x in [spread_5y, spread_10y, dv01_5y, dv01_10y]) and (dv01_10y - dv01_5y) != 0:
                curve_5s10s = spread_10y - spread_5y
                forward_5y5y = ((spread_5y + curve_5s10s) * dv01_10y - spread_5y * dv01_5y) / (dv01_10y - dv01_5y)
                forward_results['5Y5Y'] = round(forward_5y5y, 2)
        
        # Calculate 3Y2Y forward (2Y forward starting in 3Y)
        if all(x in monitor_df.index for x in ['3Y', '5Y', 'DV01_3Y', 'DV01_5Y']):
            spread_3y = monitor_df.loc['3Y', current_col]
            spread_5y = monitor_df.loc['5Y', current_col]
            dv01_3y = monitor_df.loc['DV01_3Y', current_col]
            dv01_5y = monitor_df.loc['DV01_5Y', current_col]
            
            if all(pd.notna(x) for x in [spread_3y, spread_5y, dv01_3y, dv01_5y]) and (dv01_5y - dv01_3y) != 0:
                curve_3s5s = spread_5y - spread_3y
                forward_3y2y = ((spread_3y + curve_3s5s) * dv01_5y - spread_3y * dv01_3y) / (dv01_5y - dv01_3y)
                forward_results['3Y2Y'] = round(forward_3y2y, 2)
        
        # Calculate 7Y3Y forward (3Y forward starting in 7Y)
        if all(x in monitor_df.index for x in ['7Y', '10Y', 'DV01_7Y', 'DV01_10Y']):
            spread_7y = monitor_df.loc['7Y', current_col]
            spread_10y = monitor_df.loc['10Y', current_col]
            dv01_7y = monitor_df.loc['DV01_7Y', current_col]
            dv01_10y = monitor_df.loc['DV01_10Y', current_col]
            
            if all(pd.notna(x) for x in [spread_7y, spread_10y, dv01_7y, dv01_10y]) and (dv01_10y - dv01_7y) != 0:
                curve_7s10s = spread_10y - spread_7y
                forward_7y3y = ((spread_7y + curve_7s10s) * dv01_10y - spread_7y * dv01_7y) / (dv01_10y - dv01_7y)
                forward_results['7Y3Y'] = round(forward_7y3y, 2)
    
    return forward_results

def calculate_carry(monitor_df, forward_spreads, notional_mm=10):
    """
    Calculate 1Y carry P&L for curve STEEPENER trades
    
    For a steepener (short front, long back):
    Formula: Carry P&L = Notional × 0.01% × ((Spread_short + Curve) - (DV01_long/DV01_short) × Spread_short)
    
    This should typically be NEGATIVE for steepeners in upward-sloping curves
    """
    carry_results = {}
    
    # Convert notional to dollars and apply 1bp factor
    notional_dollars = notional_mm * 1_000_000 * 0.0001  # 0.01% = 0.0001
    
    if monitor_df.shape[1] > 0:
        current_col = monitor_df.columns[0]
        
        # 3s5s steepener carry (short 3Y, long 5Y)
        if all(x in monitor_df.index for x in ['3Y', '5Y', 'DV01_3Y', 'DV01_5Y']):
            spread_3y = monitor_df.loc['3Y', current_col]  # Short leg
            spread_5y = monitor_df.loc['5Y', current_col]  # Long leg
            dv01_3y = monitor_df.loc['DV01_3Y', current_col]
            dv01_5y = monitor_df.loc['DV01_5Y', current_col]
            
            if all(pd.notna(x) for x in [spread_3y, spread_5y, dv01_3y, dv01_5y]) and dv01_3y != 0:
                curve_3s5s = spread_5y - spread_3y
                # Steepener carry formula
                carry_pnl = notional_dollars * ((spread_3y + curve_3s5s) - (dv01_5y/dv01_3y) * spread_3y)
                carry_results['3s5s'] = round(carry_pnl, 0)
        
        # 5s10s steepener carry (short 5Y, long 10Y)
        if all(x in monitor_df.index for x in ['5Y', '10Y', 'DV01_5Y', 'DV01_10Y']):
            spread_5y = monitor_df.loc['5Y', current_col]  # Short leg
            spread_10y = monitor_df.loc['10Y', current_col]  # Long leg
            dv01_5y = monitor_df.loc['DV01_5Y', current_col]
            dv01_10y = monitor_df.loc['DV01_10Y', current_col]
            
            if all(pd.notna(x) for x in [spread_5y, spread_10y, dv01_5y, dv01_10y]) and dv01_5y != 0:
                curve_5s10s = spread_10y - spread_5y
                carry_pnl = notional_dollars * ((spread_5y + curve_5s10s) - (dv01_10y/dv01_5y) * spread_5y)
                carry_results['5s10s'] = round(carry_pnl, 0)
        
        # 7s10s steepener carry (short 7Y, long 10Y)
        if all(x in monitor_df.index for x in ['7Y', '10Y', 'DV01_7Y', 'DV01_10Y']):
            spread_7y = monitor_df.loc['7Y', current_col]  # Short leg
            spread_10y = monitor_df.loc['10Y', current_col]  # Long leg
            dv01_7y = monitor_df.loc['DV01_7Y', current_col]
            dv01_10y = monitor_df.loc['DV01_10Y', current_col]
            
            if all(pd.notna(x) for x in [spread_7y, spread_10y, dv01_7y, dv01_10y]) and dv01_7y != 0:
                curve_7s10s = spread_10y - spread_7y
                carry_pnl = notional_dollars * ((spread_7y + curve_7s10s) - (dv01_10y/dv01_7y) * spread_7y)
                carry_results['7s10s'] = round(carry_pnl, 0)
    
    return carry_results

# Main execution
try:
    conn = sqlite3.connect(RAW_DB_PATH, check_same_thread=False)
    
    # Index selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        index_family = st.selectbox(
            "Index Family",
            ["ITRAXX", "CDX"],
            key="series_monitor_family"
        )
    
    with col2:
        if index_family == "ITRAXX":
            indices = ["EU_IG", "EU_XO"]
        else:
            indices = ["US_IG", "US_HY"]
        
        selected_index = st.selectbox(
            "Select Index",
            indices,
            key="series_monitor_index"
        )
    
    with col3:
        # Get current series number
        current_series_query = f"""
            SELECT MAX(series_number) as current_series
            FROM raw_historical_spreads
            WHERE index_name = '{selected_index}'
        """
        current_series_result = pd.read_sql_query(current_series_query, conn)
        current_series = current_series_result['current_series'].iloc[0] if not current_series_result.empty else 43
        
        st.metric("On-The-Run", f"Series {current_series}")
    
    # Display main monitor table
    st.subheader(f"{index_family} {selected_index} - Series Comparison")
    
    # Build the monitor table
    monitor_df = build_monitor_table(conn, selected_index, current_series)
    
    if not monitor_df.empty:
        # First row - Main data tables and small chart
        col1, col2, col3, col4 = st.columns([2.5, 1.5, 1.5, 2])
        
        with col1:
            st.write("**Spreads by Tenor and Series**")
            
            # Format the display - show spreads only
            display_df = monitor_df.copy()
            
            # Extract DV01 rows for separate display
            dv01_rows = display_df[display_df.index.str.startswith('DV01_')].copy()
            spread_rows = display_df[~display_df.index.str.startswith('DV01_')].copy()
            
            # Display spreads
            st.dataframe(
                spread_rows.round(2),
                use_container_width=True,
                height=150
            )
            
            # Display DV01 directly below spreads
            st.write("**DV01 Values (Calculated)**")
            if not dv01_rows.empty:
                # Rename index to remove DV01_ prefix for cleaner display
                dv01_rows.index = dv01_rows.index.str.replace('DV01_', '')
                st.dataframe(
                    dv01_rows.round(2),
                    use_container_width=True,
                    height=150
                )
        
        with col2:
            st.write("**Curve Metrics**")
            
            metrics = calculate_curve_metrics(monitor_df)
            
            # Display curve slopes
            curve_df = pd.DataFrame([
                {'Curve': '3s5s', 'Spread': f"{metrics.get('3s5s', 'N/A'):.1f}" if isinstance(metrics.get('3s5s'), (int, float)) else 'N/A'},
                {'Curve': '5s7s', 'Spread': f"{metrics.get('5s7s', 'N/A'):.1f}" if isinstance(metrics.get('5s7s'), (int, float)) else 'N/A'},
                {'Curve': '7s10s', 'Spread': f"{metrics.get('7s10s', 'N/A'):.1f}" if isinstance(metrics.get('7s10s'), (int, float)) else 'N/A'},
                {'Curve': '5s10s', 'Spread': f"{metrics.get('5s10s', 'N/A'):.1f}" if isinstance(metrics.get('5s10s'), (int, float)) else 'N/A'}
            ])
            st.dataframe(curve_df, hide_index=True, height=150)
            
            # Show roll if available
            if 'Roll_3s5s' in metrics or 'Roll_5s10s' in metrics:
                st.write("**Roll (vs Previous Series)**")
                roll_df = pd.DataFrame([
                    {'Curve': '3s5s Roll', 'Value': f"{metrics.get('Roll_3s5s', 'N/A'):.2f}" if isinstance(metrics.get('Roll_3s5s'), (int, float)) else 'N/A'},
                    {'Curve': '5s10s Roll', 'Value': f"{metrics.get('Roll_5s10s', 'N/A'):.2f}" if isinstance(metrics.get('Roll_5s10s'), (int, float)) else 'N/A'}
                ])
                st.dataframe(roll_df, hide_index=True, height=100)
        
        with col3:
            # Calculate forward spreads and carry with notional
            forward_spreads = calculate_forward_spreads(monitor_df)
            carry_values = calculate_carry(monitor_df, forward_spreads, notional_mm=10)
            
            st.write("**1Y Carry Calculation**")
            st.caption("P&L for $10MM notional")
            carry_df = pd.DataFrame([
                {'Curve': '3s5s', 'Carry ($)': f"${carry_values.get('3s5s', 0):,.0f}" if '3s5s' in carry_values else 'N/A'},
                {'Curve': '5s10s', 'Carry ($)': f"${carry_values.get('5s10s', 0):,.0f}" if '5s10s' in carry_values else 'N/A'},
                {'Curve': '7s10s', 'Carry ($)': f"${carry_values.get('7s10s', 0):,.0f}" if '7s10s' in carry_values else 'N/A'}
            ])
            st.dataframe(carry_df, hide_index=True, height=120)
            
            st.write("**Forward Spreads**")
            forward_df = pd.DataFrame([
                {'Forward': '3Y2Y', 'Spread (bps)': f"{forward_spreads.get('3Y2Y', 'N/A'):.1f}" if isinstance(forward_spreads.get('3Y2Y'), (int, float)) else 'N/A'},
                {'Forward': '5Y5Y', 'Spread (bps)': f"{forward_spreads.get('5Y5Y', 'N/A'):.1f}" if isinstance(forward_spreads.get('5Y5Y'), (int, float)) else 'N/A'},
                {'Forward': '7Y3Y', 'Spread (bps)': f"{forward_spreads.get('7Y3Y', 'N/A'):.1f}" if isinstance(forward_spreads.get('7Y3Y'), (int, float)) else 'N/A'}
            ])
            st.dataframe(forward_df, hide_index=True, height=120)
        
        with col4:
            # Small chart in the right column
            st.write("**Curve Visualization**")
            
            fig = go.Figure()
            
            # Plot only current and previous series for cleaner view
            tenors_map = {'3Y': 3, '5Y': 5, '7Y': 7, '10Y': 10}
            colors = ['blue', 'red', 'gray']
            
            for i, col in enumerate(monitor_df.columns[:3]):  # Show only 3 series
                if not col.startswith('DV01'):
                    y_values = []
                    x_values = []
                    
                    for tenor in ['3Y', '5Y', '7Y', '10Y']:
                        if tenor in monitor_df.index:
                            val = monitor_df.loc[tenor, col]
                            if pd.notna(val):
                                y_values.append(val)
                                x_values.append(tenors_map[tenor])
                    
                    if y_values:
                        fig.add_trace(go.Scatter(
                            x=x_values,
                            y=y_values,
                            mode='lines+markers',
                            name=col,
                            line=dict(
                                color=colors[i % len(colors)],
                                width=2 if i == 0 else 1,
                                dash=None if i == 0 else 'dot'
                            ),
                            marker=dict(size=6)
                        ))
            
            fig.update_layout(
                xaxis_title="Tenor",
                yaxis_title="Spread (bps)",
                xaxis=dict(
                    tickmode='array',
                    tickvals=[3, 5, 7, 10],
                    ticktext=['3Y', '5Y', '7Y', '10Y']
                ),
                height=300,  # Slightly taller since we have more space
                margin=dict(l=0, r=0, t=10, b=30),  # Minimal margins
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Optional: Additional row for more analytics if needed
        st.divider()
        
        # Series info summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Series", f"S{current_series}")
        
        with col2:
            if monitor_df.shape[1] > 0 and '5Y' in monitor_df.index:
                current_5y = monitor_df.loc['5Y', monitor_df.columns[0]]
                st.metric("5Y Spread", f"{current_5y:.1f} bps" if pd.notna(current_5y) else "N/A")
        
        with col3:
            if '5s10s' in metrics:
                st.metric("5s10s Slope", f"{metrics['5s10s']:.1f} bps")
        
        with col4:
            st.metric("Notional", "$10MM")
    
    else:
        st.warning(f"No data available for {selected_index}")
    
    conn.close()
    
except Exception as e:
    st.error(f"Error loading series monitor: {e}")
    st.info("Please check database connection and data availability")