# pages/4_Option_Pricing.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, src_dir)

# Import the option pricing system
from models.cds_index_options import CDSIndexOptionPricer, VolSurfaceDatabase

# Initialize components
pricer = CDSIndexOptionPricer()
vol_db = VolSurfaceDatabase()

st.set_page_config(page_title="Option Pricing", layout="wide")
st.title("CDS Index Option Pricing")

# Sidebar for database management
with st.sidebar:
    st.header("Database Management")
    
    # Show current database status
    available_dates = vol_db.get_available_dates()
    if available_dates:
        st.success(f"✓ Database connected")
        st.write(f"Latest data: {available_dates[0]}")
        st.write(f"Total dates: {len(available_dates)}")
    else:
        st.error("⚠ No data available")
    
    st.divider()
    
    # Database update section
    st.subheader("Update Vol Surfaces")
    
    excel_path = st.text_input(
        "Excel Path",
        value=r"C:\Users\alessandro.esposito\Portman Square Capital LLP\Portman Square Capital - Documents\S\CSA\Credit Index Trading\vol_db.xlsx"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        update_date = st.date_input("Data Date", value=datetime.now())
    with col2:
        force_update = st.checkbox("Force Update")
    
    if st.button("Update Database", type="primary"):
        with st.spinner("Updating database..."):
            try:
                count = vol_db.create_or_update_database(
                    excel_path, 
                    update_date.strftime("%Y-%m-%d"),
                    force_update
                )
                st.success(f"Updated {count} options")
                st.rerun()
            except Exception as e:
                st.error(f"Update failed: {e}")

# Main content - Option Pricing
tab1, tab2, tab3, tab4 = st.tabs(["Single Option", "Strategy Builder", "Surface Analysis", "Validation"])

with tab1:
    st.header("Single Option Pricing")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        index_name = st.selectbox("Index", ["EU_IG", "EU_XO", "US_IG", "US_HY"])
        
    with col2:
        tenor = st.selectbox("Tenor", ["1m", "2m", "3m", "4m", "5m", "6m"])
        
    with col3:
        # Get forward level for strike suggestions
        try:
            market_data = pricer.get_market_data(index_name, tenor)
            if not market_data.empty:
                forward = market_data['forward_level'].iloc[0]
                if index_name == "US_HY":
                    # Price-based strikes for CDX HY
                    default_strike = round(forward)
                    strike = st.number_input("Strike (Price)", value=float(default_strike), step=0.25)
                else:
                    # Spread-based strikes
                    default_strike = round(forward)
                    strike = st.number_input("Strike (Spread)", value=float(default_strike), step=1.0)
            else:
                strike = st.number_input("Strike", value=100.0)
        except:
            strike = st.number_input("Strike", value=100.0)
            
    with col4:
        option_type = st.selectbox("Type", ["Payer", "Receiver"])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        notional = st.number_input("Notional (MM)", value=10.0, min_value=0.1, step=1.0) * 1_000_000
        
    with col2:
        if available_dates:
            data_date = st.selectbox("Data Date", available_dates)
        else:
            data_date = st.date_input("Data Date", value=datetime.now()).strftime("%Y-%m-%d")
            
    with col3:
        # Underlying shift for scenario analysis
        shift_bps = st.number_input("Shift Underlying (bps)", value=0, step=1, min_value=-100, max_value=100)
        
    with col4:
        st.write("")  # Spacer
        price_button = st.button("Calculate Price", type="primary", key="single_price")
    
    if price_button:
        try:
            # Get market data
            market_data = pricer.get_market_data(index_name, tenor, data_date)
            
            if market_data.empty:
                st.error("No market data available")
            else:
                # Apply shift to forward if specified
                original_forward = market_data['forward_level'].iloc[0]
                shifted_forward = original_forward + shift_bps
                
                # Price with original forward
                result = pricer.price_single_option(
                    index_name=index_name,
                    tenor=tenor,
                    strike=strike,
                    option_type=option_type,
                    notional=notional,
                    data_date=data_date
                )
                
                # If shift specified, also price with shifted forward
                if shift_bps != 0:
                    # Create shifted market data
                    market_data_shifted = market_data.copy()
                    market_data_shifted['forward_level'] = shifted_forward
                    
                    # We need to manually calculate with shift
                    expiry_date = pd.to_datetime(market_data['expiry'].iloc[0], format='%d-%b-%y')
                    value_date = pd.to_datetime(data_date)
                    days_to_expiry = (expiry_date - value_date).days
                    
                    vol = pricer.interpolate_vol(market_data, strike, option_type)
                    
                    result_shifted = pricer.price_option(
                        forward=shifted_forward,
                        strike=strike,
                        vol=vol,
                        days_to_expiry=days_to_expiry,
                        option_type=option_type,
                        index_name=index_name
                    )
                
                # Display results
                st.subheader("Pricing Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Market Parameters**")
                    params_df = pd.DataFrame([
                        ["Index", index_name],
                        ["Tenor", tenor],
                        ["Strike", f"{strike:.2f}"],
                        ["Forward", f"{original_forward:.2f}"],
                        ["Volatility", f"{result['vol']:.1f}%"],
                        ["Days to Expiry", result['days_to_expiry']],
                        ["Notional", f"${notional:,.0f}"]
                    ], columns=["Parameter", "Value"])
                    st.dataframe(params_df, hide_index=True, use_container_width=True)
                
                with col2:
                    st.write("**Option Value**")
                    
                    if shift_bps == 0:
                        value_df = pd.DataFrame([
                            ["Upfront (bps)", f"{result['upfront_bps']:.2f}"],
                            ["Upfront ($)", f"${result['upfront_currency']:,.2f}"],
                            ["Delta (%)", f"{result['delta']:.1f}"],
                            ["Gamma", f"{result['gamma']:.4f}"],
                            ["Vega ($)", f"${result['vega_currency']:,.2f}"],
                            ["Theta ($/day)", f"${result['theta_currency']:,.2f}"],
                            ["Duration Used", f"{result.get('duration_used', 'N/A'):.3f}" if 'duration_used' in result else "N/A"]
                        ], columns=["Metric", "Value"])
                    else:
                        # Show comparison with shift
                        value_df = pd.DataFrame([
                            ["", "Original", "Shifted", "Change"],
                            ["Forward", f"{original_forward:.2f}", f"{shifted_forward:.2f}", f"{shift_bps:+d} bps"],
                            ["Upfront (bps)", f"{result['upfront_bps']:.2f}", f"{result_shifted['upfront_bps']:.2f}", f"{result_shifted['upfront_bps'] - result['upfront_bps']:+.2f}"],
                            ["Delta (%)", f"{result['delta']:.1f}", f"{result_shifted['delta']:.1f}", f"{result_shifted['delta'] - result['delta']:+.1f}"],
                            ["P&L from shift", "", "", f"${(result_shifted['upfront_bps'] - result['upfront_bps']) * notional / 10000:+,.0f}"]
                        ])
                        value_df.columns = value_df.iloc[0]
                        value_df = value_df[1:]
                    
                    st.dataframe(value_df, hide_index=True, use_container_width=True)
                
                # Compare with market prices
                st.subheader("Market Comparison")
                
                # Find the specific option in market data
                # First try exact match
                option_mask = (market_data['strike'] == strike) & (market_data['option_type'] == option_type)
                
                # If no exact match, find closest
                if not option_mask.any():
                    option_data = market_data[market_data['option_type'] == option_type]
                    if not option_data.empty:
                        # Find closest strike
                        strikes = option_data['strike'].values
                        closest_idx = np.abs(strikes - strike).argmin()
                        closest_strike = strikes[closest_idx]
                        
                        # Use closest if within reasonable range (2% for spreads, 0.5 for prices)
                        threshold = 0.5 if index_name == "US_HY" else strike * 0.02
                        if abs(closest_strike - strike) <= threshold:
                            option_mask = (market_data['strike'] == closest_strike) & (market_data['option_type'] == option_type)
                            st.info(f"Using closest available strike: {closest_strike:.2f} (requested: {strike:.2f})")
                
                if option_mask.any():
                    market_option = market_data[option_mask].iloc[0]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if pd.notna(market_option['bid']):
                            st.metric("Market Bid", f"{market_option['bid']:.2f} bps")
                        else:
                            st.metric("Market Bid", "N/A")
                    with col2:
                        if pd.notna(market_option['mid']):
                            st.metric("Market Mid", f"{market_option['mid']:.2f} bps")
                        else:
                            st.metric("Market Mid", "N/A")
                    with col3:
                        if pd.notna(market_option['ask']):
                            st.metric("Market Ask", f"{market_option['ask']:.2f} bps")
                        else:
                            st.metric("Market Ask", "N/A")
                    with col4:
                        if 'delta' in market_option and pd.notna(market_option['delta']):
                            st.metric("Market Delta", f"{market_option['delta']:.1f}%")
                        else:
                            st.metric("Market Delta", "N/A")
                    
                    # Check if within spread
                    model_price = result['upfront_bps']
                    if pd.notna(market_option['bid']) and pd.notna(market_option['ask']):
                        within_spread = market_option['bid'] <= model_price <= market_option['ask']
                        
                        if within_spread:
                            st.success(f"✓ Model price {model_price:.2f} bps is within bid/ask spread")
                        else:
                            if pd.notna(market_option['mid']):
                                diff = model_price - market_option['mid']
                                pct_diff = (diff / market_option['mid'] * 100) if market_option['mid'] != 0 else 0
                                st.warning(f"⚠ Model price differs from mid by {diff:+.2f} bps ({pct_diff:+.1f}%)")
                    elif pd.notna(market_option['mid']):
                        diff = model_price - market_option['mid']
                        pct_diff = (diff / market_option['mid'] * 100) if market_option['mid'] != 0 else 0
                        st.info(f"Model vs Mid: {diff:+.2f} bps ({pct_diff:+.1f}%)")
                else:
                    st.warning("No market data available for this strike")
                
                # Debug section at the bottom - separate from main comparison
                with st.expander("Debug Information"):
                    available_strikes = market_data[market_data['option_type'] == option_type]['strike'].unique()
                    st.write(f"Available {option_type} strikes: {sorted(available_strikes)}")
                    st.write(f"Your strike: {strike}")
                    st.write("All available options:")
                    display_df = market_data[['strike', 'option_type', 'bid', 'ask', 'mid', 'vol']].sort_values(['option_type', 'strike'])
                    st.dataframe(display_df, hide_index=True)
                
                # Special info for CDX HY
                if index_name == "US_HY" and 'implied_spread' in result:
                    st.write("**CDX HY Specific (500 bps coupon)**")
                    cdx_df = pd.DataFrame([
                        ["Implied Forward Spread", f"{result['implied_spread']:.1f} bps"],
                        ["Strike Spread", f"{result['strike_spread']:.1f} bps"],
                        ["Fixed Coupon", "500 bps"],
                        ["Recovery Rate", "25%"]
                    ], columns=["Parameter", "Value"])
                    st.dataframe(cdx_df, hide_index=True)
                    
        except Exception as e:
            st.error(f"Pricing failed: {e}")

with tab2:
    st.header("Strategy Builder")
    
    # Strategy configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        n_legs = st.number_input("Number of Legs", min_value=1, max_value=4, value=2)
    with col2:
        strategy_notional = st.number_input("Strategy Notional (MM)", value=10.0, min_value=0.1) * 1_000_000
    with col3:
        strategy_date = st.selectbox("Data Date", available_dates, key="strategy_date") if available_dates else datetime.now().strftime("%Y-%m-%d")
    
    # Configure each leg
    legs = []
    for i in range(n_legs):
        st.subheader(f"Leg {i+1}")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            leg_index = st.selectbox(f"Index", ["EU_IG", "EU_XO", "US_IG", "US_HY"], key=f"leg_idx_{i}")
        with col2:
            leg_tenor = st.selectbox(f"Tenor", ["1m", "2m", "3m", "4m", "5m", "6m"], key=f"leg_tenor_{i}")
        with col3:
            leg_strike = st.number_input(f"Strike", value=100.0, key=f"leg_strike_{i}")
        with col4:
            leg_type = st.selectbox(f"Type", ["Payer", "Receiver"], key=f"leg_type_{i}")
        with col5:
            leg_position = st.selectbox(f"Position", [1, -1], key=f"leg_pos_{i}", format_func=lambda x: "Long" if x == 1 else "Short")
        
        legs.append({
            'index_name': leg_index,
            'tenor': leg_tenor,
            'strike': leg_strike,
            'option_type': leg_type,
            'position': leg_position,
            'notional': strategy_notional
        })
    
    if st.button("Price Strategy", type="primary"):
        try:
            strategy_result = pricer.price_strategy(legs, strategy_date)
            
            # Display results
            st.subheader("Strategy Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Upfront", f"${strategy_result['total_upfront']:,.2f}")
            with col2:
                st.metric("Total Delta", f"{strategy_result['total_delta']:.1f}%")
            with col3:
                st.metric("Total Vega", f"${strategy_result['total_vega']:,.2f}")
            with col4:
                st.metric("Strategy Type", strategy_result['strategy_type'])
            
            # Leg details
            st.subheader("Leg Details")
            leg_details = []
            for i, leg_result in enumerate(strategy_result['legs']):
                leg = leg_result['leg']
                price = leg_result['price']
                leg_details.append({
                    'Leg': i+1,
                    'Index': leg['index_name'],
                    'Tenor': leg['tenor'],
                    'Strike': leg['strike'],
                    'Type': leg['option_type'],
                    'Position': "Long" if leg['position'] == 1 else "Short",
                    'Upfront (bps)': f"{price['upfront_bps']:.2f}",
                    'Contribution ($)': f"{leg_result['contribution']:,.2f}",
                    'Delta': f"{price['delta']:.1f}%"
                })
            
            legs_df = pd.DataFrame(leg_details)
            st.dataframe(legs_df, hide_index=True, use_container_width=True)
            
        except Exception as e:
            st.error(f"Strategy pricing failed: {e}")

with tab3:
    st.header("Volatility Surface Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        surf_index = st.selectbox("Index", ["EU_IG", "EU_XO", "US_IG", "US_HY"], key="surf_idx")
    with col2:
        surf_tenor = st.selectbox("Tenor", ["1m", "2m", "3m", "4m", "5m", "6m"], key="surf_tenor")
    with col3:
        surf_date = st.selectbox("Data Date", available_dates, key="surf_date") if available_dates else datetime.now().strftime("%Y-%m-%d")
    
    if st.button("Load Surface", type="primary"):
        try:
            surface_data = pricer.get_market_data(surf_index, surf_tenor, surf_date)
            
            if not surface_data.empty:
                # Get forward and ATM
                forward = surface_data['forward_level'].iloc[0]
                atm_strike = surface_data['atm_strike'].iloc[0] if 'atm_strike' in surface_data.columns else forward
                
                # Display surface info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Forward", f"{forward:.2f}")
                with col2:
                    st.metric("ATM Strike", f"{atm_strike:.2f}")
                with col3:
                    st.metric("Expiry", surface_data['expiry'].iloc[0])
                with col4:
                    st.metric("Options", len(surface_data))
                
                # Create vol smile chart
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Volatility Smile", "Vol Surface Heatmap"))
                
                # Separate by option type
                payers = surface_data[surface_data['option_type'] == 'Payer'].sort_values('strike')
                receivers = surface_data[surface_data['option_type'] == 'Receiver'].sort_values('strike')
                
                # Vol smile
                fig.add_trace(
                    go.Scatter(x=payers['strike'], y=payers['vol'], mode='markers+lines',
                              name='Payer', marker=dict(color='red')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=receivers['strike'], y=receivers['vol'], mode='markers+lines',
                              name='Receiver', marker=dict(color='blue')),
                    row=1, col=1
                )
                
                # Add ATM line
                fig.add_vline(x=forward, line_dash="dash", line_color="gray", 
                            annotation_text="Forward", row=1, col=1)
                
                # Create heatmap data
                strikes = sorted(surface_data['strike'].unique())
                vol_matrix = []
                for option_type in ['Receiver', 'Payer']:
                    vols = []
                    for strike in strikes:
                        mask = (surface_data['strike'] == strike) & (surface_data['option_type'] == option_type)
                        if mask.any():
                            vols.append(surface_data[mask]['vol'].iloc[0])
                        else:
                            vols.append(np.nan)
                    vol_matrix.append(vols)
                
                # Heatmap
                fig.add_trace(
                    go.Heatmap(z=vol_matrix, x=strikes, y=['Receiver', 'Payer'],
                              colorscale='RdBu', showscale=True, colorbar=dict(title="Vol %")),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=True)
                fig.update_xaxes(title_text="Strike", row=1, col=1)
                fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
                fig.update_xaxes(title_text="Strike", row=1, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display full surface data
                st.subheader("Surface Data")
                display_cols = ['strike', 'option_type', 'bid', 'ask', 'mid', 'vol', 'delta']
                display_data = surface_data[display_cols].sort_values(['option_type', 'strike'])
                st.dataframe(display_data, hide_index=True, use_container_width=True)
                
        except Exception as e:
            st.error(f"Failed to load surface: {e}")

with tab4:
    st.header("Model Validation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        val_index = st.selectbox("Index", ["EU_IG", "EU_XO", "US_IG", "US_HY"], key="val_idx")
    with col2:
        val_tenor = st.selectbox("Tenor", ["1m", "2m", "3m", "4m", "5m", "6m"], key="val_tenor")
    with col3:
        val_date = st.selectbox("Data Date", available_dates, key="val_date") if available_dates else datetime.now().strftime("%Y-%m-%d")
    
    if st.button("Run Validation", type="primary"):
        try:
            validation_results = pricer.validate_pricing(val_index, val_tenor, val_date)
            
            if not validation_results.empty:
                # Summary metrics
                st.subheader("Validation Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    accuracy = validation_results['within_spread'].mean() * 100
                    st.metric("Within Bid/Ask", f"{accuracy:.1f}%")
                with col2:
                    avg_error = validation_results['difference'].abs().mean()
                    st.metric("Avg Error", f"{avg_error:.2f} bps")
                with col3:
                    max_error = validation_results['difference'].abs().max()
                    st.metric("Max Error", f"{max_error:.2f} bps")
                with col4:
                    pct_error = validation_results['pct_error'].abs().mean()
                    st.metric("Avg % Error", f"{pct_error:.1f}%")
                
                # Error distribution
                fig = make_subplots(rows=1, cols=2, 
                                  subplot_titles=("Pricing Errors by Strike", "Error Distribution"))
                
                # Scatter plot of errors
                for option_type in ['Payer', 'Receiver']:
                    data = validation_results[validation_results['option_type'] == option_type]
                    fig.add_trace(
                        go.Scatter(x=data['strike'], y=data['difference'],
                                 mode='markers', name=option_type,
                                 marker=dict(size=8)),
                        row=1, col=1
                    )
                
                # Add zero line
                fig.add_hline(y=0, line_dash="solid", line_color="gray", row=1, col=1)
                
                # Histogram of errors
                fig.add_trace(
                    go.Histogram(x=validation_results['difference'], nbinsx=20,
                               name='Error Distribution'),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text="Strike", row=1, col=1)
                fig.update_yaxes(title_text="Model - Market (bps)", row=1, col=1)
                fig.update_xaxes(title_text="Error (bps)", row=1, col=2)
                fig.update_yaxes(title_text="Count", row=1, col=2)
                
                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.subheader("Detailed Results")
                
                # For CDX HY, show additional columns
                display_cols = ['strike', 'option_type', 'market_bid', 'market_mid', 'market_ask', 
                              'model_price', 'difference', 'pct_error', 'within_spread']
                
                if val_index == "US_HY" and 'implied_spread' in validation_results.columns:
                    display_cols.extend(['implied_spread', 'strike_spread'])
                
                detailed_results = validation_results[display_cols].sort_values('strike')
                
                # Color code the within_spread column
                def highlight_accuracy(row):
                    if row['within_spread']:
                        return ['background-color: lightgreen'] * len(row)
                    else:
                        return ['background-color: lightyellow'] * len(row)
                
                styled_results = detailed_results.style.apply(highlight_accuracy, axis=1)
                st.dataframe(styled_results, hide_index=True, use_container_width=True)
                
                # Export option
                csv = validation_results.to_csv(index=False)
                st.download_button("Download Results", csv, f"validation_{val_index}_{val_tenor}_{val_date}.csv", "text/csv")
                
        except Exception as e:
            st.error(f"Validation failed: {e}")