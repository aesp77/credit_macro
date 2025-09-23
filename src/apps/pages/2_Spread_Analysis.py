# Standalone Streamlit CDS Monitor - No Import Dependencies
# This version works without importing from other modules

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="CDS Monitor - Spread Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database paths
RAW_DB_PATH = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\raw\cds_indices_raw.db"
TRS_DB_PATH = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\processed\cds_trs.db"

# ============================================================================
# DATABASE WRAPPER
# ============================================================================

class RawDatabase:
    """Simple database wrapper for raw CDS data"""
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
    
    def query_historical_spreads(self, index_name, tenor, start_date=None, end_date=None):
        """Query historical spreads from raw_historical_spreads table"""
        query = """
            SELECT date, spread_bps, series_number 
            FROM raw_historical_spreads 
            WHERE index_name = ? AND tenor = ?
        """
        params = [index_name, tenor]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        try:
            df = pd.read_sql_query(query, self.conn, params=params)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            st.error(f"Query error: {e}")
            return pd.DataFrame()

# ============================================================================
# SPREAD VISUALIZATION FUNCTIONS
# ============================================================================

def render_spread_analysis():
    """Render spread analysis tab"""
    st.header("CDS Spread Analysis Dashboard")
    
    # Check if database exists
    try:
        db = RawDatabase(RAW_DB_PATH)
    except Exception as e:
        st.error(f"Cannot connect to database: {e}")
        st.info("Please ensure the database exists at: " + RAW_DB_PATH)
        return
    
    # Create sub-tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Historical Evolution", 
        "Current Curves", 
        "Series Roll Analysis", 
        "Spread Comparison",
        "Data Quality"
    ])
    
    with tab1:
        plot_historical_evolution(db)
    
    with tab2:
        plot_current_curves(db)
    
    with tab3:
        plot_series_roll_analysis(db)
    
    with tab4:
        plot_spread_comparison(db)
    
    with tab5:
        render_data_quality(db)

def plot_historical_evolution(db):
    """Plot historical spread evolution"""
    st.subheader("Historical Spread Evolution")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        index_name = st.selectbox(
            "Select Index",
            ["EU_IG", "EU_XO", "US_IG", "US_HY"],
            key="hist_index"
        )
    
    with col2:
        tenor = st.selectbox(
            "Select Tenor",
            ["1Y", "3Y", "5Y", "7Y", "10Y"],
            index=2,  # Default to 5Y
            key="hist_tenor"
        )
    
    with col3:
        period = st.selectbox(
            "Time Period",
            ["1M", "3M", "6M", "1Y", "2Y", "5Y", "All"],
            index=3,  # Default to 1Y
            key="hist_period"
        )
    
    # Calculate date range
    end_date = datetime.now()
    period_map = {
        "1M": 30, "3M": 90, "6M": 180, 
        "1Y": 365, "2Y": 730, "5Y": 1825, 
        "All": 3650
    }
    start_date = end_date - timedelta(days=period_map[period])
    
    # Query data
    df = db.query_historical_spreads(
        index_name, tenor, 
        start_date=start_date.strftime('%Y-%m-%d')
    )
    
    if not df.empty:
        # Create interactive Plotly chart
        fig = go.Figure()
        
        # Main spread line
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['spread_bps'],
            mode='lines',
            name=f'{index_name} {tenor}',
            line=dict(color='red', width=2),
            hovertemplate='Date: %{x}<br>Spread: %{y:.1f} bps<br>Series: S%{customdata}<extra></extra>',
            customdata=df['series_number']
        ))
        
        # Main spread line with series info
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['spread_bps'],
            mode='lines',
            name=f'{index_name} {tenor}',
            line=dict(color='red', width=2),
            hovertemplate='Date: %{x}<br>Spread: %{y:.1f} bps<br>Series: S%{customdata}<extra></extra>',
            customdata=df['series_number']
        ))
        # Calculate and display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_val = df.iloc[-1]['spread_bps']
            st.metric("Current", f"{current_val:.1f} bps")
        with col2:
            avg_val = df['spread_bps'].mean()
            st.metric("Average", f"{avg_val:.1f} bps")
        with col3:
            min_val = df['spread_bps'].min()
            st.metric("Min", f"{min_val:.1f} bps")
        with col4:
            max_val = df['spread_bps'].max()
            st.metric("Max", f"{max_val:.1f} bps")
        
        # Add moving averages
        show_ma = st.checkbox("Show Moving Averages", key="hist_ma")
        if show_ma:
            # 20-day MA
            df['MA20'] = df['spread_bps'].rolling(window=20, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['MA20'],
                mode='lines',
                name='20-day MA',
                line=dict(color='blue', width=1, dash='dot')
            ))
            
            # 50-day MA
            df['MA50'] = df['spread_bps'].rolling(window=50, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['MA50'],
                mode='lines',
                name='50-day MA',
                line=dict(color='green', width=1, dash='dot')
            ))
        
        fig.update_layout(
            title=f"{index_name} {tenor} Spread Evolution",
            xaxis_title="Date",
            yaxis_title="Spread (bps)",
            height=500,
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional analysis
        with st.expander("Additional Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Spread Distribution**")
                volatility = df['spread_bps'].std()
                st.write(f"Volatility: {volatility:.2f} bps")
                st.write(f"Median: {df['spread_bps'].median():.1f} bps")
                
            with col2:
                st.write("**Recent Changes**")
                if len(df) > 1:
                    daily_change = df['spread_bps'].iloc[-1] - df['spread_bps'].iloc[-2]
                    weekly_change = df['spread_bps'].iloc[-1] - df['spread_bps'].iloc[-5] if len(df) > 5 else 0
                    st.write(f"Daily Change: {daily_change:+.1f} bps")
                    st.write(f"Weekly Change: {weekly_change:+.1f} bps")
    else:
        st.warning(f"No data available for {index_name} {tenor}")

def plot_current_curves(db):
    """Plot current CDS curves"""
    st.subheader("Current CDS Curves")
    
    # Multi-select for indices
    selected_indices = st.multiselect(
        "Select Indices to Display",
        ["EU_IG", "EU_XO", "US_IG", "US_HY"],
        default=["EU_IG", "EU_XO", "US_IG", "US_HY"],
        key="curve_indices"
    )
    
    if not selected_indices:
        st.warning("Please select at least one index")
        return
    
    tenors_order = ['1Y', '3Y', '5Y', '7Y', '10Y']
    tenor_positions = {'1Y': 1, '3Y': 3, '5Y': 5, '7Y': 7, '10Y': 10}
    
    fig = go.Figure()
    colors = ['blue', 'green', 'red', 'orange']
    
    # Store curve data for table
    curve_data = []
    
    for i, index_name in enumerate(selected_indices):
        x_vals = []
        y_vals = []
        hover_texts = []
        
        for tenor in tenors_order:
            df = db.query_historical_spreads(
                index_name, tenor,
                start_date=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
            )
            
            if not df.empty:
                latest = df.iloc[-1]
                x_vals.append(tenor_positions[tenor])
                y_vals.append(latest['spread_bps'])
                hover_texts.append(f"{tenor}: {latest['spread_bps']:.1f} bps<br>Date: {latest['date']}")
                
                curve_data.append({
                    'Index': index_name,
                    'Tenor': tenor,
                    'Spread (bps)': latest['spread_bps'],
                    'Date': latest['date'].strftime('%Y-%m-%d') if hasattr(latest['date'], 'strftime') else latest['date']
                })
        
        if x_vals:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                name=index_name,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=8),
                hovertext=hover_texts,
                hoverinfo='text'
            ))
    
    fig.update_layout(
        title="Current CDS Curves Comparison",
        xaxis_title="Tenor (Years)",
        yaxis_title="Spread (bps)",
        xaxis=dict(
            tickmode='array',
            tickvals=[1, 3, 5, 7, 10],
            ticktext=['1Y', '3Y', '5Y', '7Y', '10Y']
        ),
        height=500,
        hovermode='x'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display data table
    if st.checkbox("Show Curve Data Table", key="curve_table"):
        if curve_data:
            df_curves = pd.DataFrame(curve_data)
            df_pivot = df_curves.pivot_table(
                index='Tenor', 
                columns='Index', 
                values='Spread (bps)', 
                aggfunc='first'
            )
            st.dataframe(df_pivot.round(1))
            
            # Calculate curve slopes
            st.subheader("Curve Slopes (5s10s)")
            slopes = {}
            for index in selected_indices:
                df_index = df_curves[df_curves['Index'] == index]
                spread_5y = df_index[df_index['Tenor'] == '5Y']['Spread (bps)'].values
                spread_10y = df_index[df_index['Tenor'] == '10Y']['Spread (bps)'].values
                if len(spread_5y) > 0 and len(spread_10y) > 0:
                    slopes[index] = spread_10y[0] - spread_5y[0]
            
            if slopes:
                slopes_df = pd.DataFrame(list(slopes.items()), columns=['Index', '5s10s Slope'])
                st.dataframe(slopes_df)

def plot_series_roll_analysis(db):
    """Analyze and visualize series rolls"""
    st.subheader("Series Roll Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        index_name = st.selectbox(
            "Select Index",
            ["EU_IG", "EU_XO", "US_IG", "US_HY"],
            key="roll_index"
        )
    
    with col2:
        tenor = st.selectbox(
            "Select Tenor",
            ["1Y", "3Y", "5Y", "7Y", "10Y"],
            index=2,
            key="roll_tenor"
        )
    
    # Query data with longer history
    df = db.query_historical_spreads(
        index_name, tenor,
        start_date='2020-01-01'
    )
    
    if not df.empty and 'series_number' in df.columns:
        # Create color map for series
        unique_series = sorted(df['series_number'].unique())
        colors = px.colors.qualitative.Plotly
        
        fig = go.Figure()
        
        # Plot each series separately
        for i, series_num in enumerate(unique_series):
            series_data = df[df['series_number'] == series_num]
            
            fig.add_trace(go.Scatter(
                x=series_data['date'],
                y=series_data['spread_bps'],
                mode='markers',
                name=f'Series {series_num}',
                marker=dict(
                    color=colors[i % len(colors)],
                    size=4,
                    opacity=0.7
                ),
                hovertemplate='Date: %{x}<br>Spread: %{y:.1f} bps<br>Series: ' + str(series_num) + '<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"{index_name} {tenor}: Series Roll History",
            xaxis_title="Date",
            yaxis_title="Spread (bps)",
            height=500,
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Series statistics
        st.subheader("Series Statistics")
        series_stats = []
        for series_num in unique_series:
            series_data = df[df['series_number'] == series_num]
            series_stats.append({
                'Series': f"S{series_num}",
                'Start Date': series_data['date'].min().strftime('%Y-%m-%d'),
                'End Date': series_data['date'].max().strftime('%Y-%m-%d'),
                'Avg Spread': round(series_data['spread_bps'].mean(), 1),
                'Min Spread': round(series_data['spread_bps'].min(), 1),
                'Max Spread': round(series_data['spread_bps'].max(), 1),
                'Data Points': len(series_data)
            })
        
        df_stats = pd.DataFrame(series_stats)
        st.dataframe(df_stats, hide_index=True)
    else:
        st.warning(f"No series data available for {index_name} {tenor}")

def plot_spread_comparison(db):
    """Compare spreads across different indices and tenors"""
    st.subheader("Multi-Index Spread Comparison")
    
    # Selection controls
    col1, col2 = st.columns(2)
    
    with col1:
        comparison_type = st.radio(
            "Comparison Type",
            ["Same Tenor, Different Indices", "Same Index, Different Tenors"],
            key="comp_type"
        )
    
    with col2:
        period = st.selectbox(
            "Time Period",
            ["1M", "3M", "6M", "1Y", "2Y"],
            index=2,
            key="comp_period"
        )
    
    # Calculate date range
    period_days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730}
    start_date = (datetime.now() - timedelta(days=period_days[period])).strftime('%Y-%m-%d')
    
    fig = go.Figure()
    series_list = []
    
    if comparison_type == "Same Tenor, Different Indices":
        tenor = st.selectbox("Select Tenor", ["1Y", "3Y", "5Y", "7Y", "10Y"], index=2, key="comp_tenor")
        indices = st.multiselect(
            "Select Indices",
            ["EU_IG", "EU_XO", "US_IG", "US_HY"],
            default=["EU_IG", "US_IG"],
            key="comp_indices"
        )
        
        for index_name in indices:
            df = db.query_historical_spreads(index_name, tenor, start_date=start_date)
            if not df.empty:
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['spread_bps'],
                    mode='lines',
                    name=f'{index_name}',
                    line=dict(width=2)
                ))
                series_list.append((index_name, df))
        
        title = f"Spread Comparison: {tenor} Tenor"
        
    else:  # Same Index, Different Tenors
        index_name = st.selectbox(
            "Select Index",
            ["EU_IG", "EU_XO", "US_IG", "US_HY"],
            key="comp_single_index"
        )
        tenors = st.multiselect(
            "Select Tenors",
            ["1Y", "3Y", "5Y", "7Y", "10Y"],
            default=["3Y", "5Y", "7Y"],
            key="comp_tenors"
        )
        
        for tenor in tenors:
            df = db.query_historical_spreads(index_name, tenor, start_date=start_date)
            if not df.empty:
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['spread_bps'],
                    mode='lines',
                    name=f'{tenor}',
                    line=dict(width=2)
                ))
                series_list.append((tenor, df))
        
        title = f"Spread Comparison: {index_name}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Spread (bps)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    if len(series_list) > 1 and st.checkbox("Show Correlation Matrix", key="show_corr"):
        st.subheader("Correlation Matrix")
        
        # Prepare data for correlation
        spread_dict = {}
        for name, df in series_list:
            df_indexed = df.set_index('date')['spread_bps']
            spread_dict[name] = df_indexed
        
        # Create combined dataframe
        combined_df = pd.DataFrame(spread_dict)
        combined_df = combined_df.dropna()
        
        if not combined_df.empty:
            # Calculate correlation
            corr_matrix = combined_df.corr()
            
            # Create heatmap
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(3),
                texttemplate='%{text}',
                textfont={"size": 12},
                colorbar=dict(title="Correlation")
            ))
            
            fig_corr.update_layout(
                title="Correlation Matrix",
                height=400
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)

def render_data_quality(db):
    """Render data quality analysis"""
    st.subheader("Data Quality Analysis")
    
    # Overall statistics
    try:
        # Get total records
        query = "SELECT COUNT(*) FROM raw_historical_spreads"
        total_records = pd.read_sql_query(query, db.conn).iloc[0, 0]
        
        # Get date range
        query = """
            SELECT MIN(date) as min_date, MAX(date) as max_date
            FROM raw_historical_spreads
        """
        date_range = pd.read_sql_query(query, db.conn)
        
        # Get unique counts
        query = """
            SELECT 
                COUNT(DISTINCT index_name) as indices,
                COUNT(DISTINCT tenor) as tenors,
                COUNT(DISTINCT series_number) as series
            FROM raw_historical_spreads
        """
        counts = pd.read_sql_query(query, db.conn)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{total_records:,}")
        with col2:
            st.metric("Date Range", f"{date_range['min_date'].iloc[0]} to {date_range['max_date'].iloc[0]}")
        with col3:
            st.metric("Indices", counts['indices'].iloc[0])
        with col4:
            st.metric("Tenors", counts['tenors'].iloc[0])
        
        # Detailed breakdown
        st.subheader("Data Coverage by Index and Tenor")
        
        query = """
            SELECT 
                index_name,
                tenor,
                COUNT(*) as records,
                MIN(date) as first_date,
                MAX(date) as last_date,
                ROUND(AVG(spread_bps), 1) as avg_spread
            FROM raw_historical_spreads
            GROUP BY index_name, tenor
            ORDER BY index_name, tenor
        """
        
        coverage_df = pd.read_sql_query(query, db.conn)
        
        # Create pivot table for heatmap
        pivot_df = coverage_df.pivot_table(
            values='records', 
            index='index_name', 
            columns='tenor', 
            fill_value=0
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='YlOrRd',
            text=pivot_df.values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Records")
        ))
        
        fig.update_layout(
            title="Data Points by Index and Tenor",
            xaxis_title="Tenor",
            yaxis_title="Index",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed table
        if st.checkbox("Show Detailed Coverage Table", key="show_coverage"):
            st.dataframe(coverage_df, hide_index=True)
            
    except Exception as e:
        st.error(f"Error analyzing data quality: {e}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("CDS Monitor - Spread Analysis Dashboard")
    
    # Sidebar info
    with st.sidebar:
        st.header("Dashboard Info")
        st.info("""
        This dashboard provides comprehensive analysis of CDS spreads:
        
        • Historical Evolution: Track spread movements over time
        • Current Curves: Compare term structures
        • Series Roll: Analyze series transitions
        • Comparisons: Multi-index/tenor analysis
        • Data Quality: Monitor data coverage
        """)
        
        st.divider()
        
        # Database info
        st.subheader("Database Status")
        try:
            db = RawDatabase(RAW_DB_PATH)
            query = "SELECT COUNT(*) FROM raw_historical_spreads"
            count = pd.read_sql_query(query, db.conn).iloc[0, 0]
            st.success(f"Connected to database\n{count:,} records available")
        except Exception as e:
            st.error(f"Database connection failed\n{str(e)}")
    
    # Main content
    render_spread_analysis()

if __name__ == "__main__":
    main()