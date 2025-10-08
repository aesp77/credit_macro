import streamlit as st

st.set_page_config(
    page_title="CDS Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("CDS Monitor Dashboard")
st.sidebar.success("Select a page above.")

st.markdown("""
## Welcome to the CDS Monitor

This comprehensive dashboard provides real-time monitoring and analysis of CDS indices and trading strategies.

### Available Pages:

#### **Strategy Monitor**
- Advanced strategy analysis with convexity and carry/roll decomposition
- Support for curve trades (3s5s, 5s7s, 5s10s, 7s10s, custom)
- Compression/decompression trades between indices
- Custom multi-leg strategies with flexible weighting
- Historical performance and statistical analysis

#### **Spread Analysis**
- Real-time spread monitoring across all indices and tenors
- Historical spread evolution and curve dynamics
- Relative value analysis and Z-score tracking
- Cross-market spread comparisons

#### **Series Monitor**
- Track series rolls and on-the-run vs off-the-run spreads
- Series comparison (S43, S42, S41, etc.)
- DV01 calculations and curve metrics
- Forward spread analysis and carry calculations

### Data Sources:
- **Raw Database**: Historical spreads and series information
- **TRS Database**: Total return swap calculations with roll adjustments
- **Live Data**: Integration ready for Bloomberg feeds

### Key Features:
- All calculations use actual market data (no hardcoded values)
- DV01-neutral, beta-weighted, and manual position sizing
- Proper carry/roll decomposition based on TRS methodology
- Statistical analysis including mean reversion and convexity metrics

Select a page from the sidebar to begin your analysis.
""")

# Add a summary metrics section
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Last Update", "Live")
    
with col2:
    st.metric("Indices Tracked", "6")
    
with col3:
    st.metric("Active Series", "S44")
    
with col4:
    st.metric("Database Status", "Connected")

st.divider()

# Quick links or recent alerts
st.subheader("Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("View EU_IG 5s10s Steepener"):
        st.switch_page("pages/1_Strategy_Monitor.py")
        
with col2:
    if st.button("Check Compression Trades"):
        st.switch_page("pages/1_Strategy_Monitor.py")
        
with col3:
    if st.button("Series Roll Analysis"):
        st.switch_page("pages/3_Series_Monitor.py")