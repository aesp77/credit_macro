# CDS Macro Monitor

A production-ready Python system for monitoring and trading CDS indices, replacing and enhancing the functionality of the legacy Excel-based monitor.

## Overview

This workspace provides a comprehensive framework for:
- **Real-time monitoring** of CDS indices (ITRX Europe Main/Crossover, CDX IG/HY)
- **Strategy implementation** (5s10s steepeners, compression trades, butterflies)
- **P&L tracking and reconstruction** with historical data fixes
- **Risk analytics** (DV01, carry, roll-down calculations)
- **Machine learning experiments** for spread prediction and regime detection

## Project Structure

```
credit_macro/
├── data/                   # Data storage
│   ├── raw/               # Raw Bloomberg data exports
│   ├── processed/         # Cleaned and processed datasets
│   └── cds_monitor.db     # SQLite database (auto-created)
│
├── output/                 # Generated outputs
│   ├── reports/           # Daily P&L and risk reports
│   ├── charts/            # Visualizations
│   └── dashboards/        # Streamlit/Dash apps
│
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb     # Initial data analysis
│   ├── 02_strategy_backtesting.ipynb # Strategy performance testing
│   ├── 03_ml_experiments.ipynb       # Machine learning models
│   └── 04_pnl_reconstruction.ipynb   # Fix historical P&L issues
│
├── src/                    # Source code
│   ├── data/              # Data layer
│   │   ├── bloomberg_connector.py    # Bloomberg API interface
│   │   ├── data_manager.py          # High-level data operations
│   │   └── cache.py                 # Caching utilities
│   │
│   ├── models/            # Data models
│   │   ├── enums.py                # Region, Market, Tenor, Side
│   │   ├── cds_index.py            # Index definitions
│   │   ├── spread_data.py          # Market data structures
│   │   ├── curve.py                # Term structure models
│   │   ├── position.py             # Positions and strategies
│   │   └── database.py             # Database persistence
│   │
│   └── utils/             # Utilities
│       └── logger.py               # Logging configuration
│
├── test_setup.py          # Verification script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Quick Start

### Prerequisites

- Python 3.11+ 
- Bloomberg Terminal with API access
- Access to `S:\CSA\cds_tickers_bbg.csv` (or similar ticker file)

### Installation

1. **Clone and navigate to the project:**
```bash
cd C:\source\repos\psc\packages\psc_csa_tools\credit_macro
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv

# On Windows PowerShell:
.\venv\Scripts\Activate

# On Windows Command Prompt:
venv\Scripts\activate.bat
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set Keras backend (for ML models):**
```powershell
# PowerShell
$env:KERAS_BACKEND = "torch"

# Or add to your environment permanently
```

5. **Verify installation:**
```bash
python test_setup.py
```

You should see all tests passing:
```
Models imported successfully
Data layer imported successfully
Utils imported successfully
All tests passed! Your setup is ready.
```

## Notebooks Guide

The `notebooks/` directory contains Jupyter notebooks for different workflows:

### 1. **01_data_exploration.ipynb**
- Connect to Bloomberg and fetch current spreads
- Explore index compositions and historical data
- Analyze spread relationships and basis
- Identify data quality issues from the legacy Excel system

### 2. **02_strategy_backtesting.ipynb**
- Implement and test trading strategies:
  - 5s10s steepeners/flatteners
  - Main vs Crossover compression
  - Butterfly trades (3s5s7s)
- Calculate historical P&L
- Analyze strategy performance metrics
- Generate risk reports

### 3. **03_ml_experiments.ipynb**
- Build spread prediction models using Keras-Core
- Implement regime detection algorithms
- Feature engineering from macro data
- Train and validate ML models
- Generate trading signals

### 4. **04_pnl_reconstruction.ipynb**
- Fix historical data issues from Excel
- Reconstruct accurate P&L time series
- Handle series rolls and data gaps
- Validate against known benchmarks

### To use notebooks:
```bash
jupyter notebook
# or
jupyter lab
```

## Basic Usage

### Connect to Bloomberg and Fetch Data

```python
from src.data import BloombergCDSConnector, CDSDataManager

# Initialize connector
connector = BloombergCDSConnector()

# Get current spread for ITRX EUR CDSI S41 5Y
ticker = connector.get_index_ticker('EU', 'IG', 41, '5Y')
spread_data = connector.get_current_spread(ticker)
print(f"Current spread: {spread_data['PX_LAST'].iloc[0]} bps")

# Use the data manager for higher-level operations
manager = CDSDataManager(connector)
main_indices = manager.load_main_indices()  # Loads ITXEB and ITXEX
```

### Create and Track a Strategy

```python
from src.models import Position, Strategy, Side
from datetime import datetime

# Create a 5s10s steepener
strategy = Strategy(
    name="EU_IG_5s10s_Steepener",
    strategy_type="5s10s",
    positions=[
        Position(
            index_id="EU_IG_S41_5Y",
            side=Side.SELL,  # Sell 5Y
            notional=10_000_000,
            entry_date=datetime.now(),
            entry_spread=51.0,
            entry_dv01=4500
        ),
        Position(
            index_id="EU_IG_S41_10Y", 
            side=Side.BUY,  # Buy 10Y
            notional=5_300_000,  # DV01-weighted
            entry_date=datetime.now(),
            entry_spread=65.0,
            entry_dv01=8500
        )
    ],
    creation_date=datetime.now()
)

print(f"Strategy net DV01: {strategy.net_dv01}")
print(f"Is DV01-neutral: {strategy.is_dv01_neutral()}")
```

### Build Credit Curves

```python
from src.models import Region, Market

# Build full term structure
curve = manager.build_curve(
    region=Region.EU,
    market=Market.IG,
    series=41
)

# Interpolate for any maturity
spread_4y = curve.interpolate_spread(4.0)
print(f"4Y spread (interpolated): {spread_4y:.2f} bps")

# Calculate roll-down
rolldown = curve.calculate_rolldown(horizon_days=90)
print(f"90-day roll-down: {rolldown}")
```

## Launching the Streamlit Dashboard

The CDS Monitor includes a comprehensive Streamlit web dashboard for real-time monitoring and analysis.

### One-Click Launch (Easiest)

Use the provided launch scripts to update databases and start the dashboard in one step:

**PowerShell (Recommended):**
```powershell
.\launch_dashboard.ps1
```

**Command Prompt / Batch:**
```cmd
launch_dashboard.bat
```

These scripts will:
1. Update both databases with latest Bloomberg data
2. Launch the Streamlit dashboard automatically
3. Open your browser to the dashboard

### Manual Launch

If you prefer more control:

```bash
# Step 1: Update databases with latest data (recommended before each session)
poetry run python update_databases.py

# Step 2: Launch the Streamlit app
cd src/apps
poetry run streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501` (or `8502` if port is occupied).

### Dashboard Features

The dashboard includes multiple pages:

1. **Home** - Overview and quick actions
2. **Strategy Monitor** - Advanced strategy analysis with P&L decomposition
3. **Spread Analysis** - Real-time spread monitoring and historical charts
4. **Series Monitor** - Track series rolls and compare on/off-the-run spreads
5. **ML Analysis** - Machine learning models and predictions
6. **Option Pricing** - CDS index option valuation

### Best Practices

**Before Trading Day:**
```bash
# Morning routine - use one-click launch
.\launch_dashboard.ps1
```

**During Trading Day:**
- Dashboard refreshes automatically on page interactions
- Use "Rerun" button in Streamlit to refresh data
- For critical updates, re-run `update_databases.py` and refresh browser

**After Market Close:**
- Update databases with final marks: `poetry run python update_databases.py`
- Review strategy P&L in Strategy Monitor page
- Export reports if needed

## Updating Databases

The system uses two main databases that need to be updated regularly with the latest Bloomberg data:

### Quick Update (Recommended)

Use the automated update script to update both databases at once:

```bash
# Update with last 30 days of data (default)
poetry run python update_databases.py

# Update with custom lookback period
poetry run python update_databases.py --days-back 60

# Quiet mode (less verbose output)
poetry run python update_databases.py --quiet
```

This script will:
1. **Update Raw Database** (`data/raw/cds_indices_raw.db`) - Historical CDS spreads from Bloomberg
2. **Update TRS Database** (`data/processed/cds_trs.db`) - Total Return Swap calculations

### Manual Update

If you need to update databases individually:

**Update Raw Historical Spreads:**
```bash
cd credit_macro
poetry run python -c "from src.models.database import CDSDatabase; db = CDSDatabase(r'C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\raw\cds_indices_raw.db'); db.update_historical_data(days_back=30); db.close()"
```

**Update TRS Database:**
```bash
cd credit_macro/src
poetry run python -c "from models.trs import TRSDatabaseBuilder; builder = TRSDatabaseBuilder(r'C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\raw\cds_indices_raw.db', r'C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\processed\cds_trs.db'); builder.update_trs_database(days_back=30)"
```

### Database Update Schedule

For production use, consider updating databases:
- **Daily**: Before market open or after market close
- **Intraday**: If real-time data is needed for active trading
- **Weekly**: Minimum frequency to maintain data quality

### What Gets Updated

**Raw Database (`cds_indices_raw.db`)**:
- Historical spreads for all tracked indices (EU_IG, EU_XO, US_IG, US_HY)
- All tenors (3Y, 5Y, 7Y, 10Y)
- Series information and roll dates
- Only missing or new dates are fetched (incremental updates)

**TRS Database (`cds_trs.db`)**:
- Total return calculations for long and short positions
- Carry, mark-to-market P&L, and cumulative returns
- DV01 calculations with recovery rate assumptions
- Derived from raw database, so update raw database first

## Configuration

### Bloomberg Connection
The system uses `xbbg` for Bloomberg connectivity. Ensure:
1. Bloomberg Terminal is running
2. Bloomberg API is properly installed
3. You have appropriate data permissions

### Database Paths
The default database paths are:
- **Raw Database**: `data/raw/cds_indices_raw.db`
- **TRS Database**: `data/processed/cds_trs.db`

These can be customized in the individual page files if needed.

### Logging
Logs are written to `logs/` directory with rotation. Configure in `src/utils/logger.py`.

## Testing Guide

This guide will help you verify that all updates are working correctly.

### Quick Verification Checklist

Run through these tests to verify everything is working:

**Test 1: Verify Files Exist**

```powershell
cd C:\source\repos\psc\packages\psc_csa_tools\credit_macro

# Check new files
Test-Path update_databases.py          # Should be True
Test-Path launch_dashboard.ps1         # Should be True
Test-Path launch_dashboard.bat         # Should be True
```

**Test 2: Verify S44 Update in Streamlit App**

```powershell
# Check that S44 appears in the front page
Get-Content src\apps\streamlit_app.py | Select-String "S44"

# Expected output:
#     st.metric("Active Series", "S44")
```

**Test 3: Test Update Script**

```powershell
poetry run python update_databases.py --help

# Expected output shows usage and options
```

**Test 4: Verify Database Files Exist**

```powershell
# Check that databases exist
Test-Path data\raw\cds_indices_raw.db           # Should be True
Test-Path data\processed\cds_trs.db             # Should be True
```

### Full Integration Test

To test the complete workflow:

**Step 1: Update Databases**
```powershell
cd C:\source\repos\psc\packages\psc_csa_tools\credit_macro
poetry run python update_databases.py
```

**Expected Result:**
- ✓ Raw database updated with latest data
- ✓ TRS database recalculated
- ✓ Summary showing number of records updated

**Step 2: Launch Dashboard**
```powershell
cd src\apps
poetry run streamlit run streamlit_app.py
```

**Expected Result:**
- ✓ Streamlit app starts
- ✓ Browser opens to http://localhost:8501 or 8502
- ✓ Front page shows "Active Series: S44"
- ✓ All pages load without errors

**Step 3: Verify Dashboard Data**

In the browser:

1. **Home Page:**
   - Check "Active Series" shows "S44" ✓
   - Check "Database Status" shows "Connected" ✓
   - Check "Last Update" shows "Live" ✓

2. **Strategy Monitor Page:**
   - Navigate to Strategy Monitor
   - Check that data loads ✓
   - Try different index combinations ✓

3. **Spread Analysis Page:**
   - Navigate to Spread Analysis
   - Check charts render ✓
   - Verify latest dates appear ✓

4. **Series Monitor Page:**
   - Navigate to Series Monitor
   - Check S44 data is available ✓

### One-Click Test (Easiest)

The easiest way to test everything:

```powershell
cd C:\source\repos\psc\packages\psc_csa_tools\credit_macro
.\launch_dashboard.ps1
```

**This single command will:**
1. Update both databases automatically
2. Launch the Streamlit dashboard
3. Open your browser to the app

**Success Criteria:**
- ✓ Script runs without errors
- ✓ Dashboard opens in browser
- ✓ "Active Series" shows "S44"
- ✓ All pages load with data

### Quick Command Reference

```powershell
# Navigate to project
cd C:\source\repos\psc\packages\psc_csa_tools\credit_macro

# Update databases only
poetry run python update_databases.py

# Update with custom lookback
poetry run python update_databases.py --days-back 60

# Launch dashboard only (without updating)
cd src\apps
poetry run streamlit run streamlit_app.py

# One-click: Update + Launch
.\launch_dashboard.ps1

# One-click: Batch version
launch_dashboard.bat

# View help
poetry run python update_databases.py --help
```

### What Success Looks Like

After running all tests, you should have:

- ✅ All new files created (`update_databases.py`, `launch_dashboard.ps1`, `launch_dashboard.bat`)
- ✅ Front page showing "Active Series: S44"
- ✅ Both databases updated with latest data
- ✅ Streamlit dashboard running and showing current data
- ✅ All dashboard pages working correctly

## Key Features

### Replaces Excel Functionality
- Real-time spread monitoring (Main Sheet)
- Strategy tracking (Strategies sheet)
- Curve construction (Curves_master)
- Historical data storage
- P&L calculations

### Enhancements Over Excel
- 10x faster performance
- Automated data quality checks
- Machine learning predictions
- Robust error handling
- Version control friendly
- Concurrent strategy support
- RESTful API ready

## Troubleshooting

### Common Issues and Solutions

**Issue: "ModuleNotFoundError: No module named 'models'"**

*Solution:* Make sure you're in the correct directory
```powershell
cd C:\source\repos\psc\packages\psc_csa_tools\credit_macro
poetry run python update_databases.py
```

**Issue: "Bloomberg connection failed"**

*Solution:*
1. Ensure Bloomberg Terminal is running
2. Check Bloomberg API is installed (`pip install xbbg`)
3. Verify you have data permissions

**Issue: "Database file not found"**

*Solution:* Databases will be created automatically on first run. Just run:
```powershell
poetry run python update_databases.py
```

**Issue: Streamlit shows old series (S43 instead of S44)**

*Solution:*
1. Stop the Streamlit server (Ctrl+C)
2. Relaunch: `poetry run streamlit run streamlit_app.py`
3. Hard refresh browser (Ctrl+Shift+R)

**Issue: "Port already in use"**

*Solution:* Streamlit will automatically try next port (8502, 8503, etc.)

### Import Errors
```bash
# Ensure you're in the right directory
cd credit_macro
# Verify src is in Python path
python -c "import sys; print('src' in sys.path)"
```

### Bloomberg Connection Issues
```python
# Test Bloomberg connection directly
import xbbg
from xbbg import blp

# Simple test
test = blp.bdp("ITRX EUR CDSI S41 5Y Corp", "PX_LAST")
print(test)
```

### Database Issues
```python
# Reset database
import os
if os.path.exists("data/cds_monitor.db"):
    os.remove("data/cds_monitor.db")
# It will be recreated on next use
```

## Next Steps

1. **Test Bloomberg connectivity** with your specific indices
2. **Run data exploration notebook** to understand your data
3. **Configure strategies** in the strategy backtest notebook
4. **Set up automated daily reports** using the data manager
5. **Build ML models** for spread prediction
6. **Deploy dashboard** for real-time monitoring

## Documentation

- **Models**: See docstrings in `src/models/` for detailed API documentation
- **Bloomberg**: Refer to `xbbg` documentation for additional Bloomberg functions
- **ML Models**: Keras-Core documentation for model building

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests: `python test_setup.py`
4. Submit a pull request

## License

Internal use only - Property of PSC

## Contact

For questions or issues, contact the CSA team.

---

**Version**: 1.0.0  
**Last Updated**: September 2025  
**Status**: Production Ready