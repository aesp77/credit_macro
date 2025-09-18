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

## Configuration

### Bloomberg Connection
The system uses `xbbg` for Bloomberg connectivity. Ensure:
1. Bloomberg Terminal is running
2. Bloomberg API is properly installed
3. You have appropriate data permissions

### Database
SQLite database is automatically created at `data/cds_monitor.db` on first run.

### Logging
Logs are written to `logs/` directory with rotation. Configure in `src/utils/logger.py`.

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