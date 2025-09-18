"""
Total Return Series calculations and database
Migrated from notebooks/total_return.ipynb
TO BE REVIEWED FOR REDUNDANCIES
"""

import pandas as pd
import numpy as np
from datetime import datetime



# Clean CDS Total Return Series Calculator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.models.database import CDSDatabase

def calculate_cds_total_return(spread_data: pd.DataFrame, 
                              notional: float = 10_000_000,
                              recovery_rate: float = 0.40) -> pd.DataFrame:
    """
    Calculate CDS Total Return Series for both long and short positions
    
    Args:
        spread_data: DataFrame with columns ['date', 'spread_bps', 'series_number']
        notional: Notional amount (default $10MM)
        recovery_rate: Recovery rate assumption for DV01 calculation
        
    Returns:
        DataFrame with long_tr and short_tr columns
    """
    
    if spread_data.empty:
        return pd.DataFrame()
    
    df = spread_data.copy().sort_values('date').reset_index(drop=True)
    
    # Calculate approximate DV01 for 5Y CDS
    years_to_maturity = 5.0
    df['dv01_approx'] = (1 - recovery_rate) * df['spread_bps'] * years_to_maturity * notional / 10000
    
    # Calculate daily changes
    df['spread_change'] = df['spread_bps'].diff()
    df['daily_carry'] = df['spread_bps'] / 360
    
    # Mark-to-market P&L
    df['mtm_pnl_long'] = -df['spread_change'] * df['dv01_approx'] / 100
    df['mtm_pnl_short'] = df['spread_change'] * df['dv01_approx'] / 100
    
    # Roll adjustments
    df['is_roll_date'] = df['series_number'] != df['series_number'].shift(1)
    df['roll_cost'] = 0.0
    
    roll_dates = df[df['is_roll_date'] & (df.index > 0)]
    for idx in roll_dates.index:
        if idx > 0:
            spread_jump = df.loc[idx, 'spread_bps'] - df.loc[idx-1, 'spread_bps']
            transaction_cost = 0.1
            df.loc[idx, 'roll_cost'] = (transaction_cost + abs(spread_jump) * 0.1) * df.loc[idx, 'dv01_approx'] / 100
    
    # Total daily P&L
    df['long_daily_pnl'] = (df['daily_carry'] * notional / 10000) + df['mtm_pnl_long'] - df['roll_cost']
    df['short_daily_pnl'] = -(df['daily_carry'] * notional / 10000) + df['mtm_pnl_short'] - df['roll_cost']
    
    # Daily returns
    df['long_daily_return'] = df['long_daily_pnl'] / notional
    df['short_daily_return'] = df['short_daily_pnl'] / notional
    
    # Cumulative total return indices (starting at 100)
    df['long_tr'] = 100.0
    df['short_tr'] = 100.0
    
    for i in range(1, len(df)):
        df.loc[i, 'long_tr'] = df.loc[i-1, 'long_tr'] * (1 + df.loc[i, 'long_daily_return'])
        df.loc[i, 'short_tr'] = df.loc[i-1, 'short_tr'] * (1 + df.loc[i, 'short_daily_return'])
    
    return df

def test_trs_calculation():
    """Test TRS calculation on database data"""
    
    # Use correct database path
    db_path = "C:/source/repos/psc/packages/psc_csa_tools/credit_macro/data/raw/cds_indices_raw.db"
    db = CDSDatabase(db_path)
    
    # Check database structure
    tables = db.conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    print(f"Database tables: {[t[0] for t in tables]}")
    
    if 'raw_historical_spreads' not in [t[0] for t in tables]:
        print("ERROR: raw_historical_spreads table not found")
        db.close()
        return None
    
    # Get test data
    test_data = db.query_historical_spreads('EU_IG', '5Y', '2023-01-01', '2024-12-31')
    
    if test_data.empty:
        print("No test data found")
        db.close()
        return None
    
    print(f"Retrieved {len(test_data)} data points")
    print(f"Date range: {test_data['date'].min()} to {test_data['date'].max()}")
    
    # Calculate TRS
    trs_data = calculate_cds_total_return(test_data)
    
    if trs_data.empty:
        print("TRS calculation failed")
        db.close()
        return None
    
    # Summary results
    long_total_return = (trs_data['long_tr'].iloc[-1] / trs_data['long_tr'].iloc[0] - 1) * 100
    short_total_return = (trs_data['short_tr'].iloc[-1] / trs_data['short_tr'].iloc[0] - 1) * 100
    roll_count = trs_data['is_roll_date'].sum()
    
    print(f"Long Total Return: {long_total_return:.2f}%")
    print(f"Short Total Return: {short_total_return:.2f}%") 
    print(f"Series rolls detected: {roll_count}")
    
    # Basic validation
    correlation = np.corrcoef(trs_data['long_daily_return'], trs_data['short_daily_return'])[0,1]
    print(f"Long/Short correlation: {correlation:.3f}")
    
    # Simple plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(trs_data['date'], trs_data['spread_bps'])
    plt.title('Spread Evolution')
    plt.ylabel('Spread (bps)')
    
    plt.subplot(1, 2, 2) 
    plt.plot(trs_data['date'], trs_data['long_tr'], label='Long TRS')
    plt.plot(trs_data['date'], trs_data['short_tr'], label='Short TRS')
    plt.title('Total Return Series')
    plt.ylabel('Index Level')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    db.close()
    return trs_data

def test_multiple_indices():
    """Test across multiple indices"""
    
    db_path = "C:/source/repos/psc/packages/psc_csa_tools/credit_macro/data/raw/cds_indices_raw.db"
    db = CDSDatabase(db_path)
    
    test_cases = [('US_HY', '5Y'), ('EU_XO', '5Y'), ('US_IG', '5Y')]
    results = {}
    
    for index_name, tenor in test_cases:
        data = db.query_historical_spreads(index_name, tenor, '2024-01-01')
        
        if not data.empty:
            trs = calculate_cds_total_return(data)
            if not trs.empty:
                long_ret = (trs['long_tr'].iloc[-1] / trs['long_tr'].iloc[0] - 1) * 100
                short_ret = (trs['short_tr'].iloc[-1] / trs['short_tr'].iloc[0] - 1) * 100
                results[f"{index_name}_{tenor}"] = {
                    'long_return': long_ret,
                    'short_return': short_ret,
                    'data_points': len(trs)
                }
                print(f"{index_name} {tenor}: Long {long_ret:+.2f}%, Short {short_ret:+.2f}%")
    
    db.close()
    return results

# Run tests
if __name__ == "__main__":
    main_result = test_trs_calculation()
    multi_results = test_multiple_indices()
    
    print("TRS calculation test complete")


# Corrected CDS Relative Value Strategy Calculator
# Proper implementation of all 4 sizing methods

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def interpolate_curve_rolldown(curve_spreads: dict, current_tenor: str, months: int) -> float:
    """
    Calculate rolldown by interpolating curve
    
    Args:
        curve_spreads: Dict with tenor keys ('1Y', '3Y', '5Y', '7Y', '10Y') and spread values
        current_tenor: Current position tenor (e.g., '5Y')
        months: Number of months to roll down
        
    Returns:
        Rolldown in bps (positive = beneficial rolldown)
    """
    
    # Convert tenor to years
    tenor_map = {'1Y': 1, '3Y': 3, '5Y': 5, '7Y': 7, '10Y': 10}
    current_years = tenor_map.get(current_tenor, 5)
    target_years = current_years - (months / 12)
    
    if target_years <= 0:
        return 0.0
    
    # Get available tenors and spreads for interpolation
    available_years = []
    available_spreads = []
    
    for tenor, years in tenor_map.items():
        if tenor in curve_spreads and not pd.isna(curve_spreads[tenor]):
            available_years.append(years)
            available_spreads.append(curve_spreads[tenor])
    
    if len(available_years) < 2:
        return 0.0
    
    # Linear interpolation
    target_spread = np.interp(target_years, available_years, available_spreads)
    current_spread = curve_spreads.get(current_tenor, 0)
    
    # Rolldown = current_spread - interpolated_spread (positive = tightening = gain for long)
    rolldown = current_spread - target_spread
    
    return rolldown

def calculate_corrected_strategy_metrics(long_data: pd.DataFrame, short_data: pd.DataFrame, 
                                       long_notional: float = 10_000_000,
                                       connector=None) -> dict:
    """
    Calculate relative value strategy metrics with corrected methods
    
    Args:
        long_data: EU_XO TRS data
        short_data: EU_IG TRS data  
        long_notional: Fixed notional for long leg
        connector: Bloomberg connector for real DV01 and curve data
        
    Returns:
        Dictionary with corrected sizing recommendations
    """
    
    # Merge data on common dates
    merged = pd.merge(long_data[['date', 'spread_bps']], 
                     short_data[['date', 'spread_bps']], 
                     on='date', suffixes=('_xo', '_ig'))
    
    if merged.empty:
        return {"error": "No overlapping dates"}
    
    # Current market data
    current = merged.iloc[-1]
    xo_spread = current['spread_bps_xo']
    ig_spread = current['spread_bps_ig']
    
    # Calculate historical beta - CORRECTED
    merged['xo_change'] = merged['spread_bps_xo'].diff()
    merged['ig_change'] = merged['spread_bps_ig'].diff()
    
    valid_data = merged.dropna()
    if len(valid_data) > 30:
        # Correct regression: XO_change = alpha + beta * IG_change
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_data['ig_change'], valid_data['xo_change'])
        beta = slope
        correlation = r_value
    else:
        beta = 4.0  # Assume XO moves 4x IG if insufficient data
        correlation = 0.9
    
    # Get real Bloomberg DV01 data if connector available
    if connector:
        try:
            xo_dv01 = connector.get_dv01("ITXEX543 Curncy")  # EU_XO 5Y
            ig_dv01 = connector.get_dv01("ITXEB543 Curncy")  # EU_IG 5Y
            
            # Get curve data for rolldown calculation
            xo_curve_data = connector.get_curve_data("ITRX EUR XOVER", series=43)
            ig_curve_data = connector.get_curve_data("ITRX EUR CDSI", series=43)
            
            # Calculate rolldown (3-month example)
            xo_rolldown = interpolate_curve_rolldown(xo_curve_data, "5Y", 3)
            ig_rolldown = interpolate_curve_rolldown(ig_curve_data, "5Y", 3)
            
        except Exception as e:
            print(f"Bloomberg data unavailable: {e}")
            # Use reasonable defaults for 5Y CDS
            xo_dv01 = 50000  # ~$50K per bp for 5Y CDS
            ig_dv01 = 48000  # Slightly lower for IG vs XO
            xo_rolldown = 2.0  # 2bp rolldown estimate
            ig_rolldown = 1.0  # 1bp rolldown estimate
    else:
        # Use reasonable defaults without connector
        xo_dv01 = 50000  # ~$50K per bp for 5Y CDS
        ig_dv01 = 48000  # Slightly lower for IG vs XO
        xo_rolldown = 2.0  # 2bp rolldown estimate
        ig_rolldown = 1.0  # 1bp rolldown estimate
    
    # Sizing calculations - CORRECTED
    sizing_methods = {}
    
    # Method 1: 1-for-1 notional - SIMPLE
    sizing_methods['1_for_1'] = {
        'long_notional': long_notional,
        'short_notional': long_notional,
        'description': '1-for-1 notional'
    }
    
    # Method 2: Beta-adjusted - CORRECTED
    # If IG moves 1bp, XO moves beta bps
    # To hedge $10MM XO exposure, need $10MM * beta of IG short
    beta_adjusted_short = long_notional * beta
    sizing_methods['beta_adjusted'] = {
        'long_notional': long_notional,
        'short_notional': beta_adjusted_short,
        'beta': beta,
        'description': f'Beta-adjusted ({beta:.2f})'
    }
    
    # Method 3: DV01 neutral - CORRECTED
    # Equal dollar DV01 exposure
    dv01_ratio = xo_dv01 / ig_dv01 if ig_dv01 != 0 else 1.0
    dv01_neutral_short = long_notional * dv01_ratio
    sizing_methods['dv01_neutral'] = {
        'long_notional': long_notional,
        'short_notional': dv01_neutral_short,
        'dv01_ratio': dv01_ratio,
        'xo_dv01': xo_dv01,
        'ig_dv01': ig_dv01,
        'description': f'DV01 neutral ({dv01_ratio:.2f})'
    }
    
    # Method 4: Carry neutral - CORRECTED with rolldown
    # Total carry = coupon carry + rolldown
    xo_total_carry = (xo_spread / 4) + xo_rolldown  # Quarterly
    ig_total_carry = (ig_spread / 4) + ig_rolldown
    
    carry_ratio = xo_total_carry / ig_total_carry if ig_total_carry != 0 else 1.0
    carry_neutral_short = long_notional * carry_ratio
    sizing_methods['carry_neutral'] = {
        'long_notional': long_notional,
        'short_notional': carry_neutral_short,
        'carry_ratio': carry_ratio,
        'xo_total_carry': xo_total_carry,
        'ig_total_carry': ig_total_carry,
        'xo_rolldown': xo_rolldown,
        'ig_rolldown': ig_rolldown,
        'description': f'Carry neutral ({carry_ratio:.2f})'
    }
    
    # Calculate expected P&L for each method
    for method_name, method in sizing_methods.items():
        long_size = method['long_notional']
        short_size = method['short_notional']
        
        # 3-month carry P&L
        if method_name == 'carry_neutral':
            xo_carry = method['xo_total_carry']
            ig_carry = method['ig_total_carry']
        else:
            xo_carry = xo_spread / 4  # Just coupon carry
            ig_carry = ig_spread / 4
        
        carry_pnl_long = (xo_carry / 10000) * long_size
        carry_pnl_short = -(ig_carry / 10000) * short_size  # Short pays carry
        total_carry = carry_pnl_long + carry_pnl_short
        
        # Net DV01 (dollars per bp)
        net_dv01 = (xo_dv01 * long_size / 10000000) - (ig_dv01 * short_size / 10000000)
        
        method.update({
            'carry_3m': total_carry,
            'net_dv01': net_dv01,
            'long_carry_3m': carry_pnl_long,
            'short_carry_3m': carry_pnl_short
        })
    
    return {
        'current_data': {
            'xo_spread': xo_spread,
            'ig_spread': ig_spread,
            'basis': xo_spread - ig_spread,
            'xo_dv01': xo_dv01,
            'ig_dv01': ig_dv01,
            'date': current['date']
        },
        'historical_metrics': {
            'beta': beta,
            'correlation': correlation,
            'data_points': len(valid_data)
        },
        'sizing_methods': sizing_methods
    }

def test_corrected_calculations():
    """Test the corrected relative value calculations"""
    
    # Load data
    db_path = "C:/source/repos/psc/packages/psc_csa_tools/credit_macro/data/raw/cds_indices_raw.db"
    
    import sys
    sys.path.append('../src')
    from src.models.database import CDSDatabase
    
    db = CDSDatabase(db_path)
    
    # Get data for both indices
    xo_data = db.query_historical_spreads('EU_XO', '5Y', '2023-01-01')
    ig_data = db.query_historical_spreads('EU_IG', '5Y', '2023-01-01')
    
    if xo_data.empty or ig_data.empty:
        print("Data unavailable")
        return None
    
    print(f"Data points - XO: {len(xo_data)}, IG: {len(ig_data)}")
    
    # Calculate strategy metrics with corrected methods
    strategy_metrics = calculate_corrected_strategy_metrics(xo_data, ig_data)
    
    # Display results
    print("\nCORRECTED RELATIVE VALUE ANALYSIS")
    print("Long EU_XO 5Y vs Short EU_IG 5Y")
    print("="*50)
    
    current = strategy_metrics['current_data']
    print(f"Current EU_XO spread: {current['xo_spread']:.1f} bps")
    print(f"Current EU_IG spread: {current['ig_spread']:.1f} bps")
    print(f"Current basis: {current['basis']:.1f} bps")
    
    hist = strategy_metrics['historical_metrics']
    print(f"Historical beta (XO vs IG): {hist['beta']:.3f}")
    print(f"Correlation: {hist['correlation']:.3f}")
    
    print(f"\nSIZING METHODS (Long $10MM XO):")
    print("-"*50)
    
    for method_name, method in strategy_metrics['sizing_methods'].items():
        print(f"\n{method_name.upper()}:")
        print(f"  {method['description']}")
        print(f"  Long EU_XO:  ${method['long_notional']:,.0f}")
        print(f"  Short EU_IG: ${method['short_notional']:,.0f}")
        print(f"  Hedge ratio: {method['short_notional']/method['long_notional']:.2f}")
        print(f"  3M Carry P&L: ${method['carry_3m']:,.0f}")
        print(f"  Net DV01: ${method['net_dv01']:,.0f}")
    
    db.close()
    return strategy_metrics

# Test the corrected calculations
if __name__ == "__main__":
    results = test_corrected_calculations()


# TRS Database Builder
# Creates Total Return Series database for all CDS indices and tenors

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from src.models.database import CDSDatabase

class TRSDatabaseBuilder:
    """Build and maintain TRS database for all CDS indices"""
    
    def __init__(self, raw_db_path: str, trs_db_path: str):
        """
        Initialize TRS database builder
        
        Args:
            raw_db_path: Path to raw CDS spreads database
            trs_db_path: Path to output TRS database
        """
        self.raw_db_path = raw_db_path
        self.trs_db_path = trs_db_path
        
        # Ensure output directory exists
        Path(self.trs_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Define all indices and tenors to process
        self.indices = {
            'EU_IG': {'base': 'ITRX EUR CDSI', 'recovery': 0.40},
            'EU_XO': {'base': 'ITRX EUR XOVER', 'recovery': 0.40},
            'US_IG': {'base': 'CDX IG', 'recovery': 0.40},
            'US_HY': {'base': 'CDX HY', 'recovery': 0.40},
            'EU_SUBFIN': {'base': 'ITRX EUR SUBFIN', 'recovery': 0.40},
            'EU_SENFIN': {'base': 'ITRX EUR SENFIN', 'recovery': 0.40}
        }
        
        self.tenors = ['3Y', '5Y', '7Y', '10Y']
        
    def calculate_trs(self, spread_data: pd.DataFrame, 
                     tenor: str,
                     recovery_rate: float = 0.40,
                     notional: float = 10_000_000) -> pd.DataFrame:
        """
        Calculate Total Return Series for given spread data
        
        Returns DataFrame with columns:
        - date, spread_bps, series_number
        - long_tr, short_tr (indexed to 100)
        - long_pnl, short_pnl (cumulative P&L)
        - daily_return_long, daily_return_short
        """
        
        if spread_data.empty:
            return pd.DataFrame()
        
        df = spread_data.copy().sort_values('date').reset_index(drop=True)
        
        # Extract years from tenor
        tenor_years = float(tenor.replace('Y', ''))
        
        # Calculate approximate DV01
        df['dv01_approx'] = (1 - recovery_rate) * df['spread_bps'] * tenor_years * notional / 10000
        
        # Calculate daily changes
        df['spread_change'] = df['spread_bps'].diff()
        df['daily_carry'] = df['spread_bps'] / 360
        
        # Mark-to-market P&L
        df['mtm_pnl_long'] = -df['spread_change'] * df['dv01_approx'] / 100
        df['mtm_pnl_short'] = df['spread_change'] * df['dv01_approx'] / 100
        
        # Roll adjustments
        df['is_roll_date'] = df['series_number'] != df['series_number'].shift(1)
        df['roll_cost'] = 0.0
        
        roll_dates = df[df['is_roll_date'] & (df.index > 0)]
        for idx in roll_dates.index:
            if idx > 0:
                spread_jump = df.loc[idx, 'spread_bps'] - df.loc[idx-1, 'spread_bps']
                transaction_cost = 0.5  # bps
                df.loc[idx, 'roll_cost'] = (transaction_cost + abs(spread_jump) * 0.1) * df.loc[idx, 'dv01_approx'] / 100
        
        # Total daily P&L
        df['long_daily_pnl'] = (df['daily_carry'] * notional / 10000) + df['mtm_pnl_long'] - df['roll_cost']
        df['short_daily_pnl'] = -(df['daily_carry'] * notional / 10000) + df['mtm_pnl_short'] - df['roll_cost']
        
        # Daily returns
        df['daily_return_long'] = df['long_daily_pnl'] / notional
        df['daily_return_short'] = df['short_daily_pnl'] / notional
        
        # Cumulative total return indices (starting at 100)
        df['long_tr'] = 100.0
        df['short_tr'] = 100.0
        
        for i in range(1, len(df)):
            df.loc[i, 'long_tr'] = df.loc[i-1, 'long_tr'] * (1 + df.loc[i, 'daily_return_long'])
            df.loc[i, 'short_tr'] = df.loc[i-1, 'short_tr'] * (1 + df.loc[i, 'daily_return_short'])
        
        # Cumulative P&L
        df['long_pnl'] = df['long_daily_pnl'].cumsum()
        df['short_pnl'] = df['short_daily_pnl'].cumsum()
        
        # Keep essential columns
        result = df[['date', 'spread_bps', 'series_number', 
                    'long_tr', 'short_tr', 'long_pnl', 'short_pnl',
                    'daily_return_long', 'daily_return_short']].copy()
        
        return result
    
    def build_trs_database(self, start_date: str = '2020-01-01', verbose: bool = False):
        """
        Build complete TRS database for all indices and tenors
        """
        
        # Connect to raw database
        raw_db = CDSDatabase(self.raw_db_path)
        
        # Create TRS database
        trs_conn = sqlite3.connect(self.trs_db_path)
        
        # Track processing
        processed = []
        failed = []
        
        for index_name, index_info in self.indices.items():
            for tenor in self.tenors:
                key = f"{index_name}_{tenor}"
                
                try:
                    # Get raw spread data
                    spread_data = raw_db.query_historical_spreads(
                        index_name, tenor, start_date
                    )
                    
                    if spread_data.empty:
                        failed.append(key)
                        continue
                    
                    # Calculate TRS
                    trs_data = self.calculate_trs(
                        spread_data, 
                        tenor,
                        recovery_rate=index_info['recovery']
                    )
                    
                    if trs_data.empty:
                        failed.append(key)
                        continue
                    
                    # Save to database
                    table_name = f"trs_{index_name.lower()}_{tenor.lower()}"
                    trs_data.to_sql(table_name, trs_conn, 
                                   if_exists='replace', index=False)
                    
                    processed.append(key)
                    
                    if verbose:
                        print(f"Processed {key}: {len(trs_data)} records")
                    
                except Exception as e:
                    failed.append(key)
                    if verbose:
                        print(f"Failed {key}: {e}")
        
        # Create metadata table
        metadata = pd.DataFrame({
            'index_name': [k.split('_')[0] + '_' + k.split('_')[1] for k in processed],
            'tenor': [k.split('_')[-1] for k in processed],
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'record_count': [0] * len(processed)
        })
        
        metadata.to_sql('metadata', trs_conn, if_exists='replace', index=False)
        
        # Create indices for faster queries
        cursor = trs_conn.cursor()
        for index_name in self.indices.keys():
            for tenor in self.tenors:
                table_name = f"trs_{index_name.lower()}_{tenor.lower()}"
                try:
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(date)")
                except:
                    pass
        
        trs_conn.commit()
        trs_conn.close()
        raw_db.close()
        
        return processed, failed
    
    def get_trs_data(self, index_name: str, tenor: str, 
                     start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve TRS data from database
        """
        conn = sqlite3.connect(self.trs_db_path)
        
        table_name = f"trs_{index_name.lower()}_{tenor.lower()}"
        
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
        except:
            df = pd.DataFrame()
        
        conn.close()
        return df
    
    def test_database(self):
        """
        Test the TRS database to ensure all expected series are present
        """
        conn = sqlite3.connect(self.trs_db_path)
        cursor = conn.cursor()
        
        # Get all tables
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'trs_%'").fetchall()
        table_list = [t[0] for t in tables]
        
        print(f"Total TRS tables: {len(table_list)}")
        
        # Check expected vs actual
        expected = []
        for index in self.indices.keys():
            for tenor in self.tenors:
                expected.append(f"trs_{index.lower()}_{tenor.lower()}")
        
        missing = [t for t in expected if t not in table_list]
        extra = [t for t in table_list if t not in expected]
        
        if missing:
            print(f"\nMissing tables: {len(missing)}")
            for m in missing:
                print(f"  - {m}")
        
        if extra:
            print(f"\nExtra tables: {len(extra)}")
            for e in extra:
                print(f"  + {e}")
        
        # Test data retrieval for each table
        test_results = {}
        for table in table_list:
            try:
                count_query = f"SELECT COUNT(*) as cnt, MIN(date) as min_date, MAX(date) as max_date FROM {table}"
                result = cursor.execute(count_query).fetchone()
                test_results[table] = {
                    'count': result[0],
                    'min_date': result[1],
                    'max_date': result[2]
                }
            except Exception as e:
                test_results[table] = {'error': str(e)}
        
        # Summary statistics
        total_records = sum(r['count'] for r in test_results.values() if 'count' in r)
        print(f"\nTotal records across all tables: {total_records:,}")
        
        # Show sample of data coverage
        print("\nData coverage sample:")
        for table, result in list(test_results.items())[:5]:
            if 'count' in result:
                print(f"  {table}: {result['count']} records ({result['min_date']} to {result['max_date']})")
        
        conn.close()
        
        return {
            'tables': len(table_list),
            'missing': missing,
            'test_results': test_results,
            'total_records': total_records
        }

# Main execution
if __name__ == "__main__":
    # Define paths
    raw_db_path = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\raw\cds_indices_raw.db"
    trs_db_path = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\processed\cds_trs.db"
    
    # Build TRS database
    print("Building TRS database...")
    builder = TRSDatabaseBuilder(raw_db_path, trs_db_path)
    processed, failed = builder.build_trs_database(start_date='2023-01-01', verbose=True)
    
    print(f"\nProcessed: {len(processed)} series")
    print(f"Failed: {len(failed)} series")
    
    # Run tests
    print("\nRunning database tests...")
    test_results = builder.test_database()
    
    # Test data retrieval
    print("\nTesting data retrieval...")
    test_data = builder.get_trs_data('EU_IG', '5Y', '2024-01-01', '2024-12-31')
    if not test_data.empty:
        print(f"Successfully retrieved {len(test_data)} records for EU_IG 5Y")
        print(f"Long TR range: {test_data['long_tr'].min():.2f} - {test_data['long_tr'].max():.2f}")
        print(f"Short TR range: {test_data['short_tr'].min():.2f} - {test_data['short_tr'].max():.2f}")


# TRS Database Builder
# Creates Total Return Series database for all CDS indices and tenors

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from src.models.database import CDSDatabase

class TRSDatabaseBuilder:
    """Build and maintain TRS database for all CDS indices"""
    
    def __init__(self, raw_db_path: str, trs_db_path: str):
        """
        Initialize TRS database builder
        
        Args:
            raw_db_path: Path to raw CDS spreads database
            trs_db_path: Path to output TRS database
        """
        self.raw_db_path = raw_db_path
        self.trs_db_path = trs_db_path
        
        # Ensure output directory exists
        Path(self.trs_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Define all indices and tenors to process
        self.indices = {
            'EU_IG': {'base': 'ITRX EUR CDSI', 'recovery': 0.40},
            'EU_XO': {'base': 'ITRX EUR XOVER', 'recovery': 0.40},
            'US_IG': {'base': 'CDX IG', 'recovery': 0.40},
            'US_HY': {'base': 'CDX HY', 'recovery': 0.40},
            'EU_SUBFIN': {'base': 'ITRX EUR SUBFIN', 'recovery': 0.40},
            'EU_SENFIN': {'base': 'ITRX EUR SENFIN', 'recovery': 0.40}
        }
        
        self.tenors = ['3Y', '5Y', '7Y', '10Y']
        
    def calculate_trs(self, spread_data: pd.DataFrame, 
                     tenor: str,
                     recovery_rate: float = 0.40,
                     notional: float = 10_000_000) -> pd.DataFrame:
        """
        Calculate Total Return Series for given spread data
        
        Returns DataFrame with columns:
        - date, spread_bps, series_number
        - long_tr, short_tr (indexed to 100)
        - long_pnl, short_pnl (cumulative P&L)
        - daily_return_long, daily_return_short
        """
        
        if spread_data.empty:
            return pd.DataFrame()
        
        df = spread_data.copy().sort_values('date').reset_index(drop=True)
        
        # Extract years from tenor
        tenor_years = float(tenor.replace('Y', ''))
        
        # Calculate approximate DV01
        df['dv01_approx'] = (1 - recovery_rate) * df['spread_bps'] * tenor_years * notional / 10000
        
        # Calculate daily changes
        df['spread_change'] = df['spread_bps'].diff()
        df['daily_carry'] = df['spread_bps'] / 360
        
        # Mark-to-market P&L
        df['mtm_pnl_long'] = -df['spread_change'] * df['dv01_approx'] / 100
        df['mtm_pnl_short'] = df['spread_change'] * df['dv01_approx'] / 100
        
        # Roll adjustments
        df['is_roll_date'] = df['series_number'] != df['series_number'].shift(1)
        df['roll_cost'] = 0.0
        
        roll_dates = df[df['is_roll_date'] & (df.index > 0)]
        for idx in roll_dates.index:
            if idx > 0:
                spread_jump = df.loc[idx, 'spread_bps'] - df.loc[idx-1, 'spread_bps']
                transaction_cost = 0.5  # bps
                df.loc[idx, 'roll_cost'] = (transaction_cost + abs(spread_jump) * 0.1) * df.loc[idx, 'dv01_approx'] / 100
        
        # Total daily P&L
        df['long_daily_pnl'] = (df['daily_carry'] * notional / 10000) + df['mtm_pnl_long'] - df['roll_cost']
        df['short_daily_pnl'] = -(df['daily_carry'] * notional / 10000) + df['mtm_pnl_short'] - df['roll_cost']
        
        # Daily returns
        df['daily_return_long'] = df['long_daily_pnl'] / notional
        df['daily_return_short'] = df['short_daily_pnl'] / notional
        
        # Cumulative total return indices (starting at 100)
        df['long_tr'] = 100.0
        df['short_tr'] = 100.0
        
        for i in range(1, len(df)):
            df.loc[i, 'long_tr'] = df.loc[i-1, 'long_tr'] * (1 + df.loc[i, 'daily_return_long'])
            df.loc[i, 'short_tr'] = df.loc[i-1, 'short_tr'] * (1 + df.loc[i, 'daily_return_short'])
        
        # Cumulative P&L
        df['long_pnl'] = df['long_daily_pnl'].cumsum()
        df['short_pnl'] = df['short_daily_pnl'].cumsum()
        
        # Keep essential columns
        result = df[['date', 'spread_bps', 'series_number', 
                    'long_tr', 'short_tr', 'long_pnl', 'short_pnl',
                    'daily_return_long', 'daily_return_short']].copy()
        
        return result
    
    def build_trs_database(self, start_date: str = '2020-01-01', verbose: bool = False):
        """
        Build complete TRS database for all indices and tenors
        """
        
        # Connect to raw database
        raw_db = CDSDatabase(self.raw_db_path)
        
        # Create TRS database
        trs_conn = sqlite3.connect(self.trs_db_path)
        
        # Track processing
        processed = []
        failed = []
        
        for index_name, index_info in self.indices.items():
            for tenor in self.tenors:
                key = f"{index_name}_{tenor}"
                
                try:
                    # Get raw spread data
                    spread_data = raw_db.query_historical_spreads(
                        index_name, tenor, start_date
                    )
                    
                    if spread_data.empty:
                        failed.append(key)
                        continue
                    
                    # Calculate TRS
                    trs_data = self.calculate_trs(
                        spread_data, 
                        tenor,
                        recovery_rate=index_info['recovery']
                    )
                    
                    if trs_data.empty:
                        failed.append(key)
                        continue
                    
                    # Save to database
                    table_name = f"trs_{index_name.lower()}_{tenor.lower()}"
                    trs_data.to_sql(table_name, trs_conn, 
                                   if_exists='replace', index=False)
                    
                    processed.append(key)
                    
                    if verbose:
                        print(f"Processed {key}: {len(trs_data)} records")
                    
                except Exception as e:
                    failed.append(key)
                    if verbose:
                        print(f"Failed {key}: {e}")
        
        # Create metadata table
        metadata = pd.DataFrame({
            'index_name': [k.split('_')[0] + '_' + k.split('_')[1] for k in processed],
            'tenor': [k.split('_')[-1] for k in processed],
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'record_count': [0] * len(processed)
        })
        
        metadata.to_sql('metadata', trs_conn, if_exists='replace', index=False)
        
        # Create indices for faster queries
        cursor = trs_conn.cursor()
        for index_name in self.indices.keys():
            for tenor in self.tenors:
                table_name = f"trs_{index_name.lower()}_{tenor.lower()}"
                try:
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(date)")
                except:
                    pass
        
        trs_conn.commit()
        trs_conn.close()
        raw_db.close()
        
        return processed, failed
    
    def get_trs_data(self, index_name: str, tenor: str, 
                     start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve TRS data from database
        """
        conn = sqlite3.connect(self.trs_db_path)
        
        table_name = f"trs_{index_name.lower()}_{tenor.lower()}"
        
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
        except:
            df = pd.DataFrame()
        
        conn.close()
        return df
    
    def test_database(self):
        """
        Test the TRS database to ensure all expected series are present
        """
        conn = sqlite3.connect(self.trs_db_path)
        cursor = conn.cursor()
        
        # Get all tables
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'trs_%'").fetchall()
        table_list = [t[0] for t in tables]
        
        print(f"Total TRS tables: {len(table_list)}")
        
        # Check expected vs actual
        expected = []
        for index in self.indices.keys():
            for tenor in self.tenors:
                expected.append(f"trs_{index.lower()}_{tenor.lower()}")
        
        missing = [t for t in expected if t not in table_list]
        extra = [t for t in table_list if t not in expected]
        
        if missing:
            print(f"\nMissing tables: {len(missing)}")
            for m in missing:
                print(f"  - {m}")
        
        if extra:
            print(f"\nExtra tables: {len(extra)}")
            for e in extra:
                print(f"  + {e}")
        
        # Test data retrieval for each table
        test_results = {}
        for table in table_list:
            try:
                count_query = f"SELECT COUNT(*) as cnt, MIN(date) as min_date, MAX(date) as max_date FROM {table}"
                result = cursor.execute(count_query).fetchone()
                test_results[table] = {
                    'count': result[0],
                    'min_date': result[1],
                    'max_date': result[2]
                }
            except Exception as e:
                test_results[table] = {'error': str(e)}
        
        # Summary statistics
        total_records = sum(r['count'] for r in test_results.values() if 'count' in r)
        print(f"\nTotal records across all tables: {total_records:,}")
        
        # Show sample of data coverage
        print("\nData coverage sample:")
        for table, result in list(test_results.items())[:5]:
            if 'count' in result:
                print(f"  {table}: {result['count']} records ({result['min_date']} to {result['max_date']})")
        
        conn.close()
        
        return {
            'tables': len(table_list),
            'missing': missing,
            'test_results': test_results,
            'total_records': total_records
        }

# Main execution
if __name__ == "__main__":
    # Define paths
    raw_db_path = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\raw\cds_indices_raw.db"
    trs_db_path = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\processed\cds_trs.db"
    
    # Build TRS database
    print("Building TRS database...")
    builder = TRSDatabaseBuilder(raw_db_path, trs_db_path)
    processed, failed = builder.build_trs_database(start_date='2023-01-01', verbose=True)
    
    print(f"\nProcessed: {len(processed)} series")
    print(f"Failed: {len(failed)} series")
    
    # Run tests
    print("\nRunning database tests...")
    test_results = builder.test_database()
    
    # Test data retrieval
    print("\nTesting data retrieval...")
    test_data = builder.get_trs_data('EU_IG', '5Y', '2024-01-01', '2024-12-31')
    if not test_data.empty:
        print(f"Successfully retrieved {len(test_data)} records for EU_IG 5Y")
        print(f"Long TR range: {test_data['long_tr'].min():.2f} - {test_data['long_tr'].max():.2f}")
        print(f"Short TR range: {test_data['short_tr'].min():.2f} - {test_data['short_tr'].max():.2f}")


# TRS Database Builder
# Creates Total Return Series database for all CDS indices and tenors

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from src.models.database import CDSDatabase

class TRSDatabaseBuilder:
    """Build and maintain TRS database for all CDS indices"""
    
    def __init__(self, raw_db_path: str, trs_db_path: str):
        """
        Initialize TRS database builder
        
        Args:
            raw_db_path: Path to raw CDS spreads database
            trs_db_path: Path to output TRS database
        """
        self.raw_db_path = raw_db_path
        self.trs_db_path = trs_db_path
        
        # Ensure output directory exists
        Path(self.trs_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Define all indices and tenors to process
        self.indices = {
            'EU_IG': {'base': 'ITRX EUR CDSI', 'recovery': 0.40},
            'EU_XO': {'base': 'ITRX EUR XOVER', 'recovery': 0.40},
            'US_IG': {'base': 'CDX IG', 'recovery': 0.40},
            'US_HY': {'base': 'CDX HY', 'recovery': 0.40},
            'EU_SUBFIN': {'base': 'ITRX EUR SUBFIN', 'recovery': 0.40},
            'EU_SENFIN': {'base': 'ITRX EUR SENFIN', 'recovery': 0.40}
        }
        
        self.tenors = ['3Y', '5Y', '7Y', '10Y']
        
    def calculate_trs(self, spread_data: pd.DataFrame, 
                     tenor: str,
                     recovery_rate: float = 0.40,
                     notional: float = 10_000_000) -> pd.DataFrame:
        """
        Calculate Total Return Series for given spread data
        
        Returns DataFrame with columns:
        - date, spread_bps, series_number
        - long_tr, short_tr (indexed to 100)
        - long_pnl, short_pnl (cumulative P&L)
        - daily_return_long, daily_return_short
        """
        
        if spread_data.empty:
            return pd.DataFrame()
        
        df = spread_data.copy().sort_values('date').reset_index(drop=True)
        
        # Extract years from tenor
        tenor_years = float(tenor.replace('Y', ''))
        
        # Calculate approximate DV01
        df['dv01_approx'] = (1 - recovery_rate) * df['spread_bps'] * tenor_years * notional / 10000
        
        # Calculate daily changes
        df['spread_change'] = df['spread_bps'].diff()
        df['daily_carry'] = df['spread_bps'] / 360
        
        # Mark-to-market P&L
        df['mtm_pnl_long'] = -df['spread_change'] * df['dv01_approx'] / 100
        df['mtm_pnl_short'] = df['spread_change'] * df['dv01_approx'] / 100
        
        # Roll adjustments
        df['is_roll_date'] = df['series_number'] != df['series_number'].shift(1)
        df['roll_cost'] = 0.0
        
        roll_dates = df[df['is_roll_date'] & (df.index > 0)]
        for idx in roll_dates.index:
            if idx > 0:
                spread_jump = df.loc[idx, 'spread_bps'] - df.loc[idx-1, 'spread_bps']
                transaction_cost = 0.5  # bps
                df.loc[idx, 'roll_cost'] = (transaction_cost + abs(spread_jump) * 0.1) * df.loc[idx, 'dv01_approx'] / 100
        
        # Total daily P&L
        df['long_daily_pnl'] = (df['daily_carry'] * notional / 10000) + df['mtm_pnl_long'] - df['roll_cost']
        df['short_daily_pnl'] = -(df['daily_carry'] * notional / 10000) + df['mtm_pnl_short'] - df['roll_cost']
        
        # Daily returns
        df['daily_return_long'] = df['long_daily_pnl'] / notional
        df['daily_return_short'] = df['short_daily_pnl'] / notional
        
        # Cumulative total return indices (starting at 100)
        df['long_tr'] = 100.0
        df['short_tr'] = 100.0
        
        for i in range(1, len(df)):
            df.loc[i, 'long_tr'] = df.loc[i-1, 'long_tr'] * (1 + df.loc[i, 'daily_return_long'])
            df.loc[i, 'short_tr'] = df.loc[i-1, 'short_tr'] * (1 + df.loc[i, 'daily_return_short'])
        
        # Cumulative P&L
        df['long_pnl'] = df['long_daily_pnl'].cumsum()
        df['short_pnl'] = df['short_daily_pnl'].cumsum()
        
        # Keep essential columns
        result = df[['date', 'spread_bps', 'series_number', 
                    'long_tr', 'short_tr', 'long_pnl', 'short_pnl',
                    'daily_return_long', 'daily_return_short']].copy()
        
        return result
    
    def build_trs_database(self, start_date: str = '2020-01-01', verbose: bool = False):
        """
        Build complete TRS database for all indices and tenors
        """
        
        # Connect to raw database
        raw_db = CDSDatabase(self.raw_db_path)
        
        # Create TRS database
        trs_conn = sqlite3.connect(self.trs_db_path)
        
        # Track processing
        processed = []
        failed = []
        
        for index_name, index_info in self.indices.items():
            for tenor in self.tenors:
                key = f"{index_name}_{tenor}"
                
                try:
                    # Get raw spread data
                    spread_data = raw_db.query_historical_spreads(
                        index_name, tenor, start_date
                    )
                    
                    if spread_data.empty:
                        failed.append(key)
                        continue
                    
                    # Calculate TRS
                    trs_data = self.calculate_trs(
                        spread_data, 
                        tenor,
                        recovery_rate=index_info['recovery']
                    )
                    
                    if trs_data.empty:
                        failed.append(key)
                        continue
                    
                    # Save to database
                    table_name = f"trs_{index_name.lower()}_{tenor.lower()}"
                    trs_data.to_sql(table_name, trs_conn, 
                                   if_exists='replace', index=False)
                    
                    processed.append(key)
                    
                    if verbose:
                        print(f"Processed {key}: {len(trs_data)} records")
                    
                except Exception as e:
                    failed.append(key)
                    if verbose:
                        print(f"Failed {key}: {e}")
        
        # Create metadata table
        metadata = pd.DataFrame({
            'index_name': [k.split('_')[0] + '_' + k.split('_')[1] for k in processed],
            'tenor': [k.split('_')[-1] for k in processed],
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'record_count': [0] * len(processed)
        })
        
        metadata.to_sql('metadata', trs_conn, if_exists='replace', index=False)
        
        # Create indices for faster queries
        cursor = trs_conn.cursor()
        for index_name in self.indices.keys():
            for tenor in self.tenors:
                table_name = f"trs_{index_name.lower()}_{tenor.lower()}"
                try:
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(date)")
                except:
                    pass
        
        trs_conn.commit()
        trs_conn.close()
        raw_db.close()
        
        return processed, failed
    
    def get_trs_data(self, index_name: str, tenor: str, 
                     start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve TRS data from database
        """
        conn = sqlite3.connect(self.trs_db_path)
        
        table_name = f"trs_{index_name.lower()}_{tenor.lower()}"
        
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
        except:
            df = pd.DataFrame()
        
        conn.close()
        return df
    
    def test_database(self):
        """
        Test the TRS database to ensure all expected series are present
        """
        conn = sqlite3.connect(self.trs_db_path)
        cursor = conn.cursor()
        
        # Get all tables
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'trs_%'").fetchall()
        table_list = [t[0] for t in tables]
        
        print(f"Total TRS tables: {len(table_list)}")
        
        # Check expected vs actual
        expected = []
        for index in self.indices.keys():
            for tenor in self.tenors:
                expected.append(f"trs_{index.lower()}_{tenor.lower()}")
        
        missing = [t for t in expected if t not in table_list]
        extra = [t for t in table_list if t not in expected]
        
        if missing:
            print(f"\nMissing tables: {len(missing)}")
            for m in missing:
                print(f"  - {m}")
        
        if extra:
            print(f"\nExtra tables: {len(extra)}")
            for e in extra:
                print(f"  + {e}")
        
        # Test data retrieval for each table
        test_results = {}
        for table in table_list:
            try:
                count_query = f"SELECT COUNT(*) as cnt, MIN(date) as min_date, MAX(date) as max_date FROM {table}"
                result = cursor.execute(count_query).fetchone()
                test_results[table] = {
                    'count': result[0],
                    'min_date': result[1],
                    'max_date': result[2]
                }
            except Exception as e:
                test_results[table] = {'error': str(e)}
        
        # Summary statistics
        total_records = sum(r['count'] for r in test_results.values() if 'count' in r)
        print(f"\nTotal records across all tables: {total_records:,}")
        
        # Show sample of data coverage
        print("\nData coverage sample:")
        for table, result in list(test_results.items())[:5]:
            if 'count' in result:
                print(f"  {table}: {result['count']} records ({result['min_date']} to {result['max_date']})")
        
        conn.close()
        
        return {
            'tables': len(table_list),
            'missing': missing,
            'test_results': test_results,
            'total_records': total_records
        }

# Main execution
if __name__ == "__main__":
    # Define paths
    raw_db_path = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\raw\cds_indices_raw.db"
    trs_db_path = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\processed\cds_trs.db"
    
    # Build TRS database
    print("Building TRS database...")
    builder = TRSDatabaseBuilder(raw_db_path, trs_db_path)
    processed, failed = builder.build_trs_database(start_date='2023-01-01', verbose=True)
    
    print(f"\nProcessed: {len(processed)} series")
    print(f"Failed: {len(failed)} series")
    
    # Run tests
    print("\nRunning database tests...")
    test_results = builder.test_database()
    
    # Test data retrieval
    print("\nTesting data retrieval...")
    test_data = builder.get_trs_data('EU_IG', '5Y', '2024-01-01', '2024-12-31')
    if not test_data.empty:
        print(f"Successfully retrieved {len(test_data)} records for EU_IG 5Y")
        print(f"Long TR range: {test_data['long_tr'].min():.2f} - {test_data['long_tr'].max():.2f}")
        print(f"Short TR range: {test_data['short_tr'].min():.2f} - {test_data['short_tr'].max():.2f}")


# TRS Database Builder
# Creates Total Return Series database for all CDS indices and tenors

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from src.models.database import CDSDatabase

class TRSDatabaseBuilder:
    """Build and maintain TRS database for all CDS indices"""
    
    def __init__(self, raw_db_path: str, trs_db_path: str):
        """
        Initialize TRS database builder
        
        Args:
            raw_db_path: Path to raw CDS spreads database
            trs_db_path: Path to output TRS database
        """
        self.raw_db_path = raw_db_path
        self.trs_db_path = trs_db_path
        
        # Ensure output directory exists
        Path(self.trs_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Define all indices and tenors to process
        self.indices = {
            'EU_IG': {'base': 'ITRX EUR CDSI', 'recovery': 0.40},
            'EU_XO': {'base': 'ITRX EUR XOVER', 'recovery': 0.40},
            'US_IG': {'base': 'CDX IG', 'recovery': 0.40},
            'US_HY': {'base': 'CDX HY', 'recovery': 0.40},
            'EU_SUBFIN': {'base': 'ITRX EUR SUBFIN', 'recovery': 0.40},
            'EU_SENFIN': {'base': 'ITRX EUR SENFIN', 'recovery': 0.40}
        }
        
        self.tenors = ['3Y', '5Y', '7Y', '10Y']
        
    def calculate_trs(self, spread_data: pd.DataFrame, 
                     tenor: str,
                     recovery_rate: float = 0.40,
                     notional: float = 10_000_000) -> pd.DataFrame:
        """
        Calculate Total Return Series for given spread data
        
        Returns DataFrame with columns:
        - date, spread_bps, series_number
        - long_tr, short_tr (indexed to 100)
        - long_pnl, short_pnl (cumulative P&L)
        - daily_return_long, daily_return_short
        """
        
        if spread_data.empty:
            return pd.DataFrame()
        
        df = spread_data.copy().sort_values('date').reset_index(drop=True)
        
        # Extract years from tenor
        tenor_years = float(tenor.replace('Y', ''))
        
        # Calculate approximate DV01
        df['dv01_approx'] = (1 - recovery_rate) * df['spread_bps'] * tenor_years * notional / 10000
        
        # Calculate daily changes
        df['spread_change'] = df['spread_bps'].diff()
        df['daily_carry'] = df['spread_bps'] / 360
        
        # Mark-to-market P&L
        df['mtm_pnl_long'] = -df['spread_change'] * df['dv01_approx'] / 100
        df['mtm_pnl_short'] = df['spread_change'] * df['dv01_approx'] / 100
        
        # Roll adjustments
        df['is_roll_date'] = df['series_number'] != df['series_number'].shift(1)
        df['roll_cost'] = 0.0
        
        roll_dates = df[df['is_roll_date'] & (df.index > 0)]
        for idx in roll_dates.index:
            if idx > 0:
                spread_jump = df.loc[idx, 'spread_bps'] - df.loc[idx-1, 'spread_bps']
                transaction_cost = 0.5  # bps
                df.loc[idx, 'roll_cost'] = (transaction_cost + abs(spread_jump) * 0.1) * df.loc[idx, 'dv01_approx'] / 100
        
        # Total daily P&L
        df['long_daily_pnl'] = (df['daily_carry'] * notional / 10000) + df['mtm_pnl_long'] - df['roll_cost']
        df['short_daily_pnl'] = -(df['daily_carry'] * notional / 10000) + df['mtm_pnl_short'] - df['roll_cost']
        
        # Daily returns
        df['daily_return_long'] = df['long_daily_pnl'] / notional
        df['daily_return_short'] = df['short_daily_pnl'] / notional
        
        # Cumulative total return indices (starting at 100)
        df['long_tr'] = 100.0
        df['short_tr'] = 100.0
        
        for i in range(1, len(df)):
            df.loc[i, 'long_tr'] = df.loc[i-1, 'long_tr'] * (1 + df.loc[i, 'daily_return_long'])
            df.loc[i, 'short_tr'] = df.loc[i-1, 'short_tr'] * (1 + df.loc[i, 'daily_return_short'])
        
        # Cumulative P&L
        df['long_pnl'] = df['long_daily_pnl'].cumsum()
        df['short_pnl'] = df['short_daily_pnl'].cumsum()
        
        # Keep essential columns
        result = df[['date', 'spread_bps', 'series_number', 
                    'long_tr', 'short_tr', 'long_pnl', 'short_pnl',
                    'daily_return_long', 'daily_return_short']].copy()
        
        return result
    
    def build_trs_database(self, start_date: str = '2020-01-01', verbose: bool = False):
        """
        Build complete TRS database for all indices and tenors
        """
        
        # Connect to raw database
        raw_db = CDSDatabase(self.raw_db_path)
        
        # Create TRS database
        trs_conn = sqlite3.connect(self.trs_db_path)
        
        # Track processing
        processed = []
        failed = []
        
        for index_name, index_info in self.indices.items():
            for tenor in self.tenors:
                key = f"{index_name}_{tenor}"
                
                try:
                    # Get raw spread data
                    spread_data = raw_db.query_historical_spreads(
                        index_name, tenor, start_date
                    )
                    
                    if spread_data.empty:
                        failed.append(key)
                        continue
                    
                    # Calculate TRS
                    trs_data = self.calculate_trs(
                        spread_data, 
                        tenor,
                        recovery_rate=index_info['recovery']
                    )
                    
                    if trs_data.empty:
                        failed.append(key)
                        continue
                    
                    # Save to database
                    table_name = f"trs_{index_name.lower()}_{tenor.lower()}"
                    trs_data.to_sql(table_name, trs_conn, 
                                   if_exists='replace', index=False)
                    
                    processed.append(key)
                    
                    if verbose:
                        print(f"Processed {key}: {len(trs_data)} records")
                    
                except Exception as e:
                    failed.append(key)
                    if verbose:
                        print(f"Failed {key}: {e}")
        
        # Create metadata table
        metadata = pd.DataFrame({
            'index_name': [k.split('_')[0] + '_' + k.split('_')[1] for k in processed],
            'tenor': [k.split('_')[-1] for k in processed],
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'record_count': [0] * len(processed)
        })
        
        metadata.to_sql('metadata', trs_conn, if_exists='replace', index=False)
        
        # Create indices for faster queries
        cursor = trs_conn.cursor()
        for index_name in self.indices.keys():
            for tenor in self.tenors:
                table_name = f"trs_{index_name.lower()}_{tenor.lower()}"
                try:
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(date)")
                except:
                    pass
        
        trs_conn.commit()
        trs_conn.close()
        raw_db.close()
        
        return processed, failed
    
    def get_trs_data(self, index_name: str, tenor: str, 
                     start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve TRS data from database
        """
        conn = sqlite3.connect(self.trs_db_path)
        
        table_name = f"trs_{index_name.lower()}_{tenor.lower()}"
        
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
        except:
            df = pd.DataFrame()
        
        conn.close()
        return df
    
    def test_database(self):
        """
        Test the TRS database to ensure all expected series are present
        """
        conn = sqlite3.connect(self.trs_db_path)
        cursor = conn.cursor()
        
        # Get all tables
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'trs_%'").fetchall()
        table_list = [t[0] for t in tables]
        
        print(f"Total TRS tables: {len(table_list)}")
        
        # Check expected vs actual
        expected = []
        for index in self.indices.keys():
            for tenor in self.tenors:
                expected.append(f"trs_{index.lower()}_{tenor.lower()}")
        
        missing = [t for t in expected if t not in table_list]
        extra = [t for t in table_list if t not in expected]
        
        if missing:
            print(f"\nMissing tables: {len(missing)}")
            for m in missing:
                print(f"  - {m}")
        
        if extra:
            print(f"\nExtra tables: {len(extra)}")
            for e in extra:
                print(f"  + {e}")
        
        # Test data retrieval for each table
        test_results = {}
        for table in table_list:
            try:
                count_query = f"SELECT COUNT(*) as cnt, MIN(date) as min_date, MAX(date) as max_date FROM {table}"
                result = cursor.execute(count_query).fetchone()
                test_results[table] = {
                    'count': result[0],
                    'min_date': result[1],
                    'max_date': result[2]
                }
            except Exception as e:
                test_results[table] = {'error': str(e)}
        
        # Summary statistics
        total_records = sum(r['count'] for r in test_results.values() if 'count' in r)
        print(f"\nTotal records across all tables: {total_records:,}")
        
        # Show sample of data coverage
        print("\nData coverage sample:")
        for table, result in list(test_results.items())[:5]:
            if 'count' in result:
                print(f"  {table}: {result['count']} records ({result['min_date']} to {result['max_date']})")
        
        conn.close()
        
        return {
            'tables': len(table_list),
            'missing': missing,
            'test_results': test_results,
            'total_records': total_records
        }

# Main execution
if __name__ == "__main__":
    # Define paths
    raw_db_path = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\raw\cds_indices_raw.db"
    trs_db_path = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\processed\cds_trs.db"
    
    # Build TRS database
    print("Building TRS database...")
    builder = TRSDatabaseBuilder(raw_db_path, trs_db_path)
    processed, failed = builder.build_trs_database(start_date='2023-01-01', verbose=True)
    
    print(f"\nProcessed: {len(processed)} series")
    print(f"Failed: {len(failed)} series")
    
    # Run tests
    print("\nRunning database tests...")
    test_results = builder.test_database()
    
    # Test data retrieval
    print("\nTesting data retrieval...")
    test_data = builder.get_trs_data('EU_IG', '5Y', '2024-01-01', '2024-12-31')
    if not test_data.empty:
        print(f"Successfully retrieved {len(test_data)} records for EU_IG 5Y")
        print(f"Long TR range: {test_data['long_tr'].min():.2f} - {test_data['long_tr'].max():.2f}")
        print(f"Short TR range: {test_data['short_tr'].min():.2f} - {test_data['short_tr'].max():.2f}")


# TRS Database Builder
# Creates Total Return Series database for all CDS indices and tenors

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from src.models.database import CDSDatabase

class TRSDatabaseBuilder:
    """Build and maintain TRS database for all CDS indices"""
    
    def __init__(self, raw_db_path: str, trs_db_path: str):
        """
        Initialize TRS database builder
        
        Args:
            raw_db_path: Path to raw CDS spreads database
            trs_db_path: Path to output TRS database
        """
        self.raw_db_path = raw_db_path
        self.trs_db_path = trs_db_path
        
        # Ensure output directory exists
        Path(self.trs_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Define all indices and tenors to process
        self.indices = {
            'EU_IG': {'base': 'ITRX EUR CDSI', 'recovery': 0.40},
            'EU_XO': {'base': 'ITRX EUR XOVER', 'recovery': 0.40},
            'US_IG': {'base': 'CDX IG', 'recovery': 0.40},
            'US_HY': {'base': 'CDX HY', 'recovery': 0.40},
            'EU_SUBFIN': {'base': 'ITRX EUR SUBFIN', 'recovery': 0.40},
            'EU_SENFIN': {'base': 'ITRX EUR SENFIN', 'recovery': 0.40}
        }
        
        self.tenors = ['3Y', '5Y', '7Y', '10Y']
        
    def calculate_trs(self, spread_data: pd.DataFrame, 
                     tenor: str,
                     recovery_rate: float = 0.40,
                     notional: float = 10_000_000) -> pd.DataFrame:
        """
        Calculate Total Return Series for given spread data
        
        Returns DataFrame with columns:
        - date, spread_bps, series_number
        - long_tr, short_tr (indexed to 100)
        - long_pnl, short_pnl (cumulative P&L)
        - daily_return_long, daily_return_short
        """
        
        if spread_data.empty:
            return pd.DataFrame()
        
        df = spread_data.copy().sort_values('date').reset_index(drop=True)
        
        # Extract years from tenor
        tenor_years = float(tenor.replace('Y', ''))
        
        # Calculate approximate DV01
        df['dv01_approx'] = (1 - recovery_rate) * df['spread_bps'] * tenor_years * notional / 10000
        
        # Calculate daily changes
        df['spread_change'] = df['spread_bps'].diff()
        df['daily_carry'] = df['spread_bps'] / 360
        
        # Mark-to-market P&L
        df['mtm_pnl_long'] = -df['spread_change'] * df['dv01_approx'] / 100
        df['mtm_pnl_short'] = df['spread_change'] * df['dv01_approx'] / 100
        
        # Roll adjustments
        df['is_roll_date'] = df['series_number'] != df['series_number'].shift(1)
        df['roll_cost'] = 0.0
        
        roll_dates = df[df['is_roll_date'] & (df.index > 0)]
        for idx in roll_dates.index:
            if idx > 0:
                spread_jump = df.loc[idx, 'spread_bps'] - df.loc[idx-1, 'spread_bps']
                transaction_cost = 0.5  # bps
                df.loc[idx, 'roll_cost'] = (transaction_cost + abs(spread_jump) * 0.1) * df.loc[idx, 'dv01_approx'] / 100
        
        # Total daily P&L
        df['long_daily_pnl'] = (df['daily_carry'] * notional / 10000) + df['mtm_pnl_long'] - df['roll_cost']
        df['short_daily_pnl'] = -(df['daily_carry'] * notional / 10000) + df['mtm_pnl_short'] - df['roll_cost']
        
        # Daily returns
        df['daily_return_long'] = df['long_daily_pnl'] / notional
        df['daily_return_short'] = df['short_daily_pnl'] / notional
        
        # Cumulative total return indices (starting at 100)
        df['long_tr'] = 100.0
        df['short_tr'] = 100.0
        
        for i in range(1, len(df)):
            df.loc[i, 'long_tr'] = df.loc[i-1, 'long_tr'] * (1 + df.loc[i, 'daily_return_long'])
            df.loc[i, 'short_tr'] = df.loc[i-1, 'short_tr'] * (1 + df.loc[i, 'daily_return_short'])
        
        # Cumulative P&L
        df['long_pnl'] = df['long_daily_pnl'].cumsum()
        df['short_pnl'] = df['short_daily_pnl'].cumsum()
        
        # Keep essential columns
        result = df[['date', 'spread_bps', 'series_number', 
                    'long_tr', 'short_tr', 'long_pnl', 'short_pnl',
                    'daily_return_long', 'daily_return_short']].copy()
        
        return result
    
    def build_trs_database(self, start_date: str = '2020-01-01', verbose: bool = False):
        """
        Build complete TRS database for all indices and tenors
        """
        
        # Connect to raw database
        raw_db = CDSDatabase(self.raw_db_path)
        
        # Create TRS database
        trs_conn = sqlite3.connect(self.trs_db_path)
        
        # Track processing
        processed = []
        failed = []
        
        for index_name, index_info in self.indices.items():
            for tenor in self.tenors:
                key = f"{index_name}_{tenor}"
                
                try:
                    # Get raw spread data
                    spread_data = raw_db.query_historical_spreads(
                        index_name, tenor, start_date
                    )
                    
                    if spread_data.empty:
                        failed.append(key)
                        continue
                    
                    # Calculate TRS
                    trs_data = self.calculate_trs(
                        spread_data, 
                        tenor,
                        recovery_rate=index_info['recovery']
                    )
                    
                    if trs_data.empty:
                        failed.append(key)
                        continue
                    
                    # Save to database
                    table_name = f"trs_{index_name.lower()}_{tenor.lower()}"
                    trs_data.to_sql(table_name, trs_conn, 
                                   if_exists='replace', index=False)
                    
                    processed.append(key)
                    
                    if verbose:
                        print(f"Processed {key}: {len(trs_data)} records")
                    
                except Exception as e:
                    failed.append(key)
                    if verbose:
                        print(f"Failed {key}: {e}")
        
        # Create metadata table
        metadata = pd.DataFrame({
            'index_name': [k.split('_')[0] + '_' + k.split('_')[1] for k in processed],
            'tenor': [k.split('_')[-1] for k in processed],
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'record_count': [0] * len(processed)
        })
        
        metadata.to_sql('metadata', trs_conn, if_exists='replace', index=False)
        
        # Create indices for faster queries
        cursor = trs_conn.cursor()
        for index_name in self.indices.keys():
            for tenor in self.tenors:
                table_name = f"trs_{index_name.lower()}_{tenor.lower()}"
                try:
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(date)")
                except:
                    pass
        
        trs_conn.commit()
        trs_conn.close()
        raw_db.close()
        
        return processed, failed
    
    def get_trs_data(self, index_name: str, tenor: str, 
                     start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve TRS data from database
        """
        conn = sqlite3.connect(self.trs_db_path)
        
        table_name = f"trs_{index_name.lower()}_{tenor.lower()}"
        
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
        except:
            df = pd.DataFrame()
        
        conn.close()
        return df
    
    def test_database(self):
        """
        Test the TRS database to ensure all expected series are present
        """
        conn = sqlite3.connect(self.trs_db_path)
        cursor = conn.cursor()
        
        # Get all tables
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'trs_%'").fetchall()
        table_list = [t[0] for t in tables]
        
        print(f"Total TRS tables: {len(table_list)}")
        
        # Check expected vs actual
        expected = []
        for index in self.indices.keys():
            for tenor in self.tenors:
                expected.append(f"trs_{index.lower()}_{tenor.lower()}")
        
        missing = [t for t in expected if t not in table_list]
        extra = [t for t in table_list if t not in expected]
        
        if missing:
            print(f"\nMissing tables: {len(missing)}")
            for m in missing:
                print(f"  - {m}")
        
        if extra:
            print(f"\nExtra tables: {len(extra)}")
            for e in extra:
                print(f"  + {e}")
        
        # Test data retrieval for each table
        test_results = {}
        for table in table_list:
            try:
                count_query = f"SELECT COUNT(*) as cnt, MIN(date) as min_date, MAX(date) as max_date FROM {table}"
                result = cursor.execute(count_query).fetchone()
                test_results[table] = {
                    'count': result[0],
                    'min_date': result[1],
                    'max_date': result[2]
                }
            except Exception as e:
                test_results[table] = {'error': str(e)}
        
        # Summary statistics
        total_records = sum(r['count'] for r in test_results.values() if 'count' in r)
        print(f"\nTotal records across all tables: {total_records:,}")
        
        # Show sample of data coverage
        print("\nData coverage sample:")
        for table, result in list(test_results.items())[:5]:
            if 'count' in result:
                print(f"  {table}: {result['count']} records ({result['min_date']} to {result['max_date']})")
        
        conn.close()
        
        return {
            'tables': len(table_list),
            'missing': missing,
            'test_results': test_results,
            'total_records': total_records
        }

# Main execution
if __name__ == "__main__":
    # Define paths
    raw_db_path = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\raw\cds_indices_raw.db"
    trs_db_path = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\processed\cds_trs.db"
    
    # Build TRS database
    print("Building TRS database...")
    builder = TRSDatabaseBuilder(raw_db_path, trs_db_path)
    processed, failed = builder.build_trs_database(start_date='2023-01-01', verbose=True)
    
    print(f"\nProcessed: {len(processed)} series")
    print(f"Failed: {len(failed)} series")
    
    # Run tests
    print("\nRunning database tests...")
    test_results = builder.test_database()
    
    # Test data retrieval
    print("\nTesting data retrieval...")
    test_data = builder.get_trs_data('EU_IG', '5Y', '2024-01-01', '2024-12-31')
    if not test_data.empty:
        print(f"Successfully retrieved {len(test_data)} records for EU_IG 5Y")
        print(f"Long TR range: {test_data['long_tr'].min():.2f} - {test_data['long_tr'].max():.2f}")
        print(f"Short TR range: {test_data['short_tr'].min():.2f} - {test_data['short_tr'].max():.2f}")
