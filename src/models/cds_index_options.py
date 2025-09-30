# File: cds_index_option_pricing_system.py
"""
Complete Credit Index Option Pricing System
Integrates vol surface database management with option pricing
Includes special CDX HY handling with 500 bps fixed coupon
Ready for Streamlit deployment
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
import re
import logging
import warnings

# ============================================================================
# VOL SURFACE DATABASE MANAGER
# ============================================================================

class VolSurfaceDatabase:
    """
    Vol Surface Database Manager for Credit Index Options
    Handles Excel import and database management
    """
    
    def __init__(self, db_path=None):
        """Initialize database connection and logging"""
        if db_path is None:
            self.db_path = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\raw\vol_surfaces.db"
        else:
            self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for the database operations"""
        log_path = os.path.join(os.path.dirname(self.db_path), "vol_surface_updates.log")
        
        # Create a logger specific to this class
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def check_if_date_exists(self, data_date):
        """Check if data for specific date already exists"""
        if not os.path.exists(self.db_path):
            return False
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if table exists first
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='vol_surfaces'
        """)
        
        if not cursor.fetchone():
            conn.close()
            return False
        
        cursor.execute('''
            SELECT COUNT(*) FROM vol_surfaces 
            WHERE data_date = ?
        ''', (data_date,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def create_or_update_database(self, excel_path, data_date=None, force_update=False):
        """
        Create or update the vol surface database from Excel
        """
        if not os.path.exists(excel_path):
            self.logger.error(f"Excel file not found: {excel_path}")
            return 0
        
        if data_date is None:
            data_date = datetime.now().strftime("%Y-%m-%d")
        
        # Check if data already exists
        if not force_update and self.check_if_date_exists(data_date):
            self.logger.info(f"Data for {data_date} already exists. Use force_update=True to overwrite.")
            return 0
        
        self.logger.info(f"Processing vol surfaces for date: {data_date}")
        
        # Read Excel file
        xl_file = pd.ExcelFile(excel_path)
        sheet_names = xl_file.sheet_names
        self.logger.info(f"Found {len(sheet_names)} sheets")
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        self._create_tables(cursor)
        
        # Delete existing data for this date if force_update
        if force_update and self.check_if_date_exists(data_date):
            cursor.execute('DELETE FROM vol_surfaces WHERE data_date = ?', (data_date,))
            cursor.execute('DELETE FROM surface_metadata WHERE data_date = ?', (data_date,))
            self.logger.info(f"Deleted existing data for {data_date}")
        
        # Process each sheet
        total_options = 0
        for sheet_name in sheet_names:
            count = self._process_sheet(cursor, xl_file, sheet_name, data_date)
            total_options += count
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Total options stored: {total_options}")
        
        return total_options
    
    def _create_tables(self, cursor):
        """Create database tables with proper schema"""
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vol_surfaces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_date DATE NOT NULL,
                sheet_name TEXT NOT NULL,
                index_name TEXT NOT NULL,
                tenor TEXT NOT NULL,
                expiry DATE NOT NULL,
                spot_level REAL,
                forward_level REAL NOT NULL,
                atm_strike REAL,
                strike REAL NOT NULL,
                option_type TEXT NOT NULL,
                bid REAL,
                ask REAL,
                mid REAL,
                delta REAL,
                vol REAL NOT NULL,
                change REAL,
                breakeven REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(data_date, index_name, tenor, strike, option_type)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS surface_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_date DATE NOT NULL,
                sheet_name TEXT NOT NULL,
                index_name TEXT NOT NULL,
                tenor TEXT NOT NULL,
                expiry DATE NOT NULL,
                spot_level REAL,
                forward_level REAL NOT NULL,
                atm_strike REAL,
                update_time TEXT,
                UNIQUE(data_date, sheet_name)
            )
        ''')
        
        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_vol_date ON vol_surfaces(data_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_vol_index ON vol_surfaces(index_name, tenor)')
    
    def _extract_tenor(self, sheet_name):
        """Extract tenor from sheet name"""
        # Handle both '_' and '-' separators
        if '_' in sheet_name:
            parts = sheet_name.split('_')
        elif '-' in sheet_name:
            parts = sheet_name.split('-')
        else:
            return None
        
        if len(parts) > 1:
            tenor = parts[-1]
            if re.match(r'^\d+m$', tenor):
                return tenor
        
        return None
    
    def _process_sheet(self, cursor, xl_file, sheet_name, data_date):
        """Process a single sheet and store data"""
        
        # Extract tenor from sheet name
        tenor = self._extract_tenor(sheet_name)
        if not tenor:
            self.logger.warning(f"Skipping {sheet_name} - cannot extract tenor")
            return 0
        
        # Read sheet data
        df = xl_file.parse(sheet_name, header=None)
        
        # Parse the data
        metadata, options = self._parse_sheet_data(df, sheet_name, tenor)
        
        if not metadata or not options:
            self.logger.warning(f"No valid data found in {sheet_name}")
            return 0
        
        # Add data_date
        metadata['data_date'] = data_date
        
        # Store metadata
        cursor.execute('''
            INSERT OR REPLACE INTO surface_metadata 
            (data_date, sheet_name, index_name, tenor, expiry, 
             spot_level, forward_level, atm_strike, update_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data_date,
            sheet_name,
            metadata['index_name'],
            tenor,
            metadata['expiry'],
            metadata.get('spot_level'),
            metadata['forward_level'],
            metadata.get('atm_strike'),
            metadata.get('update_time')
        ))
        
        # Store options
        for option in options:
            cursor.execute('''
                INSERT OR REPLACE INTO vol_surfaces 
                (data_date, sheet_name, index_name, tenor, expiry,
                 spot_level, forward_level, atm_strike, strike,
                 option_type, bid, ask, mid, delta, vol, change, breakeven)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data_date,
                sheet_name,
                metadata['index_name'],
                tenor,
                metadata['expiry'],
                metadata.get('spot_level'),
                metadata['forward_level'],
                metadata.get('atm_strike'),
                option['strike'],
                option['option_type'],
                option.get('bid'),
                option.get('ask'),
                option.get('mid'),
                option.get('delta'),
                option['vol'],
                option.get('change'),
                option.get('breakeven')
            ))
        
        self.logger.info(f"  {sheet_name}: {len(options)} options stored")
        
        return len(options)
    
    def _parse_sheet_data(self, df, sheet_name, tenor):
        """Parse sheet data to extract metadata and options"""
        
        metadata = {'tenor': tenor}
        options = []
        
        # Convert to lines for parsing
        lines = []
        for _, row in df.iterrows():
            line = ' | '.join([str(cell) if pd.notna(cell) else '' for cell in row])
            if line.strip():
                lines.append(line)
        
        # Parse metadata from header lines
        for line in lines[:10]:  # Check more lines for metadata
            # Update time
            if 'Last updated:' in line:
                match = re.search(r'(\d{1,2}-\w{3}-\d{4})', line)
                if match:
                    metadata['update_time'] = match.group(1)
            
            # Index, expiry, forward, ATM
            if 'Options:' in line or 'Fwd' in line:
                # Determine index type
                if 'MAIN' in line.upper() or 'CDX.IG' in line.upper():
                    metadata['index_name'] = 'EU_IG' if 'MAIN' in line.upper() else 'US_IG'
                elif 'XOVER' in line.upper() or 'XO' in line.upper():
                    metadata['index_name'] = 'EU_XO'
                elif 'HY' in line.upper() or 'CDX.HY' in line.upper():
                    metadata['index_name'] = 'US_HY'
                elif 'IG' in line.upper():
                    metadata['index_name'] = 'US_IG'
                
                # Extract expiry date
                match = re.search(r'(\d{1,2}-\w{3}-\d{2})', line)
                if match:
                    metadata['expiry'] = match.group(1)
                
                # Extract forward level
                match = re.search(r'Fwd\s*@\s*([\d.]+)', line)
                if match:
                    metadata['forward_level'] = float(match.group(1))
                
                # Extract ATM strike
                match = re.search(r'Delta\s*@\s*([\d.]+)', line)
                if match:
                    metadata['atm_strike'] = float(match.group(1))
        
        # If we didn't find index_name, try to infer from sheet name
        if 'index_name' not in metadata:
            sheet_upper = sheet_name.upper()
            if 'MAIN' in sheet_upper or 'EU_IG' in sheet_upper:
                metadata['index_name'] = 'EU_IG'
            elif 'XO' in sheet_upper or 'EU_XO' in sheet_upper:
                metadata['index_name'] = 'EU_XO'
            elif 'US_HY' in sheet_upper or 'CDX_HY' in sheet_upper:
                metadata['index_name'] = 'US_HY'
            elif 'US_IG' in sheet_upper or 'CDX_IG' in sheet_upper:
                metadata['index_name'] = 'US_IG'
        
        # Validate metadata
        if 'index_name' not in metadata or 'forward_level' not in metadata:
            return None, None
        
        # Parse option data lines
        for line in lines:
            if '|' not in line:
                continue
            
            # Skip header lines
            if 'Delta' in line and 'Vol' in line:
                continue
            
            parts = line.split('|')
            
            # Process each option (Receiver and Payer pairs)
            for i in range(0, len(parts), 2):
                if i+1 >= len(parts):
                    break
                
                strike_part = parts[i].strip()
                data_part = parts[i+1].strip() if i+1 < len(parts) else ''
                
                # Extract strike
                strike_match = re.search(r'([\d.]+)', strike_part)
                if not strike_match or not data_part:
                    continue
                
                try:
                    strike = float(strike_match.group(1))
                    option_type = 'Receiver' if i == 0 else 'Payer'
                    
                    # Parse option details
                    option = self._parse_option_details(data_part, strike, option_type)
                    if option and option['vol']:
                        options.append(option)
                except:
                    continue
        
        return metadata, options
    
    def _parse_option_details(self, data_str, strike, option_type):
        """Parse individual option data"""
        
        tokens = data_str.split()
        if len(tokens) < 2:
            return None
        
        option = {
            'strike': strike,
            'option_type': option_type,
            'vol': None
        }
        
        # Bid/Ask
        if '/' in tokens[0]:
            try:
                parts = tokens[0].split('/')
                option['bid'] = float(parts[0])
                option['ask'] = float(parts[1])
                option['mid'] = (option['bid'] + option['ask']) / 2
            except:
                pass
        
        # Process remaining tokens
        for i, token in enumerate(tokens[1:], 1):
            if '%' in token:
                try:
                    option['delta'] = float(token.rstrip('%'))
                except:
                    pass
            elif i == 2 and re.match(r'^[\d.]+$', token):
                option['vol'] = float(token)
            elif i == 3 and re.match(r'^-?[\d.]+$', token):
                option['change'] = float(token)
            elif i == 4 and re.match(r'^[\d.]+$', token):
                option['breakeven'] = float(token)
        
        return option if option['vol'] else None
    
    def query_surface(self, index_name=None, tenor=None, data_date=None):
        """Query vol surface data"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM vol_surfaces WHERE 1=1"
        params = []
        
        if index_name:
            query += " AND index_name = ?"
            params.append(index_name)
        
        if tenor:
            query += " AND tenor = ?"
            params.append(tenor)
        
        if data_date:
            query += " AND data_date = ?"
            params.append(data_date)
        else:
            # Get most recent date if not specified
            query += " AND data_date = (SELECT MAX(data_date) FROM vol_surfaces)"
        
        query += " ORDER BY strike, option_type"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_available_dates(self):
        """Get list of available data dates"""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT DISTINCT data_date FROM vol_surfaces ORDER BY data_date DESC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df['data_date'].tolist() if not df.empty else []
    
    def get_available_indices(self, data_date=None):
        """Get list of available indices"""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT DISTINCT index_name FROM vol_surfaces"
        params = []
        if data_date:
            query += " WHERE data_date = ?"
            params.append(data_date)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df['index_name'].tolist() if not df.empty else []


# ============================================================================
# OPTION PRICING ENGINE
# ============================================================================

class CDSIndexOptionPricer:
    """
    Complete CDS Index Option Pricing System
    Properly accounts for exercise into 5Y underlying index
    Special handling for CDX HY with 500 bps fixed coupon
    """
    
    def __init__(self, db_path=None):
        if db_path is None:
            self.db_path = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\raw\vol_surfaces.db"
        else:
            self.db_path = db_path
        
        # Standard durations for CDS indices by maturity
        self.index_durations = {
            '6m': 0.5,
            '1m': 4.5,  # Options exercise into 5Y
            '2m': 4.5,
            '3m': 4.5,
            '4m': 4.5,
            '5m': 4.5,
            '6m': 4.5,
            '1y': 1.0,
            '2y': 1.8,
            '3y': 2.7,
            '5y': 4.5,
            '7y': 6.0,
            '10y': 8.0
        }
        
        # Option exercise into underlying index mapping
        self.exercise_mapping = {
            '1m': '5y',
            '2m': '5y',
            '3m': '5y',
            '4m': '5y',
            '5m': '5y',
            '6m': '5y'
        }
        
        # Fixed coupons for indices (in bps)
        self.fixed_coupons = {
            'EU_IG': 100,   # 100 bps for iTraxx IG
            'EU_XO': 500,   # 500 bps for iTraxx XO
            'US_IG': 100,   # 100 bps for CDX IG
            'US_HY': 500    # 500 bps for CDX HY
        }
    
    def calculate_risky_duration(self, spread_bps: float, tenor_years: float, 
                                 recovery_rate: float = 0.4, risk_free_rate: float = 0.03) -> float:
        """
        Calculate actual risky duration (RPV01) for CDS
        EXACTLY matching the Series Monitor calculation
        
        This MUST match the DV01 calculation used elsewhere in production
        """
        if spread_bps <= 0 or tenor_years <= 0:
            return tenor_years * 0.95
        
        # Convert spread to hazard rate - EXACTLY as in Series Monitor
        spread_decimal = spread_bps / 10000
        hazard_rate = spread_decimal / (1 - recovery_rate)
        
        # Combined discount rate (risk-free + credit) - EXACTLY as in Series Monitor
        total_discount = risk_free_rate + hazard_rate
        
        # Calculate RPV01 (Risky PV01) - EXACTLY as in Series Monitor
        if abs(total_discount * tenor_years) < 0.001:
            # Taylor expansion for numerical stability
            rpv01 = tenor_years * (1 - total_discount * tenor_years / 2 + 
                                  (total_discount * tenor_years)**2 / 6)
        else:
            # Full analytical formula
            rpv01 = (1 - np.exp(-total_discount * tenor_years)) / total_discount
        
        return rpv01
    
    def get_underlying_duration(self, option_tenor: str, index_name: str = None, 
                               forward_spread: float = None) -> float:
        """
        Get duration of underlying index that option exercises into
        Now calculates actual risky duration instead of using hardcoded values
        """
        # Options exercise into 5Y index
        underlying_tenor = self.exercise_mapping.get(option_tenor, '5y')
        tenor_years = float(underlying_tenor.replace('y', '').replace('Y', ''))
        
        # If we have the forward spread, calculate actual duration
        if forward_spread is not None and index_name is not None:
            recovery = self.recovery_rates.get(index_name, 0.4)
            return self.calculate_risky_duration(forward_spread, tenor_years, recovery)
        
        # Fallback to approximate duration if no spread available
        return tenor_years * 0.95
    
    def get_market_data(self, index_name: str, tenor: str, data_date: str = None) -> pd.DataFrame:
        """Retrieve market data from database"""
        if data_date is None:
            data_date = datetime.now().strftime("%Y-%m-%d")
        
        conn = sqlite3.connect(self.db_path)
        
        # First try the exact date
        query = """
            SELECT * FROM vol_surfaces
            WHERE index_name = ? AND tenor = ? 
            AND data_date = ?
            ORDER BY strike, option_type
        """
        df = pd.read_sql_query(query, conn, params=(index_name, tenor, data_date))
        
        # If no data for exact date, get most recent
        if df.empty:
            query = """
                SELECT * FROM vol_surfaces
                WHERE index_name = ? AND tenor = ? 
                AND data_date = (
                    SELECT MAX(data_date) FROM vol_surfaces 
                    WHERE data_date <= ? AND index_name = ? AND tenor = ?
                )
                ORDER BY strike, option_type
            """
            df = pd.read_sql_query(query, conn, params=(
                index_name, tenor, data_date, index_name, tenor
            ))
        
        conn.close()
        
        return df
    
    def price_option_cdx_hy(self, forward_price: float, strike_price: float, vol: float,
                            days_to_expiry: int, option_type: str = "Payer") -> Dict:
        """
        Special handling for CDX HY which has 500 bps fixed coupon
        Forward and strikes are in price terms but reflect spread dynamics
        Now uses calculated duration instead of hardcoded 4.5
        """
        
        T = days_to_expiry / 365.0
        
        # CDX HY specific parameters
        COUPON = 500  # bps fixed coupon
        
        # Convert prices to implied spreads (initial approximation)
        initial_duration = 4.5
        forward_spread = COUPON - (forward_price - 100) * 100 / initial_duration
        strike_spread = COUPON - (strike_price - 100) * 100 / initial_duration
        
        # Calculate actual duration based on forward spread
        DURATION = self.calculate_risky_duration(forward_spread, 5.0, 0.25)
        
        # Recalculate spreads with actual duration
        forward_spread = COUPON - (forward_price - 100) * 100 / DURATION
        strike_spread = COUPON - (strike_price - 100) * 100 / DURATION
        
        if T <= 0 or vol <= 0:
            # Intrinsic value in spread terms
            if option_type == "Receiver":  # Call on price = Put on spread
                spread_value = max(strike_spread - forward_spread, 0)
            else:  # Payer: Put on price = Call on spread
                spread_value = max(forward_spread - strike_spread, 0)
            
            # Convert back to price terms
            price_value = spread_value * DURATION / 100
            return {
                'spread_value': price_value,
                'upfront_bps': price_value * 100,
                'delta': 100.0 if spread_value > 0 else 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'implied_spread': forward_spread,
                'strike_spread': strike_spread,
                'duration_used': DURATION
            }
        
        # Black model on IMPLIED SPREADS
        sigma = vol / 100.0
        d1 = (np.log(forward_spread / strike_spread) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # IMPORTANT: Receiver on price = Put on spread (spread goes up, price goes down)
        if option_type == "Receiver":
            # Put on spread (Call on price)
            spread_value = strike_spread * norm.cdf(-d2) - forward_spread * norm.cdf(-d1)
            delta = -norm.cdf(-d1)  # Put delta on spread
        else:  # Payer
            # Call on spread (Put on price)
            spread_value = forward_spread * norm.cdf(d1) - strike_spread * norm.cdf(d2)
            delta = norm.cdf(d1)  # Call delta on spread
        
        # Convert spread option value to price points
        price_value = spread_value * DURATION / 100
        upfront_bps = price_value * 100
        
        # Adjust delta for price quotation (inverse relationship)
        price_delta = -delta * 100 if option_type == "Receiver" else delta * 100
        
        # Calculate gamma and vega in spread terms, then convert
        gamma = norm.pdf(d1) / (forward_spread * sigma * np.sqrt(T))
        vega = forward_spread * norm.pdf(d1) * np.sqrt(T) * DURATION / 10000
        theta = -(forward_spread * norm.pdf(d1) * sigma * DURATION / 100) / (2 * np.sqrt(T)) / 365
        
        return {
            'spread_value': price_value,
            'upfront_bps': upfront_bps,
            'delta': price_delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'implied_spread': forward_spread,
            'strike_spread': strike_spread,
            'duration_used': DURATION
        }
    
    def price_option(self, forward: float, strike: float, vol: float, 
                     days_to_expiry: int, option_type: str = "Payer",
                     underlying_duration: float = 4.5, model: str = "Black76",
                     index_name: str = None) -> Dict:
        """
        Price CDS index option with correct duration adjustment
        """
        
        # Special case for CDX HY (US_HY)
        if index_name == "US_HY":
            return self.price_option_cdx_hy(
                forward_price=forward,
                strike_price=strike,
                vol=vol,
                days_to_expiry=days_to_expiry,
                option_type=option_type
            )
        
        # Standard pricing for other indices
        T = days_to_expiry / 365.0
        
        if T <= 0 or vol <= 0:
            spread_value = max(forward - strike, 0) if option_type == "Payer" else max(strike - forward, 0)
            upfront_bps = spread_value * underlying_duration
            
            return {
                'spread_value': spread_value,
                'upfront_bps': upfront_bps,
                'delta': 100.0 if spread_value > 0 else 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'model_used': model
            }
        
        # Black-76 model for spread-based options
        sigma = vol / 100.0
        d1 = (np.log(forward / strike) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "Payer":
            value = forward * norm.cdf(d1) - strike * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:  # Receiver
            value = strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
        
        # Greeks
        gamma = norm.pdf(d1) / (forward * sigma * np.sqrt(T))
        
        # Apply duration multiplier
        upfront_bps = value * underlying_duration
        vega = forward * norm.pdf(d1) * np.sqrt(T) * underlying_duration / 100
        theta = -(forward * norm.pdf(d1) * sigma * underlying_duration) / (2 * np.sqrt(T)) / 365
        
        return {
            'spread_value': value,
            'upfront_bps': upfront_bps,
            'delta': delta * 100,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'model_used': model
        }
    
    def price_single_option(self, index_name: str, tenor: str, strike: float,
                           option_type: str = "Payer", notional: float = 10_000_000,
                           data_date: str = None) -> Dict:
        """
        Price a single option using stored vol surface
        """
        
        # Get market data
        market_data = self.get_market_data(index_name, tenor, data_date)
        
        if market_data.empty:
            raise ValueError(f"No market data found for {index_name} {tenor}")
        
        # Get parameters
        forward = market_data['forward_level'].iloc[0]
        expiry_str = market_data['expiry'].iloc[0]
        
        # Calculate days to expiry
        expiry_date = pd.to_datetime(expiry_str, format='%d-%b-%y')
        value_date = pd.to_datetime(data_date) if data_date else datetime.now()
        days_to_expiry = (expiry_date - value_date).days
        
        # Get underlying duration
        underlying_duration = self.get_underlying_duration(tenor)
        
        # Interpolate vol for the strike
        vol = self.interpolate_vol(market_data, strike, option_type)
        
        # Price the option
        result = self.price_option(
            forward=forward,
            strike=strike,
            vol=vol,
            days_to_expiry=days_to_expiry,
            option_type=option_type,
            underlying_duration=underlying_duration,
            index_name=index_name
        )
        
        # Add metadata and convert to currency
        result['index'] = index_name
        result['tenor'] = tenor
        result['strike'] = strike
        result['option_type'] = option_type
        result['forward'] = forward
        result['vol'] = vol
        result['days_to_expiry'] = days_to_expiry
        result['notional'] = notional
        result['duration_used'] = underlying_duration
        
        # Convert to currency
        result['upfront_currency'] = result['upfront_bps'] * notional / 10000
        result['vega_currency'] = result['vega'] * notional / 10000  # vega is already in bps
        result['theta_currency'] = result['theta'] * notional / 10000  # theta is in bps/day
        result['quotation'] = 'price' if index_name == "US_HY" else 'spread'
        print(f"DEBUG: {index_name} - Forward: {forward}, Duration calculated: {underlying_duration:.3f}")
        return result
    
    def interpolate_vol(self, market_data: pd.DataFrame, strike: float, 
                       option_type: str = "Payer") -> float:
        """Interpolate vol for a given strike from market surface"""
        
        df_type = market_data[market_data['option_type'] == option_type].copy()
        
        if df_type.empty:
            raise ValueError(f"No {option_type} options in surface")
        
        df_type = df_type.sort_values('strike')
        
        # If strike is outside range, use flat extrapolation
        if strike <= df_type['strike'].min():
            return df_type.iloc[0]['vol']
        elif strike >= df_type['strike'].max():
            return df_type.iloc[-1]['vol']
        
        # Linear interpolation
        return np.interp(strike, df_type['strike'], df_type['vol'])
    
    def validate_pricing(self, index_name: str, tenor: str, data_date: str = None) -> pd.DataFrame:
        """
        Validate model against market prices
        Returns comparison dataframe with accuracy metrics
        """
        
        market_data = self.get_market_data(index_name, tenor, data_date)
        
        if market_data.empty:
            return pd.DataFrame()
        
        # Get parameters
        forward = market_data['forward_level'].iloc[0]
        expiry_str = market_data['expiry'].iloc[0]
        
        # Calculate days to expiry
        expiry_date = pd.to_datetime(expiry_str, format='%d-%b-%y')
        value_date = pd.to_datetime(data_date) if data_date else datetime.now()
        days_to_expiry = (expiry_date - value_date).days
        
        # Get underlying duration
        underlying_duration = self.get_underlying_duration(tenor)
        
        results = []
        for _, row in market_data.iterrows():
            if pd.isna(row['mid']):
                continue
            
            model = self.price_option(
                forward=forward,
                strike=row['strike'],
                vol=row['vol'],
                days_to_expiry=days_to_expiry,
                option_type=row['option_type'],
                underlying_duration=underlying_duration,
                index_name=index_name
            )
            
            # Calculate accuracy metrics
            difference = model['upfront_bps'] - row['mid']
            pct_error = (difference / row['mid'] * 100) if row['mid'] != 0 else 0
            within_spread = False
            
            if pd.notna(row['bid']) and pd.notna(row['ask']):
                within_spread = row['bid'] <= model['upfront_bps'] <= row['ask']
            
            result_row = {
                'strike': row['strike'],
                'option_type': row['option_type'],
                'market_bid': row['bid'],
                'market_ask': row['ask'],
                'market_mid': row['mid'],
                'model_price': model['upfront_bps'],
                'difference': difference,
                'pct_error': pct_error,
                'within_spread': within_spread,
                'market_vol': row['vol'],
                'model_delta': model['delta'],
                'quotation': 'price' if index_name == "US_HY" else 'spread'
            }
            
            # Add CDX HY specific info
            if index_name == "US_HY" and 'implied_spread' in model:
                result_row['implied_spread'] = model['implied_spread']
                result_row['strike_spread'] = model['strike_spread']
            
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def price_strategy(self, legs: List[Dict], data_date: str = None) -> Dict:
        """
        Price multi-leg option strategy
        """
        
        total_upfront = 0
        total_delta_weighted = 0
        total_gamma = 0
        total_vega = 0
        total_theta = 0
        
        leg_results = []
        
        for leg in legs:
            # Price individual leg
            leg_price = self.price_single_option(
                index_name=leg['index_name'],
                tenor=leg['tenor'],
                strike=leg['strike'],
                option_type=leg['option_type'],
                notional=abs(leg.get('notional', 10_000_000)),
                data_date=data_date
            )
            
            # Apply position
            position = leg.get('position', 1)
            notional_ratio = leg.get('notional', 10_000_000) / 10_000_000
            
            # Aggregate
            total_upfront += position * leg_price['upfront_currency']
            total_delta_weighted += position * leg_price['delta'] * notional_ratio
            total_gamma += position * leg_price['gamma'] * notional_ratio
            total_vega += position * leg_price['vega_currency']
            total_theta += position * leg_price['theta_currency']
            
            # Store results
            leg_results.append({
                'leg': leg,
                'price': leg_price,
                'contribution': position * leg_price['upfront_currency']
            })
        
        return {
            'total_upfront': total_upfront,
            'total_delta': total_delta_weighted,
            'total_gamma': total_gamma,
            'total_vega': total_vega,
            'total_theta': total_theta,
            'legs': leg_results,
            'strategy_type': self._identify_strategy_type(legs)
        }
    
    def _identify_strategy_type(self, legs: List[Dict]) -> str:
        """Identify common strategy patterns"""
        if len(legs) == 1:
            return "Single Option"
        elif len(legs) == 2:
            if legs[0]['option_type'] == legs[1]['option_type']:
                if legs[0]['position'] * legs[1]['position'] < 0:
                    if legs[0]['strike'] < legs[1]['strike']:
                        return "Bull Spread" if legs[0]['option_type'] == "Payer" else "Bear Spread"
                    else:
                        return "Bear Spread" if legs[0]['option_type'] == "Payer" else "Bull Spread"
            else:
                if legs[0]['strike'] == legs[1]['strike']:
                    return "Straddle" if legs[0]['position'] == legs[1]['position'] else "Synthetic"
        elif len(legs) == 4:
            strikes = sorted([leg['strike'] for leg in legs])
            if len(set(strikes)) == 4:
                return "Condor"
            elif len(set(strikes)) == 3:
                return "Butterfly"
        
        return "Custom Strategy"


# ============================================================================
# DAILY UPDATE FUNCTION
# ============================================================================

def daily_update(excel_path=None, force_update=False):
    """
    Daily update script - can be called from notebook, script, or scheduler
    """
    
    if excel_path is None:
        excel_path = r"C:\Users\alessandro.esposito\Portman Square Capital LLP\Portman Square Capital - Documents\S\CSA\Credit Index Trading\vol_db.xlsx"
    
    vol_db = VolSurfaceDatabase()
    today = datetime.now().strftime("%Y-%m-%d")
    
    if not force_update and vol_db.check_if_date_exists(today):
        print(f"Data for {today} already exists. Use force_update=True to overwrite.")
        return 0
    
    total = vol_db.create_or_update_database(excel_path, data_date=today, force_update=force_update)
    print(f"Updated {total} options for {today}")
    
    return total


# ============================================================================
# VALIDATION AND TESTING FUNCTIONS
# ============================================================================

def run_validation_tests(data_date=None):
    """
    Run comprehensive validation tests
    """
    
    pricer = CDSIndexOptionPricer()
    
    if data_date is None:
        data_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\nRUNNING VALIDATION TESTS FOR {data_date}")
    print("="*80)
    
    # Test each index
    indices = ['EU_IG', 'EU_XO', 'US_IG', 'US_HY']
    tenors = ['3m']
    
    all_results = []
    
    for index_name in indices:
        for tenor in tenors:
            print(f"\nValidating {index_name} {tenor}")
            print("-"*40)
            
            try:
                validation = pricer.validate_pricing(index_name, tenor, data_date)
                
                if not validation.empty:
                    # Calculate accuracy metrics
                    within_spread = validation['within_spread'].sum()
                    total = len(validation)
                    accuracy = within_spread / total * 100 if total > 0 else 0
                    
                    avg_error = validation['difference'].abs().mean()
                    avg_pct_error = validation['pct_error'].abs().mean()
                    
                    print(f"  Options priced: {total}")
                    print(f"  Within bid/ask: {within_spread}/{total} ({accuracy:.1f}%)")
                    print(f"  Avg absolute error: {avg_error:.2f} bps")
                    print(f"  Avg % error: {avg_pct_error:.1f}%")
                    
                    # Special info for CDX HY
                    if index_name == "US_HY" and 'implied_spread' in validation.columns:
                        avg_implied = validation['implied_spread'].mean()
                        print(f"  Avg implied spread: {avg_implied:.1f} bps")
                    
                    validation['index'] = index_name
                    validation['tenor'] = tenor
                    all_results.append(validation)
                else:
                    print(f"  No data available")
            
            except Exception as e:
                print(f"  Error: {e}")
    
    if all_results:
        full_results = pd.concat(all_results, ignore_index=True)
        
        print("\n" + "="*80)
        print("OVERALL ACCURACY SUMMARY")
        print("-"*40)
        
        total_within = full_results['within_spread'].sum()
        total_options = len(full_results)
        overall_accuracy = total_within / total_options * 100 if total_options > 0 else 0
        
        print(f"Total options validated: {total_options}")
        print(f"Within bid/ask spread: {total_within} ({overall_accuracy:.1f}%)")
        print(f"Average absolute error: {full_results['difference'].abs().mean():.2f} bps")
        
        return full_results
    
    return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    print("Credit Index Option Pricing System")
    print("="*80)
    
    # Check if we need to update the database
    vol_db = VolSurfaceDatabase()
    today = datetime.now().strftime("%Y-%m-%d")
    
    if not vol_db.check_if_date_exists(today):
        print(f"\nNo data for {today}. Please run daily_update() first.")
    else:
        print(f"\nDatabase has data for {today}")
        
        # Run validation tests
        validation_results = run_validation_tests(today)
        
        if not validation_results.empty:
            print(f"\nValidation complete. Results stored in 'validation_results' DataFrame")