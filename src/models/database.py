"""
Database layer for CDS data persistence
SQLite implementation for storing historical data and positions
"""
import sqlite3
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from xbbg import blp

from .spread_data import CDSSpreadData
from .curve import CDSCurve
from .position import Position, Strategy
from .cds_index import CDSIndexDefinition
from .enums import Region, Market, Tenor, Side

logger = logging.getLogger(__name__)

# CDS Series Roll Schedule - Extended back 10 years
ROLL_DATES = [
    ('2025-09-20', '2026-03-19', {'ITRX': 44, 'CDX_IG': 45, 'CDX_HY': 45}),
    ('2025-03-20', '2025-09-19', {'ITRX': 43, 'CDX_IG': 44, 'CDX_HY': 44}),
    ('2024-09-20', '2025-03-19', {'ITRX': 42, 'CDX_IG': 43, 'CDX_HY': 43}),
    ('2024-03-20', '2024-09-19', {'ITRX': 41, 'CDX_IG': 42, 'CDX_HY': 42}),
    ('2023-09-20', '2024-03-19', {'ITRX': 40, 'CDX_IG': 41, 'CDX_HY': 41}),
    ('2023-03-20', '2023-09-19', {'ITRX': 39, 'CDX_IG': 40, 'CDX_HY': 40}),
    ('2022-09-20', '2023-03-19', {'ITRX': 38, 'CDX_IG': 39, 'CDX_HY': 39}),
    ('2022-03-20', '2022-09-19', {'ITRX': 37, 'CDX_IG': 38, 'CDX_HY': 38}),
    ('2021-09-20', '2022-03-19', {'ITRX': 36, 'CDX_IG': 37, 'CDX_HY': 37}),
    ('2021-03-20', '2021-09-19', {'ITRX': 35, 'CDX_IG': 36, 'CDX_HY': 36}),
    ('2020-09-20', '2021-03-19', {'ITRX': 34, 'CDX_IG': 35, 'CDX_HY': 35}),
    ('2020-03-20', '2020-09-19', {'ITRX': 33, 'CDX_IG': 34, 'CDX_HY': 34}),
    ('2019-09-20', '2020-03-19', {'ITRX': 32, 'CDX_IG': 33, 'CDX_HY': 33}),
    ('2019-03-20', '2019-09-19', {'ITRX': 31, 'CDX_IG': 32, 'CDX_HY': 32}),
    ('2018-09-20', '2019-03-19', {'ITRX': 30, 'CDX_IG': 31, 'CDX_HY': 31}),
    ('2018-03-20', '2018-09-19', {'ITRX': 29, 'CDX_IG': 30, 'CDX_HY': 30}),
    ('2017-09-20', '2018-03-19', {'ITRX': 28, 'CDX_IG': 29, 'CDX_HY': 29}),
    ('2017-03-20', '2017-09-19', {'ITRX': 27, 'CDX_IG': 28, 'CDX_HY': 28}),
    ('2016-09-20', '2017-03-19', {'ITRX': 26, 'CDX_IG': 27, 'CDX_HY': 27}),
    ('2016-03-20', '2016-09-19', {'ITRX': 25, 'CDX_IG': 26, 'CDX_HY': 26}),
    ('2015-09-20', '2016-03-19', {'ITRX': 24, 'CDX_IG': 25, 'CDX_HY': 25}),
    ('2015-03-20', '2015-09-19', {'ITRX': 23, 'CDX_IG': 24, 'CDX_HY': 24}),
]

SERIES_MAPPINGS = {
    'EU_IG': {
        'base': 'ITXEB', 
        'family': 'ITRX',
        'tenors': {'1Y': '143', '3Y': '343', '5Y': '543', '7Y': '743', '10Y': '043'}
    },
    'EU_XO': {
        'base': 'ITXEX',
        'family': 'ITRX', 
        'tenors': {'3Y': '343', '5Y': '543', '7Y': '743', '10Y': '043'}
    },
    'US_IG': {
        'base': 'CDXIG',
        'family': 'CDX_IG',
        'tenors': {'3Y': '344', '5Y': '544', '7Y': '744', '10Y': '044'}
    },
    'US_HY': {
        'base': 'CDXHY',
        'family': 'CDX_HY',
        'tenors': {'3Y': '343', '5Y': '543', '7Y': '743', '10Y': '043'},
        'suffix': 'BEST'
    }
}


class CDSDatabase:
    """SQLite database for storing CDS data"""
    
    def __init__(self, db_path: Union[str, Path] = "data/cds_data.db"):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self._init_tables()
        logger.info(f"Database initialized at {self.db_path}")
    
    def _init_tables(self):
        """Initialize database tables"""
        
        # Index definitions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS index_definitions (
                index_id TEXT PRIMARY KEY,
                region TEXT NOT NULL,
                market TEXT NOT NULL,
                series INTEGER NOT NULL,
                tenor TEXT NOT NULL,
                ticker TEXT NOT NULL,
                start_date DATE,
                maturity_date DATE,
                recovery_rate REAL,
                coupon REAL,
                notional REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(region, market, series, tenor)
            )
        """)
        
        # Spread data table with partitioning by date for performance
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spread_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                bid REAL,
                ask REAL,
                mid REAL,
                last REAL,
                volume REAL,
                dv01 REAL,
                change_1d REAL,
                change_pct_1d REAL,
                FOREIGN KEY (index_id) REFERENCES index_definitions(index_id),
                UNIQUE(index_id, timestamp)
            )
        """)
        
        # Create index for faster queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_spread_data_timestamp 
            ON spread_data(index_id, timestamp DESC)
        """)
        
        # Curves table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS curves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region TEXT NOT NULL,
                market TEXT NOT NULL,
                series INTEGER NOT NULL,
                observation_date TIMESTAMP NOT NULL,
                curve_data TEXT NOT NULL,  -- JSON serialized
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(region, market, series, observation_date)
            )
        """)
        
        # Positions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_id TEXT NOT NULL,
                strategy_name TEXT,
                side INTEGER NOT NULL,
                notional REAL NOT NULL,
                entry_date TIMESTAMP NOT NULL,
                entry_spread REAL NOT NULL,
                entry_dv01 REAL NOT NULL,
                exit_date TIMESTAMP,
                exit_spread REAL,
                exit_dv01 REAL,
                pnl REAL,
                status TEXT DEFAULT 'OPEN',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (index_id) REFERENCES index_definitions(index_id)
            )
        """)
        
        # Strategies table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_name TEXT PRIMARY KEY,
                strategy_type TEXT NOT NULL,
                creation_date TIMESTAMP NOT NULL,
                target_dv01 REAL,
                stop_loss REAL,
                take_profit REAL,
                status TEXT DEFAULT 'ACTIVE',
                metadata TEXT,  -- JSON for additional fields
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # P&L history table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pnl_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                date DATE NOT NULL,
                daily_pnl REAL NOT NULL,
                cumulative_pnl REAL NOT NULL,
                positions_count INTEGER,
                net_dv01 REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (strategy_name) REFERENCES strategies(strategy_name),
                UNIQUE(strategy_name, date)
            )
        """)
        
        self.conn.commit()
    
    def _init_historical_tables(self):
        """Initialize tables for historical raw data"""
        
        # Raw historical spreads table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_historical_spreads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                index_name TEXT NOT NULL,
                tenor TEXT NOT NULL,
                spread_bps REAL NOT NULL,
                series_number INTEGER NOT NULL,
                bloomberg_ticker TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, index_name, tenor)
            )
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_raw_historical_lookup 
            ON raw_historical_spreads(index_name, tenor, date DESC)
        """)
        
        self.conn.commit()
    
    def save_index_definition(self, index_def: CDSIndexDefinition):
        """Save or update index definition"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO index_definitions 
                (index_id, region, market, series, tenor, ticker, 
                 start_date, maturity_date, recovery_rate, coupon, notional)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                index_def.unique_id,
                index_def.region.value,
                index_def.market.value,
                index_def.series,
                index_def.tenor.value,
                index_def.ticker,
                index_def.start_date,
                index_def.maturity_date,
                index_def.recovery_rate,
                index_def.coupon,
                index_def.notional
            ))
            self.conn.commit()
            logger.debug(f"Saved index definition: {index_def.unique_id}")
        except Exception as e:
            logger.error(f"Error saving index definition: {e}")
            self.conn.rollback()
            raise
    
    def save_spread_data(self, spread_data: Union[CDSSpreadData, List[CDSSpreadData]]):
        """Save spread data (single or batch)"""
        if isinstance(spread_data, CDSSpreadData):
            spread_data = [spread_data]
        
        try:
            for spread in spread_data:
                self.conn.execute("""
                    INSERT OR REPLACE INTO spread_data
                    (index_id, timestamp, bid, ask, mid, last, 
                     volume, dv01, change_1d, change_pct_1d)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    spread.index_id,
                    spread.timestamp,
                    spread.bid,
                    spread.ask,
                    spread.mid,
                    spread.last,
                    spread.volume,
                    spread.dv01,
                    spread.change_1d,
                    spread.change_pct_1d
                ))
            self.conn.commit()
            logger.debug(f"Saved {len(spread_data)} spread data points")
        except Exception as e:
            logger.error(f"Error saving spread data: {e}")
            self.conn.rollback()
            raise
    
    def save_curve(self, curve: CDSCurve):
        """Save curve data"""
        try:
            curve_json = json.dumps({
                'spreads': {t.value: s for t, s in curve.spreads.items()},
                'dv01s': {t.value: d for t, d in curve.dv01s.items()}
            })
            
            self.conn.execute("""
                INSERT OR REPLACE INTO curves
                (region, market, series, observation_date, curve_data)
                VALUES (?, ?, ?, ?, ?)
            """, (
                curve.region.value,
                curve.market.value,
                curve.series,
                curve.observation_date,
                curve_json
            ))
            self.conn.commit()
            logger.debug(f"Saved curve: {curve.curve_id}")
        except Exception as e:
            logger.error(f"Error saving curve: {e}")
            self.conn.rollback()
            raise
    
    def get_latest_spread(self, index_id: str) -> Optional[CDSSpreadData]:
        """Get latest spread for an index"""
        cursor = self.conn.execute("""
            SELECT * FROM spread_data 
            WHERE index_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (index_id,))
        
        row = cursor.fetchone()
        if row:
            return CDSSpreadData(
                index_id=row['index_id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                bid=row['bid'],
                ask=row['ask'],
                mid=row['mid'],
                last=row['last'],
                volume=row['volume'],
                dv01=row['dv01'],
                change_1d=row['change_1d'],
                change_pct_1d=row['change_pct_1d']
            )
        return None
    
    def get_historical_spreads(self, 
                              index_id: str,
                              start_date: datetime,
                              end_date: datetime = None) -> pd.DataFrame:
        """Get historical spreads for an index"""
        if end_date is None:
            end_date = datetime.now()
        
        query = """
            SELECT timestamp, bid, ask, mid, last, dv01, volume,
                   change_1d, change_pct_1d
            FROM spread_data
            WHERE index_id = ? 
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """
        
        df = pd.read_sql_query(
            query,
            self.conn,
            params=(index_id, start_date.isoformat(), end_date.isoformat()),
            parse_dates=['timestamp']
        )
        
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def save_position(self, position: Position, strategy_name: str = None):
        """Save position to database"""
        try:
            cursor = self.conn.execute("""
                INSERT INTO positions
                (index_id, strategy_name, side, notional, 
                 entry_date, entry_spread, entry_dv01, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """, (
                position.index_id,
                strategy_name,
                position.side.value,
                position.notional,
                position.entry_date,
                position.entry_spread,
                position.entry_dv01
            ))
            self.conn.commit()
            logger.debug(f"Saved position: {cursor.lastrowid}")
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving position: {e}")
            self.conn.rollback()
            raise
    
    def update_position_exit(self, 
                           position_id: int,
                           exit_date: datetime,
                           exit_spread: float,
                           exit_dv01: float,
                           pnl: float):
        """Update position with exit information"""
        try:
            self.conn.execute("""
                UPDATE positions
                SET exit_date = ?, exit_spread = ?, exit_dv01 = ?, 
                    pnl = ?, status = 'CLOSED', updated_at = CURRENT_TIMESTAMP
                WHERE position_id = ?
            """, (exit_date, exit_spread, exit_dv01, pnl, position_id))
            self.conn.commit()
            logger.debug(f"Closed position: {position_id}")
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            self.conn.rollback()
            raise
    
    def save_strategy(self, strategy: Strategy):
        """Save or update strategy"""
        try:
            metadata_json = json.dumps(strategy.metadata) if strategy.metadata else None
            
            self.conn.execute("""
                INSERT OR REPLACE INTO strategies
                (strategy_name, strategy_type, creation_date, 
                 target_dv01, stop_loss, take_profit, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy.name,
                strategy.strategy_type,
                strategy.creation_date,
                strategy.target_dv01,
                strategy.stop_loss,
                strategy.take_profit,
                metadata_json
            ))
            
            # Save all positions
            for position in strategy.positions:
                self.save_position(position, strategy.name)
            
            self.conn.commit()
            logger.debug(f"Saved strategy: {strategy.name}")
        except Exception as e:
            logger.error(f"Error saving strategy: {e}")
            self.conn.rollback()
            raise
    
    def get_open_positions(self, strategy_name: Optional[str] = None) -> List[Dict]:
        """Get all open positions"""
        query = """
            SELECT * FROM positions 
            WHERE status = 'OPEN'
        """
        params = []
        
        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)
        
        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def save_pnl_history(self, strategy_name: str, date: datetime, 
                        daily_pnl: float, cumulative_pnl: float,
                        positions_count: int = None, net_dv01: float = None):
        """Save P&L history for a strategy"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO pnl_history
                (strategy_name, date, daily_pnl, cumulative_pnl, 
                 positions_count, net_dv01)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                strategy_name,
                date.date() if hasattr(date, 'date') else date,
                daily_pnl,
                cumulative_pnl,
                positions_count,
                net_dv01
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving P&L history: {e}")
            self.conn.rollback()
            raise
    
    def get_pnl_history(self, strategy_name: str, 
                       start_date: datetime = None,
                       end_date: datetime = None) -> pd.DataFrame:
        """Get P&L history for a strategy"""
        query = """
            SELECT date, daily_pnl, cumulative_pnl, positions_count, net_dv01
            FROM pnl_history
            WHERE strategy_name = ?
        """
        params = [strategy_name]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date.date() if hasattr(start_date, 'date') else start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.date() if hasattr(end_date, 'date') else end_date)
        
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, self.conn, params=params, parse_dates=['date'])
        
        if not df.empty:
            df.set_index('date', inplace=True)
        
        return df
    
    def get_series_for_date(self, target_date: datetime, index_family: str) -> int:
        """Get the correct series number for a given date and index family"""
        target_str = target_date.strftime('%Y-%m-%d')
        
        for start_date, end_date, series_map in ROLL_DATES:
            if start_date <= target_str <= end_date:
                return series_map.get(index_family, 43)
        
        # Extrapolate backwards for older dates
        earliest_date = datetime.strptime(ROLL_DATES[-1][0], '%Y-%m-%d')
        if target_date < earliest_date:
            months_back = ((earliest_date.year - target_date.year) * 12 + 
                          (earliest_date.month - target_date.month))
            periods_back = (months_back // 6) + 1
            base_series = ROLL_DATES[-1][2].get(index_family, 23)
            return max(base_series - periods_back, 1)
        
        return series_map.get(index_family, 43)

    def build_historical_tickers(self, region_market: str, target_date: datetime) -> dict:
        """Build Bloomberg tickers for specific date using appropriate series"""
        config = SERIES_MAPPINGS[region_market]
        family = config['family']
        series = self.get_series_for_date(target_date, family)
        
        tickers = {}
        for tenor in config['tenors'].keys():
            tenor_num = tenor[0] if tenor != '10Y' else '0'
            ticker_suffix = f"{tenor_num}{series:02d}"
            
            if config.get('suffix'):  # US_HY case
                ticker = f"{config['base']}{ticker_suffix} {config['suffix']} Curncy"
            else:
                ticker = f"{config['base']}{ticker_suffix} Curncy"
                
            tickers[tenor] = ticker
        
        return tickers

    def populate_historical_raw_data(self, years_back: int = 10):
        """
        Populate database with historical raw CDS data using series roll logic
        Integrates with existing database structure
        """
        
        # Initialize historical tables
        self._init_historical_tables()
        
        start_date = datetime.now() - timedelta(days=365 * years_back)
        end_date = datetime.now()
        
        print(f"Populating historical raw data")
        print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print("=" * 60)
        
        # Generate business days
        business_days = pd.bdate_range(start_date, end_date, freq='B')
        print(f"Processing {len(business_days)} business days...")
        
        batch_data = []
        batch_size = 500
        
        try:
            for i, date in enumerate(business_days):
                if i % 100 == 0:
                    print(f"  Progress: {i+1}/{len(business_days)} days ({date.strftime('%Y-%m-%d')})")
                
                # Get data for each index on this date
                for region_market in SERIES_MAPPINGS.keys():
                    family = SERIES_MAPPINGS[region_market]['family']
                    series = self.get_series_for_date(date, family)
                    
                    # Build tickers for this specific date
                    tickers = self.build_historical_tickers(region_market, date)
                    
                    if not tickers:
                        continue
                        
                    try:
                        # Get Bloomberg data for this exact date
                        data = blp.bdh(list(tickers.values()), flds='PX_LAST',
                                      start_date=date.strftime('%Y%m%d'),
                                      end_date=date.strftime('%Y%m%d'))
                        
                        if data is not None and not data.empty:
                            if isinstance(data.columns, pd.MultiIndex):
                                data.columns = data.columns.get_level_values(0)
                            
                            # Store raw spread for each tenor
                            for tenor, ticker in tickers.items():
                                if ticker in data.columns and len(data) > 0:
                                    raw_spread = data.iloc[0][ticker]
                                    
                                    if pd.notna(raw_spread):
                                        batch_data.append({
                                            'date': date.date(),
                                            'index_name': region_market,
                                            'tenor': tenor,
                                            'spread_bps': float(raw_spread),
                                            'series_number': series,
                                            'bloomberg_ticker': ticker
                                        })
                                        
                                if len(batch_data) >= batch_size:
                                    self._insert_historical_batch(batch_data)
                                    batch_data = []
                                    
                    except Exception as e:
                        continue
            
            # Insert remaining data
            if batch_data:
                self._insert_historical_batch(batch_data)
            
            # Print summary
            cursor = self.conn.execute("SELECT COUNT(*) FROM raw_historical_spreads")
            total_records = cursor.fetchone()[0]
            
            print(f"\nHistorical raw data population complete!")
            print(f"Total records: {total_records:,}")
            
            # Show example data
            cursor = self.conn.execute("""
                SELECT date, index_name, tenor, spread_bps, series_number 
                FROM raw_historical_spreads 
                WHERE index_name = 'US_HY' AND tenor = '3Y'
                ORDER BY date DESC LIMIT 5
            """)
            
            print(f"\nExample - US_HY 3Y Raw Spreads:")
            for row in cursor.fetchall():
                date, index_name, tenor, spread, series = row
                print(f"  {date}: {spread:.2f} bps (S{series})")
            
            logger.info(f"Historical data population completed: {total_records:,} records")
            
        except Exception as e:
            logger.error(f"Error populating historical data: {e}")
            self.conn.rollback()
            raise

    def _insert_historical_batch(self, batch_data):
        """Insert batch of historical data"""
        if not batch_data:
            return
        
        try:
            self.conn.executemany("""
                INSERT OR IGNORE INTO raw_historical_spreads
                (date, index_name, tenor, spread_bps, series_number, bloomberg_ticker)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                (row['date'], row['index_name'], row['tenor'], 
                 row['spread_bps'], row['series_number'], row['bloomberg_ticker'])
                for row in batch_data
            ])
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error inserting historical batch: {e}")
            self.conn.rollback()
            raise

    def query_historical_spreads(self, index_name=None, tenor=None, 
                               start_date=None, end_date=None) -> pd.DataFrame:
        """Query historical raw spreads from database"""
        
        query = "SELECT * FROM raw_historical_spreads WHERE 1=1"
        params = []
        
        if index_name:
            query += " AND index_name = ?"
            params.append(index_name)
            
        if tenor:
            query += " AND tenor = ?"
            params.append(tenor)
            
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
            
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date DESC"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            
        return df

    def update_series_mapping(self, new_roll_date: datetime):
        """Update series numbers when new indices roll - for future use"""
        # This method can be called every 6 months to update the roll schedule
        # and start collecting new on-the-run data
        logger.info(f"Series mapping update triggered for {new_roll_date}")
        # Implementation for future automated updates
        pass
    
    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("Database connection closed")