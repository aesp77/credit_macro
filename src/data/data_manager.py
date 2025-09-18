"""
High-level data management operations
Orchestrates Bloomberg connector and database operations
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging

from .bloomberg_connector import BloombergCDSConnector
from ..models.database import CDSDatabase
from ..models.enums import Region, Market, Tenor
from ..models.spread_data import CDSSpreadData
from ..models.curve import CDSCurve
from ..models.cds_index import CDSIndexDefinition

logger = logging.getLogger(__name__)


class CDSDataManager:
    """
    High-level data management class
    Coordinates between Bloomberg API and database
    """
    
    def __init__(self, 
                 connector: BloombergCDSConnector = None,
                 database: CDSDatabase = None,
                 db_path: str = "data/cds_monitor.db"):
        """
        Initialize the data manager
        
        Args:
            connector: Bloomberg connector instance
            database: Database instance
            db_path: Path to database file
        """
        self.connector = connector or BloombergCDSConnector()
        self.db = database or CDSDatabase(db_path)
        logger.info("CDS Data Manager initialized")
    
    def load_main_indices(self, 
                        num_series_back: int = 4,
                        save_to_db: bool = True) -> pd.DataFrame:
        """
        Load main indices for both EU and US markets
        """
        def get_current_series_number(region: str) -> int:
            today = datetime.now()
            ref_date = datetime(2025, 3, 20)
            ref_series = {'EU': 43, 'US': 44}[region]
            months_diff = (today.year - ref_date.year) * 12 + today.month - ref_date.month
            if today.day < 20:
                months_diff -= 1
            rolls_since_ref = months_diff // 6
            return ref_series + rolls_since_ref
        
        all_spreads = []
        
        # Define configurations for each region
        configs = {
            'EU': {
                'IG': {'base': 'ITRX EUR CDSI', 'current': get_current_series_number('EU')},
                'XO': {'base': 'ITRX EUR XOVER', 'current': get_current_series_number('EU')}
            },
            'US': {
                'IG': {'base': 'CDX IG', 'current': get_current_series_number('US')},
                'HY': {'base': 'CDX HY', 'current': get_current_series_number('US')}
            }
        }
        
        # Fetch spreads for each region/market combination
        for region, markets in configs.items():
            for market, info in markets.items():
                current_series = info['current']
                logger.info(f"Loading {region} {market} S{current_series}")
                
                for series in range(current_series - num_series_back + 1, current_series + 1):
                    for tenor in ['3Y', '5Y', '7Y', '10Y']:
                        try:
                            ticker = self.connector.get_index_ticker(region, market, series, tenor)
                            spread = self.connector.get_current_spread(ticker)
                            
                            if spread is not None and not spread.empty:
                                spread['region'] = region
                                spread['market'] = market
                                spread['series'] = series
                                spread['tenor'] = tenor
                                all_spreads.append(spread)
                        except Exception as e:
                            logger.debug(f"Could not fetch {region} {market} S{series} {tenor}: {e}")
        
        if all_spreads:
            all_data = pd.concat(all_spreads, ignore_index=True)
            logger.info(f"Loaded {len(all_data)} index spreads")
        else:
            all_data = pd.DataFrame()
            logger.warning("No spreads loaded")
        
        return all_data
    
    def update_spread_data(self, 
                          index_id: str,
                          region: Union[str, Region],
                          market: Union[str, Market],
                          series: int,
                          tenor: Union[str, Tenor]) -> CDSSpreadData:
        """
        Update spread data for a specific index
        
        Args:
            index_id: Unique index identifier
            region: Region
            market: Market segment
            series: Series number
            tenor: Tenor
            
        Returns:
            CDSSpreadData object
        """
        # Get ticker
        ticker = self.connector.get_index_ticker(region, market, series, tenor)
        
        # Fetch current spread
        spread_df = self.connector.get_current_spread(ticker)
        
        if spread_df is None or spread_df.empty:
            logger.warning(f"No spread data for {ticker}")
            return None
        
        # Create spread data object
        spread_data = CDSSpreadData(
            index_id=index_id,
            timestamp=datetime.now(),
            bid=spread_df.get('PX_BID', [np.nan]).iloc[0],
            ask=spread_df.get('PX_ASK', [np.nan]).iloc[0],
            mid=(spread_df.get('PX_BID', [0]).iloc[0] + 
                 spread_df.get('PX_ASK', [0]).iloc[0]) / 2,
            last=spread_df.get('PX_LAST', [np.nan]).iloc[0],
            volume=spread_df.get('VOLUME', [np.nan]).iloc[0],
            dv01=spread_df.get('RISK_MID', [np.nan]).iloc[0],
            change_1d=spread_df.get('CHG_NET_1D', [np.nan]).iloc[0],
            change_pct_1d=spread_df.get('CHG_PCT_1D', [np.nan]).iloc[0]
        )
        
        # Save to database
        self.db.save_spread_data(spread_data)
        
        return spread_data
    
    def build_curve(self, 
                    region: Region,
                    market: Market, 
                    series: int,
                    save_to_db: bool = False) -> CDSCurve:
        """
        Build a full credit curve for a given index
        
        Args:
            region: Region
            market: Market segment
            series: Series number
            save_to_db: Whether to save to database
            
        Returns:
            CDSCurve object
        """
        # Convert to enums if needed
        if isinstance(region, str):
            region = Region(region)
        if isinstance(market, str):
            market = Market(market)
        
        logger.info(f"Building curve for {region.value} {market.value} S{series}")
        
        # Get base ticker pattern
        if region == Region.EU and market == Market.IG:
            base_ticker = "ITRX EUR CDSI"
        elif region == Region.EU and market == Market.XO:
            base_ticker = "ITRX EUR XOVER"
        elif region == Region.US and market == Market.IG:
            base_ticker = "CDX IG"
        else:
            logger.error(f"Unknown curve type: {region} {market}")
            return None
        
        # Instead of using get_curve_data, fetch each tenor individually
        # This uses the working individual ticker approach
        spreads = {}
        dv01s = {}
        
        for tenor_str in ['3Y', '5Y', '7Y', '10Y']:
            try:
                tenor = Tenor(tenor_str)
                ticker = self.connector.get_index_ticker(region.value, market.value, series, tenor_str)
                spread_data = self.connector.get_current_spread(ticker)
                
                if spread_data is not None and not spread_data.empty:
                    # Use lowercase column names as shown in your debug
                    spreads[tenor] = spread_data['px_last'].iloc[0]
                    # Calculate approximate DV01 if not available
                    if 'dv01' in spread_data.columns:
                        dv01s[tenor] = spread_data['dv01'].iloc[0]
                    else:
                        # Rough DV01 approximation: spread * notional * duration / 10000
                        years = float(tenor_str.replace('Y', ''))
                        approx_dv01 = spreads[tenor] * 10_000_000 * years * 0.01 / 10000
                        dv01s[tenor] = approx_dv01
                else:
                    logger.warning(f"No data for {ticker}")
                    spreads[tenor] = np.nan
                    dv01s[tenor] = np.nan
                    
            except ValueError as e:
                logger.warning(f"Error processing tenor {tenor_str}: {e}")
        
        curve = CDSCurve(
            region=region,
            market=market,
            series=series,
            observation_date=datetime.now(),
            spreads=spreads,
            dv01s=dv01s
        )
        
        if save_to_db:
            self.db.save_curve(curve)
        
        return curve
    
    def calculate_strategy_metrics(self, 
                                  strategy_type: str,
                                  series: int = None) -> Dict:
        """
        Calculate metrics for different strategy types
        
        Args:
            strategy_type: Type of strategy ('5s10s', 'compression', etc.)
            series: Series number to use (if None, uses current)
            
        Returns:
            Dictionary with strategy metrics
        """
        # Use current series if not specified
        if series is None:
            today = datetime.now()
            ref_date = datetime(2025, 3, 20)
            ref_series = 43
            months_diff = (today.year - ref_date.year) * 12 + today.month - ref_date.month
            if today.day < 20:
                months_diff -= 1
            series = ref_series + (months_diff // 6)
        
        logger.info(f"Calculating {strategy_type} metrics for series {series}")
        
        if strategy_type == '5s10s':
            return self._calculate_5s10s_metrics(series)
        elif strategy_type == 'compression':
            return self._calculate_compression_metrics(series)
        elif strategy_type == '3s5s7s':
            return self._calculate_butterfly_metrics(series)
        else:
            logger.error(f"Unknown strategy type: {strategy_type}")
            return {}
    
    def _calculate_5s10s_metrics(self, series: int) -> Dict:
        """Calculate 5s10s steepener metrics"""
        # Get tickers
        ticker_5y = self.connector.get_index_ticker('EU', 'IG', series, '5Y')
        ticker_10y = self.connector.get_index_ticker('EU', 'IG', series, '10Y')
        
        # Get spreads
        spread_5y = self.connector.get_current_spread(ticker_5y)
        spread_10y = self.connector.get_current_spread(ticker_10y)
        
        if spread_5y is None or spread_5y.empty or spread_10y is None or spread_10y.empty:
            return {}
        
        # Get DV01s
        dv01_5y = self.connector.get_dv01(ticker_5y)
        dv01_10y = self.connector.get_dv01(ticker_10y)
        
        # Calculate metrics
        spread_5y_val = spread_5y['px_last'].iloc[0]  # lowercase
        spread_10y_val = spread_10y['px_last'].iloc[0]  # lowercase
        slope = spread_10y_val - spread_5y_val
        
        # DV01-weighted slope
        if not np.isnan(dv01_5y) and not np.isnan(dv01_10y):
            dv01_ratio = dv01_10y / dv01_5y
            weighted_slope = slope * dv01_ratio
        else:
            weighted_slope = np.nan
        
        return {
            'series': series,
            'spread_5y': spread_5y_val,
            'spread_10y': spread_10y_val,
            'slope': slope,
            'dv01_5y': dv01_5y,
            'dv01_10y': dv01_10y,
            'dv01_ratio': dv01_10y / dv01_5y if not np.isnan(dv01_5y) else np.nan,
            'dv01_weighted_slope': weighted_slope,
            'timestamp': datetime.now()
        }
    
    def _calculate_compression_metrics(self, series: int) -> Dict:
        """Calculate compression trade metrics (Main vs Crossover)"""
        # Get tickers
        main_ticker = self.connector.get_index_ticker('EU', 'IG', series, '5Y')
        xo_ticker = self.connector.get_index_ticker('EU', 'XO', series, '5Y')
        
        # Get spreads
        main_spread = self.connector.get_current_spread(main_ticker)
        xo_spread = self.connector.get_current_spread(xo_ticker)
        
        if main_spread is None or main_spread.empty or xo_spread is None or xo_spread.empty:
            return {}
        
        # Calculate basis
        basis = self.connector.calculate_basis(xo_ticker, main_ticker)
        
        return {
            'series': series,
            'main_spread': main_spread['PX_LAST'].iloc[0],
            'xo_spread': xo_spread['PX_LAST'].iloc[0],
            'basis': basis,
            'ratio': xo_spread['PX_LAST'].iloc[0] / main_spread['PX_LAST'].iloc[0],
            'timestamp': datetime.now()
        }
    
    def _calculate_butterfly_metrics(self, series: int) -> Dict:
        """Calculate butterfly metrics (3s5s7s)"""
        # Get tickers
        ticker_3y = self.connector.get_index_ticker('EU', 'IG', series, '3Y')
        ticker_5y = self.connector.get_index_ticker('EU', 'IG', series, '5Y')
        ticker_7y = self.connector.get_index_ticker('EU', 'IG', series, '7Y')
        
        # Get spreads
        spread_3y = self.connector.get_current_spread(ticker_3y)
        spread_5y = self.connector.get_current_spread(ticker_5y)
        spread_7y = self.connector.get_current_spread(ticker_7y)
        
        if (spread_3y is None or spread_3y.empty or 
            spread_5y is None or spread_5y.empty or 
            spread_7y is None or spread_7y.empty):
            return {}
        
        # Calculate butterfly spread (2*5Y - 3Y - 7Y)
        butterfly = (2 * spread_5y['PX_LAST'].iloc[0] - 
                    spread_3y['PX_LAST'].iloc[0] - 
                    spread_7y['PX_LAST'].iloc[0])
        
        return {
            'series': series,
            'spread_3y': spread_3y['PX_LAST'].iloc[0],
            'spread_5y': spread_5y['PX_LAST'].iloc[0],
            'spread_7y': spread_7y['PX_LAST'].iloc[0],
            'butterfly': butterfly,
            'timestamp': datetime.now()
        }
    
    def _save_spreads_to_db(self, 
                           spread_df: pd.DataFrame,
                           configs: List[Dict]):
        """Save spread data to database"""
        for i, config in enumerate(configs):
            if i >= len(spread_df):
                break
            
            row = spread_df.iloc[i]
            
            # Create index ID
            region = config['region']
            market = config['market']
            series = config['series']
            tenor = config['tenor']
            index_id = f"{region}_{market}_S{series}_{tenor}"
            
            # Create spread data object
            spread_data = CDSSpreadData(
                index_id=index_id,
                timestamp=datetime.now(),
                bid=row.get('PX_BID', np.nan),
                ask=row.get('PX_ASK', np.nan),
                mid=(row.get('PX_BID', 0) + row.get('PX_ASK', 0)) / 2,
                last=row.get('PX_LAST', np.nan),
                volume=row.get('VOLUME', np.nan),
                dv01=row.get('RISK_MID', np.nan),
                change_1d=row.get('CHG_NET_1D', np.nan),
                change_pct_1d=row.get('CHG_PCT_1D', np.nan)
            )
            
            try:
                self.db.save_spread_data(spread_data)
            except Exception as e:
                logger.error(f"Error saving spread data for {index_id}: {e}")
    
    def reconstruct_historical_pnl(self,
                                  strategy_name: str,
                                  start_date: datetime,
                                  end_date: datetime = None) -> pd.DataFrame:
        """
        Reconstruct historical P&L for a strategy
        
        Args:
            strategy_name: Name of strategy
            start_date: Start date for reconstruction
            end_date: End date (default: today)
            
        Returns:
            DataFrame with daily P&L
        """
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(f"Reconstructing P&L for {strategy_name} from {start_date} to {end_date}")
        
        # Get positions for the strategy
        positions = self.db.get_open_positions(strategy_name)
        
        if not positions:
            logger.warning(f"No positions found for strategy {strategy_name}")
            return pd.DataFrame()
        
        # This would involve:
        # 1. Getting historical spreads for each position
        # 2. Calculating daily P&L based on spread changes
        # 3. Adjusting for rolls and series changes
        # 4. Including funding costs
        
        # Placeholder for now - would need full implementation
        logger.info("Full P&L reconstruction to be implemented")
        return pd.DataFrame()
    
    def close(self):
        """Close database connection"""
        self.db.close()
        logger.info("Data manager closed")