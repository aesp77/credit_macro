"""
Bloomberg API connector for CDS data
Main interface for all Bloomberg data retrieval
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from xbbg import blp

from ..models.cds_index import CDSIndex
from ..models.enums import Region, Market, Tenor
from .cache import DataCache, SpreadCache, ReferenceDataCache

logger = logging.getLogger(__name__)


class BloombergCDSConnector:
    """
    Main connector class for Bloomberg CDS data
    Handles all data retrieval and caching for CDS indices
    """
    
    # Index naming patterns for different regions/markets
    INDEX_PATTERNS = {
        'EU_IG': 'ITRX EUR CDSI S{series} {tenor}',
        'EU_XO': 'ITRX EUR XOVER S{series} {tenor}',
        'US_IG': 'CDX IG {series} {tenor}',
        'US_HY': 'CDX HY {series} {tenor}',
        'EU_SNR_FIN': 'ITRX EUR SNRFIN S{series} {tenor}',
        'EU_SUB_FIN': 'ITRX EUR SUBFIN S{series} {tenor}',
        'ASIA_IG': 'ITRX ASIA IG S{series} {tenor}',
        'EM': 'CDX EM S{series} {tenor}'
    }
    
    # Standard tenors
    TENORS = ['3Y', '5Y', '7Y', '10Y']
    
    def __init__(self, 
                 spread_cache_expiry: int = 1,
                 reference_cache_expiry: int = 1440):
        """
        Initialize the Bloomberg connector with caching
        
        Args:
            spread_cache_expiry: Minutes to cache spread data (default: 1)
            reference_cache_expiry: Minutes to cache reference data (default: 24 hours)
        """
        self.spread_cache = SpreadCache(spread_cache_expiry)
        self.reference_cache = ReferenceDataCache(reference_cache_expiry)
        logger.info("Bloomberg CDS Connector initialized")
    
    def get_index_ticker(self, 
                        region: Union[str, Region], 
                        market: Union[str, Market], 
                        series: int, 
                        tenor: Union[str, Tenor]) -> str:
        """
        Generate the Bloomberg ticker for a CDS index
        
        Args:
            region: Geographic region (EU, US, ASIA, EM)
            market: Market segment (IG, HY, XO, etc.)
            series: Series number (e.g., 41)
            tenor: Tenor (e.g., '5Y' or Tenor.Y5)
            
        Returns:
            Bloomberg ticker string
            
        Raises:
            ValueError: If unknown index type
        """
        # Convert enums to strings if needed
        if isinstance(region, Region):
            region = region.value
        if isinstance(market, Market):
            market = market.value
        if isinstance(tenor, Tenor):
            tenor = tenor.value
        
        key = f"{region}_{market}"
        if key not in self.INDEX_PATTERNS:
            raise ValueError(f"Unknown index type: {key}")
        
        pattern = self.INDEX_PATTERNS[key]
        return pattern.format(series=series, tenor=tenor)
    
    @lru_cache(maxsize=128)
    def get_index_members(self, ticker: str) -> pd.DataFrame:
        """
        Get all members of a CDS index (cached)
        
        Args:
            ticker: Bloomberg ticker for the index
            
        Returns:
            DataFrame with index constituents
        """
        cache_key = f"members_{ticker}"
        cached = self.reference_cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            bbg_ticker = f"{ticker} Corp" if not ticker.endswith('Corp') else ticker
            members = blp.bds(bbg_ticker, flds="INDX_MEMBERS")
            
            self.reference_cache.set(cache_key, members)
            logger.info(f"Retrieved {len(members)} members for {ticker}")
            return members
            
        except Exception as e:
            logger.error(f"Error retrieving members for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_index_details(self, ticker: str) -> Dict:
        """
        Get detailed information about a CDS index
        
        Args:
            ticker: Bloomberg ticker
            
        Returns:
            Dictionary with index details
        """
        cache_key = f"details_{ticker}"
        cached = self.reference_cache.get(cache_key)
        if cached is not None:
            return cached
        
        bbg_ticker = f"{ticker} Corp" if not ticker.endswith('Corp') else ticker
        
        fields = [
            "CDS_FIRST_ACCRUAL_START_DATE",
            "MATURITY",
            "RED_CODE",
            "RECOVERY_RATE",
            "QUOTED_SPREAD_CONVENTION",
            "CPN",
            "NOTL_FACE",
            "DAY_CNT_DES",
            "ISSUER",
            "SERIES"
        ]
        
        try:
            details = blp.bdp(bbg_ticker, flds=fields)
            result = details.to_dict('records')[0] if not details.empty else {}
            
            self.reference_cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving details for {ticker}: {e}")
            return {}
    
    def get_current_spread(self, tickers: Union[str, List[str]]) -> pd.DataFrame:
        """
        Get current spreads for one or more CDS indices
        
        Args:
            tickers: Single ticker or list of tickers
            
        Returns:
            DataFrame with current spreads and related data
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Check cache for each ticker
        cached_data = []
        tickers_to_fetch = []
        
        for ticker in tickers:
            cache_key = f"spread_{ticker}"
            cached = self.spread_cache.get(cache_key)
            if cached is not None:
                cached_data.append(cached)
            else:
                tickers_to_fetch.append(ticker)
        
        # Fetch non-cached data
        if tickers_to_fetch:
            bbg_tickers = [f"{t} CBIN INDEX" if not t.endswith('INDEX') else t 
                          for t in tickers_to_fetch]
            
            try:
                fields = [
                    "PX_LAST", 
                    "PX_BID", 
                    "PX_ASK",
                    "CHG_NET_1D", 
                    "CHG_PCT_1D",
                    "VOLUME",
                    "TIME",
                    "RISK_MID"  # DV01
                ]
                
                data = blp.bdp(bbg_tickers, flds=fields)
                
                # Cache the results
                for i, ticker in enumerate(tickers_to_fetch):
                    ticker_data = data.iloc[[i]] if len(data) > i else pd.DataFrame()
                    if not ticker_data.empty:
                        self.spread_cache.set(f"spread_{ticker}", ticker_data)
                        cached_data.append(ticker_data)
                
                logger.debug(f"Fetched spreads for {len(tickers_to_fetch)} tickers")
                
            except Exception as e:
                logger.error(f"Error fetching spreads: {e}")
                return pd.DataFrame()
        
        return pd.concat(cached_data) if cached_data else pd.DataFrame()
    
    def get_historical_spreads(self, 
                              ticker: str, 
                              start_date: Union[str, datetime],
                              end_date: Union[str, datetime] = None,
                              fields: List[str] = None) -> pd.DataFrame:
        """
        Get historical spread data for a CDS index
        
        Args:
            ticker: Bloomberg ticker
            start_date: Start date for historical data
            end_date: End date (default: today)
            fields: Fields to retrieve (default: px_last)
            
        Returns:
            DataFrame with historical data
        """
        if fields is None:
            fields = ['px_last']
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Convert datetime to string if needed
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        bbg_ticker = f"{ticker} CBIN INDEX" if not ticker.endswith('INDEX') else ticker
        
        try:
            data = blp.bdh(bbg_ticker, flds=fields, 
                          start_date=start_date, end_date=end_date)
            
            # Flatten multi-level columns if needed
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(-1)
            
            logger.info(f"Retrieved {len(data)} historical points for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_dv01(self, ticker: str, notional: float = 10_000_000) -> float:
        """
        Get DV01 (dollar value of 01) for a CDS index
        
        Args:
            ticker: Bloomberg ticker
            notional: Desired notional amount (default: $10M)
            
        Returns:
            DV01 value scaled to the requested notional
        """
        # Add proper caching
        cache_key = f"dv01_{ticker}_{notional}"
        if hasattr(self, '_cache') and cache_key in self._cache:
            cached_time = self._cache_timestamps.get(cache_key, datetime.min)
            if datetime.now() - cached_time < self.cache_expiry:
                return self._cache[cache_key]
        
        bbg_ticker = f"{ticker} CBIN INDEX" if not ticker.endswith('INDEX') else ticker
        
        try:
            # Try multiple Bloomberg fields for DV01
            dv01_fields = ["SW_EQV_BPV", "RISK_MID", "DV01", "CREDIT_DV01", "DOLLAR_DUR"]
            
            for field in dv01_fields:
                try:
                    data = blp.bdp(bbg_ticker, flds=field)
                    
                    if not data.empty and len(data.columns) > 0:
                        # Get the first column (field name might vary)
                        col_name = data.columns[0]
                        dv01_bloomberg = data[col_name].iloc[0]
                        
                        # Check if we got a valid number
                        if not pd.isna(dv01_bloomberg) and dv01_bloomberg != 0:
                            # Bloomberg SW_EQV_BPV is typically for $10M notional
                            bloomberg_notional = 10_000_000
                            
                            # Scale to desired notional
                            dv01_scaled = abs(dv01_bloomberg) * (notional / bloomberg_notional)
                            
                            # Cache the result
                            if hasattr(self, '_cache'):
                                self._cache[cache_key] = dv01_scaled
                                self._cache_timestamps[cache_key] = datetime.now()
                            
                            logger.info(f"DV01 for {ticker}: ${dv01_scaled:,.0f} (Bloomberg: ${dv01_bloomberg:,.0f}, field: {field})")
                            return float(dv01_scaled)
                            
                except Exception as field_error:
                    logger.debug(f"Field {field} failed for {ticker}: {field_error}")
                    continue
            
            # If all Bloomberg fields fail, calculate approximation
            logger.warning(f"No DV01 data available for {ticker}, calculating approximation")
            return self._approximate_dv01(ticker, notional)
            
        except Exception as e:
            logger.error(f"Error fetching DV01 for {ticker}: {e}")
            return self._approximate_dv01(ticker, notional)

    def _approximate_dv01(self, ticker: str, notional: float = 10_000_000) -> float:
        """
        Approximate DV01 when Bloomberg data is unavailable
        
        Args:
            ticker: Bloomberg ticker
            
        Returns:
            Approximated DV01 value
        """
        try:
            # Get current spread
            spread_data = self.get_current_spread(ticker)
            if spread_data.empty:
                return np.nan
            
            # Extract tenor from ticker
            tenor_str = None
            for t in ['3Y', '5Y', '7Y', '10Y']:
                if t in ticker:
                    tenor_str = t
                    break
            
            if not tenor_str:
                return np.nan
            
            # Get spread value (handle both uppercase and lowercase)
            spread_col = 'px_last' if 'px_last' in spread_data.columns else 'PX_LAST'
            if spread_col not in spread_data.columns:
                return np.nan
                
            spread = spread_data[spread_col].iloc[0]
            years = float(tenor_str.replace('Y', ''))
            
            # Approximate DV01: spread * notional * modified_duration / 10000
            # Modified duration ≈ years * 0.9 (rough approximation)
            mod_duration = years * 0.9
            approx_dv01 = spread * notional * mod_duration * 0.0001  # 1bp = 0.0001
            
            logger.info(f"Approximated DV01 for {ticker}: ${approx_dv01:,.0f}")
            return approx_dv01
            
        except Exception as e:
            logger.error(f"Error approximating DV01 for {ticker}: {e}")
            return np.nan
    
    def get_curve_data(self, 
                      base_ticker: str, 
                      series: int,
                      tenors: List[str] = None) -> pd.DataFrame:
        """
        Get full curve data for a CDS index across tenors
        
        Args:
            base_ticker: Base ticker (e.g., 'ITRX EUR CDSI')
            series: Series number
            tenors: List of tenors (default: all standard tenors)
            
        Returns:
            DataFrame with curve data including spreads and DV01s
        """
        if tenors is None:
            tenors = self.TENORS
        
        # Generate tickers for all tenors
        tickers = [f"{base_ticker} S{series} {tenor}" for tenor in tenors]
        
        # Get spreads for all tenors
        spreads = self.get_current_spread(tickers)
        
        if not spreads.empty:
            spreads['tenor'] = tenors[:len(spreads)]
            spreads['series'] = series
            spreads['base_ticker'] = base_ticker
            
            # Add DV01 for each tenor
            dv01s = []
            for ticker in tickers:
                dv01s.append(self.get_dv01(ticker))
            spreads['dv01'] = dv01s[:len(spreads)]
        
        return spreads
    
    def calculate_basis(self, 
                       ticker1: str, 
                       ticker2: str,
                       notional_adjustment: bool = True) -> float:
        """
        Calculate basis between two CDS indices
        
        Args:
            ticker1: First ticker
            ticker2: Second ticker  
            notional_adjustment: Whether to adjust for DV01
            
        Returns:
            Basis in bps (ticker1 - ticker2)
        """
        spread1 = self.get_current_spread(ticker1)
        spread2 = self.get_current_spread(ticker2)
        
        if spread1.empty or spread2.empty:
            return np.nan
        
        basis = spread1['PX_LAST'].iloc[0] - spread2['PX_LAST'].iloc[0]
        
        if notional_adjustment:
            dv01_1 = self.get_dv01(ticker1)
            dv01_2 = self.get_dv01(ticker2)
            
            if not np.isnan(dv01_1) and not np.isnan(dv01_2):
                # Adjust basis for DV01 ratio
                basis = basis * (dv01_1 / dv01_2)
        
        return basis
    
    def batch_get_spreads(self, 
                         index_configs: List[Dict],
                         parallel: bool = True,
                         max_workers: int = 5) -> pd.DataFrame:
        """
        Get spreads for multiple indices in batch
        
        Args:
            index_configs: List of dicts with region, market, series, tenor
            parallel: Whether to fetch in parallel
            max_workers: Maximum parallel workers
            
        Returns:
            DataFrame with all spreads
        """
        # Generate tickers from configs
        tickers = []
        for config in index_configs:
            try:
                ticker = self.get_index_ticker(**config)
                tickers.append(ticker)
            except ValueError as e:
                logger.warning(f"Skipping invalid config {config}: {e}")
        
        if not tickers:
            return pd.DataFrame