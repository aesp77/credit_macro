"""
Feature engineering for ML models
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class FeatureEngineer:
    """Generate features from CDS spread data"""
    
    def __init__(self, lookback_windows: List[int] = [5, 20, 60]):
        self.lookback_windows = lookback_windows
    
    def create_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spread-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Level features
        features['spread_level'] = df['spread_bps']
        features['spread_log'] = np.log(df['spread_bps'])
        
        # Change features
        for window in self.lookback_windows:
            features[f'return_{window}d'] = df['spread_bps'].pct_change(window)
            features[f'vol_{window}d'] = df['spread_bps'].rolling(window).std()
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(df['spread_bps'])
        features['z_score'] = self._calculate_zscore(df['spread_bps'])
        
        return features
    
    def create_curve_features(self, spreads_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create cross-index and curve features"""
        features = pd.DataFrame()
        
        # Basis trades
        if 'EU_XO_5Y' in spreads_dict and 'EU_IG_5Y' in spreads_dict:
            features['eu_basis'] = spreads_dict['EU_XO_5Y'] - spreads_dict['EU_IG_5Y']
        
        # Term structure
        if 'EU_IG_10Y' in spreads_dict and 'EU_IG_3Y' in spreads_dict:
            features['eu_ig_slope'] = spreads_dict['EU_IG_10Y'] - spreads_dict['EU_IG_3Y']
        
        return features
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_zscore(self, series: pd.Series, window: int = 60) -> pd.Series:
        """Calculate rolling z-score"""
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        zscore = (series - rolling_mean) / rolling_std
        return zscore