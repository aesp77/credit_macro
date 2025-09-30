"""
Market regime detection using unsupervised learning
"""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

class RegimeDetector:
    """Identify market regimes from CDS data"""
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        self.regime_labels = {}  # Will be assigned based on characteristics
        
    def fit_predict(self, features_df: pd.DataFrame) -> pd.Series:
        """Fit model and predict regimes"""
        # Scale features
        features_scaled = self.scaler.fit_transform(features_df.fillna(0))
        
        # Reduce dimensions
        features_pca = self.pca.fit_transform(features_scaled)
        
        # Cluster
        clusters = self.kmeans.fit_predict(features_pca)
        
        # Assign labels based on actual characteristics
        self._assign_regime_labels(features_df, clusters)
        
        # Map clusters to meaningful labels
        regime_series = pd.Series(clusters, index=features_df.index)
        regime_series = regime_series.map(self.regime_labels)
        
        return regime_series
    
    def _assign_regime_labels(self, features_df: pd.DataFrame, clusters: np.ndarray):
        """Assign regime labels based on cluster characteristics"""
        # Calculate mean spread and volatility for each cluster
        cluster_stats = {}
        
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            cluster_data = features_df[mask]
            
            avg_spread = cluster_data['spread_level'].mean() if 'spread_level' in cluster_data else 0
            avg_vol = cluster_data['vol_20d'].mean() if 'vol_20d' in cluster_data else 0
            
            cluster_stats[cluster_id] = {
                'spread': avg_spread,
                'volatility': avg_vol
            }
        
        # Sort clusters by spread level
        sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1]['spread'])
        
        # Assign labels based on spread levels and volatility
        # Lower spreads = Risk-On (bull market)
        # Higher spreads = Risk-Off/Stressed (bear market)
        
        if self.n_regimes == 4:
            self.regime_labels[sorted_clusters[0][0]] = 'Risk-On'  # Lowest spreads
            self.regime_labels[sorted_clusters[1][0]] = 'Transition-Bull'  # Low-medium spreads
            self.regime_labels[sorted_clusters[2][0]] = 'Transition-Bear'  # Medium-high spreads
            self.regime_labels[sorted_clusters[3][0]] = 'Risk-Off'  # Highest spreads
        elif self.n_regimes == 3:
            self.regime_labels[sorted_clusters[0][0]] = 'Risk-On'
            self.regime_labels[sorted_clusters[1][0]] = 'Transition'
            self.regime_labels[sorted_clusters[2][0]] = 'Risk-Off'
        else:
            # Generic labeling for other numbers
            for i, (cluster_id, _) in enumerate(sorted_clusters):
                self.regime_labels[cluster_id] = f'Regime_{i+1}_Low_to_High'
    
    def get_regime_statistics(self, features_df: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
        """Calculate statistics for each regime"""
        stats = []
        
        # Ensure both have the same index
        common_index = features_df.index.intersection(regimes.index)
        features_aligned = features_df.loc[common_index]
        regimes_aligned = regimes.loc[common_index]
        
        for regime in regimes_aligned.unique():
            mask = regimes_aligned == regime
            regime_data = features_aligned[mask]
            
            # Calculate if this is bullish or bearish
            avg_spread = regime_data['spread_level'].mean() if 'spread_level' in regime_data.columns else np.nan
            
            stats.append({
                'regime': regime,
                'frequency': mask.sum() / len(mask),
                'avg_spread': avg_spread,
                'avg_volatility': regime_data['vol_20d'].mean() if 'vol_20d' in regime_data.columns else np.nan,
                'duration_days': mask.sum(),
                'market_condition': 'Bullish' if avg_spread < 60 else 'Bearish' if avg_spread > 80 else 'Neutral'
            })
        
        return pd.DataFrame(stats).sort_values('avg_spread')
        
    
    """
Enhanced Market regime detection with dynamic thresholds and momentum
"""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

class DynamicRegimeDetector:
    """Identify market regimes using momentum and dynamic thresholds"""
    
    def __init__(self):
        self.lookback_peak = 60  # Days to look back for peaks/troughs
        self.momentum_threshold = 0.15  # 15% move threshold for regime change
        
    def fit_predict(self, features_df: pd.DataFrame) -> pd.Series:
        """Detect regimes based on dynamic thresholds and momentum"""
        
        spread = features_df['spread_level'].copy()
        
        # Calculate rolling metrics
        spread_ma20 = spread.rolling(20).mean()
        spread_ma60 = spread.rolling(60).mean()
        
        # Calculate rolling peaks and troughs
        rolling_max = spread.rolling(self.lookback_peak).max()
        rolling_min = spread.rolling(self.lookback_peak).min()
        
        # Calculate distance from recent peak/trough
        pct_from_peak = (spread - rolling_max) / rolling_max
        pct_from_trough = (spread - rolling_min) / rolling_min
        
        # Calculate momentum
        momentum_5d = spread.pct_change(5)
        momentum_20d = spread.pct_change(20)
        
        # Initialize regime series
        regimes = pd.Series(index=spread.index, dtype='object')
        
        for i in range(len(spread)):
            if i < self.lookback_peak:
                # Not enough history - use simple thresholds
                if spread.iloc[i] < spread.quantile(0.33):
                    regimes.iloc[i] = 'Risk-On'
                elif spread.iloc[i] > spread.quantile(0.67):
                    regimes.iloc[i] = 'Risk-Off'
                else:
                    regimes.iloc[i] = 'Transition-Bull'
                continue
            
            current_spread = spread.iloc[i]
            current_ma20 = spread_ma20.iloc[i] if not pd.isna(spread_ma20.iloc[i]) else current_spread
            current_ma60 = spread_ma60.iloc[i] if not pd.isna(spread_ma60.iloc[i]) else current_spread
            
            # Dynamic regime detection logic
            
            # 1. Check if we're near a recent peak (within 10% of 60-day high)
            if pct_from_peak.iloc[i] > -0.10:
                # Near peak - bearish bias
                if momentum_20d.iloc[i] < -0.10:  # Sharp decline
                    regimes.iloc[i] = 'Risk-Off'
                elif momentum_20d.iloc[i] < 0:  # Gradual decline
                    regimes.iloc[i] = 'Transition-Bear'
                else:  # Still rising near peaks
                    regimes.iloc[i] = 'Risk-Off'  # Toppy market
                    
            # 2. Check if we're near a recent trough (within 10% of 60-day low)
            elif pct_from_trough.iloc[i] < 0.10:
                # Near trough - bullish bias
                if momentum_20d.iloc[i] > 0.10:  # Sharp rally
                    regimes.iloc[i] = 'Risk-On'
                elif momentum_20d.iloc[i] > 0:  # Gradual recovery
                    regimes.iloc[i] = 'Transition-Bull'
                else:  # Still falling near troughs
                    regimes.iloc[i] = 'Risk-On'  # Bottoming process
                    
            # 3. Middle ground - use trend and momentum
            else:
                # Check trend direction
                trending_up = current_spread < current_ma20 < current_ma60
                trending_down = current_spread > current_ma20 > current_ma60
                
                if trending_up:
                    if momentum_5d.iloc[i] > 0.05:  # Strong bullish momentum
                        regimes.iloc[i] = 'Risk-On'
                    else:
                        regimes.iloc[i] = 'Transition-Bull'
                        
                elif trending_down:
                    if momentum_5d.iloc[i] < -0.05:  # Strong bearish momentum
                        regimes.iloc[i] = 'Risk-Off'
                    else:
                        regimes.iloc[i] = 'Transition-Bear'
                        
                else:  # Sideways/mixed signals
                    # Use absolute level as tiebreaker
                    if current_spread < spread.quantile(0.40):
                        regimes.iloc[i] = 'Transition-Bull'
                    elif current_spread > spread.quantile(0.60):
                        regimes.iloc[i] = 'Transition-Bear'
                    else:
                        # Look at very short term momentum
                        if momentum_5d.iloc[i] > 0:
                            regimes.iloc[i] = 'Transition-Bull'
                        else:
                            regimes.iloc[i] = 'Transition-Bear'
        
        return regimes
    
    def get_regime_statistics(self, features_df: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
        """Calculate statistics for each regime"""
        stats = []
        for regime in regimes.unique():
            if pd.notna(regime):
                mask = regimes == regime
                regime_data = features_df[mask]
                
                stats.append({
                    'regime': regime,
                    'frequency': mask.sum() / len(mask),
                    'avg_spread': regime_data['spread_level'].mean() if 'spread_level' in regime_data else np.nan,
                    'avg_volatility': regime_data['vol_20d'].mean() if 'vol_20d' in regime_data else np.nan,
                    'duration_days': mask.sum(),
                    'avg_momentum': features_df[mask]['spread_level'].pct_change(20).mean() * 100 if mask.sum() > 20 else 0
                })
        
        return pd.DataFrame(stats).sort_values('avg_spread')
    
    
    """
Advanced regime detector with peak/trough detection and momentum classification
"""
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

class AdvancedRegimeDetector:
    """Detect regimes using peak/trough analysis and momentum"""
    
    def __init__(self):
        self.peak_window = 60  # Days for peak/trough detection
        self.momentum_fast = 20  # Fast momentum period
        self.momentum_slow = 60  # Slow momentum period
        
    def detect_peaks_troughs(self, series: pd.Series, window: int = 60):
        """Identify significant peaks and troughs"""
        # Use local extrema detection
        peaks = argrelextrema(series.values, np.greater, order=window//2)[0]
        troughs = argrelextrema(series.values, np.less, order=window//2)[0]
        
        # Create a series marking peaks and troughs
        extrema = pd.Series(index=series.index, dtype='object')
        extrema.iloc[peaks] = 'peak'
        extrema.iloc[troughs] = 'trough'
        
        return extrema, peaks, troughs
    
    def fit_predict(self, features_df: pd.DataFrame) -> pd.Series:
        """Detect regimes based on peaks/troughs and momentum"""
        
        spread = features_df['spread_level'].copy()
        
        # Detect peaks and troughs
        extrema, peak_indices, trough_indices = self.detect_peaks_troughs(spread)
        
        # Calculate momentum indicators
        momentum_fast = spread.pct_change(self.momentum_fast)
        momentum_slow = spread.pct_change(self.momentum_slow)
        
        # Calculate rolling percentile rank (where we are in recent range)
        rolling_rank = spread.rolling(120).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5)
        
        # Initialize regime series
        regimes = pd.Series(index=spread.index, dtype='object')
        
        # Track last significant peak/trough
        last_peak_idx = None
        last_trough_idx = None
        last_peak_value = None
        last_trough_value = None
        
        for i in range(len(spread)):
            if i < 60:  # Initial period - use simple classification
                if spread.iloc[i] < spread.iloc[:i+1].quantile(0.40):
                    regimes.iloc[i] = 'Risk-On'
                elif spread.iloc[i] > spread.iloc[:i+1].quantile(0.60):
                    regimes.iloc[i] = 'Risk-Off'
                else:
                    regimes.iloc[i] = 'Transition-Bull'
                continue
            
            # Update last peak/trough tracking
            recent_peaks = [p for p in peak_indices if p <= i and i - p < 120]
            recent_troughs = [t for t in trough_indices if t <= i and i - t < 120]
            
            if recent_peaks:
                last_peak_idx = recent_peaks[-1]
                last_peak_value = spread.iloc[last_peak_idx]
            
            if recent_troughs:
                last_trough_idx = recent_troughs[-1]
                last_trough_value = spread.iloc[last_trough_idx]
            
            current_spread = spread.iloc[i]
            current_rank = rolling_rank.iloc[i] if not pd.isna(rolling_rank.iloc[i]) else 0.5
            fast_mom = momentum_fast.iloc[i] if not pd.isna(momentum_fast.iloc[i]) else 0
            slow_mom = momentum_slow.iloc[i] if not pd.isna(momentum_slow.iloc[i]) else 0
            
            # REGIME CLASSIFICATION LOGIC
            
            # 1. Check if we're in a crisis (sharp widening from low levels)
            if i > 0 and spread.iloc[i-1] < 60 and fast_mom > 0.30:  # 30%+ spike from low levels
                regimes.iloc[i] = 'Risk-Off'
                
            # 2. Check position relative to recent extrema
            elif last_peak_value and last_trough_value:
                # Calculate position between last peak and trough
                peak_trough_range = last_peak_value - last_trough_value
                
                if peak_trough_range > 0:
                    position_in_range = (current_spread - last_trough_value) / peak_trough_range
                    
                    # Are we closer to peak or trough?
                    if last_peak_idx and last_trough_idx:
                        # Which is more recent?
                        if last_peak_idx > last_trough_idx:
                            # Last peak is more recent - we're in decline from peak
                            if position_in_range > 0.7:  # Still near peak
                                if fast_mom < -0.05:  # Declining
                                    regimes.iloc[i] = 'Transition-Bear'
                                else:
                                    regimes.iloc[i] = 'Risk-Off'
                            elif position_in_range > 0.3:  # Middle range
                                if slow_mom < 0:
                                    regimes.iloc[i] = 'Transition-Bear'
                                else:
                                    regimes.iloc[i] = 'Transition-Bull'
                            else:  # Near trough
                                regimes.iloc[i] = 'Risk-On'
                        else:
                            # Last trough is more recent - we're in recovery from trough
                            if position_in_range < 0.3:  # Still near trough
                                if fast_mom > 0.05:  # Rising
                                    regimes.iloc[i] = 'Transition-Bull'
                                else:
                                    regimes.iloc[i] = 'Risk-On'
                            elif position_in_range < 0.7:  # Middle range
                                if slow_mom > 0:
                                    regimes.iloc[i] = 'Transition-Bull'
                                else:
                                    regimes.iloc[i] = 'Transition-Bear'
                            else:  # Near peak
                                regimes.iloc[i] = 'Risk-Off'
                
            # 3. Fallback to percentile rank and momentum
            else:
                if current_rank < 0.25:  # Low spreads (bullish)
                    if fast_mom > 0.10:
                        regimes.iloc[i] = 'Transition-Bear'  # Starting to widen
                    else:
                        regimes.iloc[i] = 'Risk-On'
                        
                elif current_rank > 0.75:  # High spreads (bearish)
                    if fast_mom < -0.10:
                        regimes.iloc[i] = 'Transition-Bull'  # Starting to tighten
                    else:
                        regimes.iloc[i] = 'Risk-Off'
                        
                else:  # Middle range - use momentum
                    if slow_mom > 0.05 and fast_mom > 0:
                        regimes.iloc[i] = 'Transition-Bear'
                    elif slow_mom < -0.05 and fast_mom < 0:
                        regimes.iloc[i] = 'Transition-Bull'
                    else:
                        # Neutral momentum - use level
                        if current_spread < 60:
                            regimes.iloc[i] = 'Risk-On'
                        elif current_spread > 80:
                            regimes.iloc[i] = 'Risk-Off'
                        else:
                            regimes.iloc[i] = 'Transition-Bull' if fast_mom < 0 else 'Transition-Bear'
        
        return regimes
    
    
    """
Smoothed regime detector WITH peak/trough detection
Combines the best of both approaches
"""
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

class SmoothedPeakTroughRegimeDetector:
    """Detect regimes using peak/trough analysis with smoothing to avoid noise"""
    
    def __init__(self):
        self.peak_window = 60  # Days for peak/trough detection
        self.momentum_fast = 20  # Fast momentum period
        self.momentum_slow = 60  # Slow momentum period
        self.min_regime_days = 10  # Minimum days before allowing regime change
        self.smoothing_window = 5  # Days for regime score smoothing
        
    def detect_peaks_troughs(self, series: pd.Series, window: int = 60):
        """Identify significant peaks and troughs"""
        peaks = argrelextrema(series.values, np.greater, order=window//2)[0]
        troughs = argrelextrema(series.values, np.less, order=window//2)[0]
        
        extrema = pd.Series(index=series.index, dtype='object')
        extrema.iloc[peaks] = 'peak'
        extrema.iloc[troughs] = 'trough'
        
        return extrema, peaks, troughs
    
    def fit_predict(self, features_df: pd.DataFrame) -> pd.Series:
        """Detect regimes based on peaks/troughs with smoothing"""
        
        spread = features_df['spread_level'].copy()
        
        # Detect peaks and troughs
        extrema, peak_indices, trough_indices = self.detect_peaks_troughs(spread)
        
        # Calculate momentum indicators
        momentum_fast = spread.pct_change(self.momentum_fast)
        momentum_slow = spread.pct_change(self.momentum_slow)
        
        # Calculate rolling percentile rank
        rolling_rank = spread.rolling(120).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
        )
        
        # Calculate regime scores (continuous values)
        regime_scores = pd.Series(index=spread.index, dtype=float)
        
        # Track last significant peak/trough
        last_peak_idx = None
        last_trough_idx = None
        last_peak_value = None
        last_trough_value = None
        
        for i in range(len(spread)):
            if i < 60:
                # Initial period - simple scoring
                if spread.iloc[i] < spread.iloc[:i+1].quantile(0.40):
                    regime_scores.iloc[i] = -1.5  # Risk-On
                elif spread.iloc[i] > spread.iloc[:i+1].quantile(0.60):
                    regime_scores.iloc[i] = 1.5  # Risk-Off
                else:
                    regime_scores.iloc[i] = 0  # Transition
                continue
            
            # Update last peak/trough tracking
            recent_peaks = [p for p in peak_indices if p <= i and i - p < 120]
            recent_troughs = [t for t in trough_indices if t <= i and i - t < 120]
            
            if recent_peaks:
                last_peak_idx = recent_peaks[-1]
                last_peak_value = spread.iloc[last_peak_idx]
            
            if recent_troughs:
                last_trough_idx = recent_troughs[-1]
                last_trough_value = spread.iloc[last_trough_idx]
            
            current_spread = spread.iloc[i]
            current_rank = rolling_rank.iloc[i] if not pd.isna(rolling_rank.iloc[i]) else 0.5
            fast_mom = momentum_fast.iloc[i] if not pd.isna(momentum_fast.iloc[i]) else 0
            slow_mom = momentum_slow.iloc[i] if not pd.isna(momentum_slow.iloc[i]) else 0
            
            # CALCULATE REGIME SCORE BASED ON PEAK/TROUGH LOGIC
            
            # Crisis detection
            if i > 0 and spread.iloc[i-1] < 60 and fast_mom > 0.30:
                regime_scores.iloc[i] = 2.0  # Strong Risk-Off
                
            # Peak/trough based scoring
            elif last_peak_value and last_trough_value:
                peak_trough_range = last_peak_value - last_trough_value
                
                if peak_trough_range > 0:
                    position_in_range = (current_spread - last_trough_value) / peak_trough_range
                    
                    if last_peak_idx and last_trough_idx:
                        if last_peak_idx > last_trough_idx:
                            # Coming down from peak
                            base_score = 0.5 + position_in_range  # 0.5 to 1.5
                            momentum_adjustment = -fast_mom * 2  # Tightening reduces score
                            regime_scores.iloc[i] = base_score + momentum_adjustment
                        else:
                            # Rising from trough
                            base_score = -0.5 + position_in_range  # -0.5 to 0.5
                            momentum_adjustment = fast_mom * 2  # Widening increases score
                            regime_scores.iloc[i] = base_score + momentum_adjustment
                    else:
                        # Fallback to position
                        regime_scores.iloc[i] = -1.5 + 3 * position_in_range
            
            # Percentile rank based scoring
            else:
                base_score = -2 + 4 * current_rank  # -2 to +2
                momentum_adjustment = fast_mom * 3 + slow_mom * 2
                regime_scores.iloc[i] = base_score + momentum_adjustment
        
        # SMOOTH THE SCORES
        smoothed_scores = regime_scores.rolling(self.smoothing_window, center=True).mean()
        smoothed_scores = smoothed_scores.fillna(regime_scores)  # Fill edges
        
        # Convert scores to regimes
        regimes = pd.Series(index=spread.index, dtype='object')
        
        for i in range(len(smoothed_scores)):
            score = smoothed_scores.iloc[i]
            if score < -1.0:
                regimes.iloc[i] = 'Risk-On'
            elif score < -0.2:
                regimes.iloc[i] = 'Transition-Bull'
            elif score < 0.2:
                regimes.iloc[i] = 'Transition'  # Neutral transition
            elif score < 1.0:
                regimes.iloc[i] = 'Transition-Bear'
            else:
                regimes.iloc[i] = 'Risk-Off'
        
        # Apply minimum duration filter
        regimes = self.enforce_minimum_duration(regimes, self.min_regime_days)
        
        # Merge adjacent similar regimes
        regimes = regimes.replace('Transition', pd.NA).fillna(method='ffill').fillna('Transition-Bull')
        
        return regimes
    
    def enforce_minimum_duration(self, regimes: pd.Series, min_days: int) -> pd.Series:
        """Ensure each regime lasts at least min_days"""
        result = regimes.copy()
        current_regime = result.iloc[0]
        regime_start = 0
        
        for i in range(1, len(result)):
            if result.iloc[i] != current_regime:
                # Check if current regime lasted long enough
                if i - regime_start < min_days:
                    # Too short, extend previous regime
                    for j in range(regime_start, i):
                        if regime_start > 0:
                            result.iloc[j] = result.iloc[regime_start - 1]
                else:
                    # Regime change is valid
                    current_regime = result.iloc[i]
                    regime_start = i
        
        # Handle last regime
        if len(result) - regime_start < min_days and regime_start > 0:
            for j in range(regime_start, len(result)):
                result.iloc[j] = result.iloc[regime_start - 1]
        
        return result