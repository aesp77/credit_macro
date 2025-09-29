# src/models/ml_predictions.py
"""
Supervised learning for spread direction prediction
"""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
from typing import Dict  

class SpreadPredictor:
    """Predict spread movements"""
    
    def __init__(self, horizon_days: int = 5, threshold_bps: float = 5):
        self.horizon_days = horizon_days
        self.threshold_bps = threshold_bps
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=20,
            random_state=42
        )
        
    def create_labels(self, spreads: pd.Series) -> pd.Series:
        """Create directional labels"""
        future_change = spreads.shift(-self.horizon_days) - spreads
        
        labels = pd.Series(index=spreads.index, dtype='object')
        labels[future_change > self.threshold_bps] = 1  # Widen
        labels[future_change < -self.threshold_bps] = -1  # Tighten  
        labels[abs(future_change) <= self.threshold_bps] = 0  # Flat
        
        return labels
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train model with time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.model.fit(X_train, y_train)
            score = self.model.score(X_val, y_val)
            scores.append(score)
        
        # Final fit on all data
        self.model.fit(X, y)
        
        return {
            'cv_score_mean': np.mean(scores),
            'cv_score_std': np.std(scores),
            'feature_importance': pd.Series(
                self.model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
        }
    
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get prediction probabilities"""
        probs = self.model.predict_proba(X)
        return pd.DataFrame(
            probs,
            index=X.index,
            columns=['prob_tighten', 'prob_flat', 'prob_widen']
        )