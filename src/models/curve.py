"""
CDS Curve models
Term structure and curve analytics
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime
import numpy as np
from .enums import Region, Market, Tenor


@dataclass
class CDSCurve:
    """Represents a full credit curve across tenors"""
    region: Region
    market: Market
    series: int
    observation_date: datetime
    spreads: Dict[Tenor, float]
    dv01s: Dict[Tenor, float] = field(default_factory=dict)
    
    @property
    def curve_id(self) -> str:
        """Unique identifier for this curve"""
        return f"{self.region.value}_{self.market.value}_S{self.series}_{self.observation_date.strftime('%Y%m%d')}"
    
    def interpolate_spread(self, years: float) -> float:
        """
        Interpolate spread for any maturity using linear interpolation
        
        Args:
            years: Time to maturity in years
            
        Returns:
            Interpolated spread in basis points
        """
        # Convert tenors to years
        tenor_years = {
            Tenor.Y1: 1, Tenor.Y3: 3, Tenor.Y5: 5,
            Tenor.Y7: 7, Tenor.Y10: 10, Tenor.Y20: 20, Tenor.Y30: 30
        }
        
        # Get available points
        points = [(tenor_years[t], s) for t, s in self.spreads.items() if t in tenor_years]
        points.sort()
        
        if not points:
            return np.nan
        
        # If years is outside range, use flat extrapolation
        if years <= points[0][0]:
            return points[0][1]
        if years >= points[-1][0]:
            return points[-1][1]
        
        # Linear interpolation
        x_vals, y_vals = zip(*points)
        return float(np.interp(years, x_vals, y_vals))
    
    def interpolate_dv01(self, years: float) -> float:
        """
        Interpolate DV01 for any maturity
        
        Args:
            years: Time to maturity in years
            
        Returns:
            Interpolated DV01
        """
        if not self.dv01s:
            return np.nan
        
        # Convert tenors to years
        tenor_years = {
            Tenor.Y1: 1, Tenor.Y3: 3, Tenor.Y5: 5,
            Tenor.Y7: 7, Tenor.Y10: 10, Tenor.Y20: 20, Tenor.Y30: 30
        }
        
        # Get available points
        points = [(tenor_years[t], d) for t, d in self.dv01s.items() if t in tenor_years]
        points.sort()
        
        if not points:
            return np.nan
        
        x_vals, y_vals = zip(*points)
        return float(np.interp(years, x_vals, y_vals))
    
    def calculate_rolldown(self, horizon_days: int = 90) -> Dict[Tenor, float]:
        """
        Calculate roll-down for each tenor
        
        Args:
            horizon_days: Number of days for roll calculation
            
        Returns:
            Dictionary of roll-down values by tenor (in basis points)
        """
        rolldown = {}
        years_to_roll = horizon_days / 365.25
        
        for tenor in self.spreads:
            tenor_years = tenor.years
            new_years = tenor_years - years_to_roll
            
            if new_years > 0:
                current_spread = self.spreads[tenor]
                rolled_spread = self.interpolate_spread(new_years)
                
                # Roll-down is the spread tightening from rolling down the curve
                # Positive roll-down means spread tightens (beneficial for long position)
                rolldown[tenor] = current_spread - rolled_spread
        
        return rolldown
    
    def calculate_carry(self, horizon_days: int = 90) -> Dict[Tenor, float]:
        """
        Calculate carry (spread income + roll-down) for each tenor
        
        Args:
            horizon_days: Number of days for carry calculation
            
        Returns:
            Dictionary of carry values by tenor (in basis points)
        """
        carry = {}
        rolldown = self.calculate_rolldown(horizon_days)
        
        for tenor in self.spreads:
            # Carry = spread income + roll-down
            spread_income = self.spreads[tenor] * (horizon_days / 365.25)
            roll = rolldown.get(tenor, 0)
            carry[tenor] = spread_income + roll
        
        return carry
    
    def get_slope(self, tenor1: Tenor, tenor2: Tenor) -> float:
        """
        Calculate slope between two tenors
        
        Args:
            tenor1: First tenor (shorter)
            tenor2: Second tenor (longer)
            
        Returns:
            Slope in basis points
        """
        if tenor1 not in self.spreads or tenor2 not in self.spreads:
            return np.nan
        
        return self.spreads[tenor2] - self.spreads[tenor1]
    
    def get_butterfly(self, short: Tenor, middle: Tenor, long: Tenor) -> float:
        """
        Calculate butterfly spread (2*middle - short - long)
        
        Args:
            short: Short tenor
            middle: Middle tenor
            long: Long tenor
            
        Returns:
            Butterfly spread in basis points
        """
        if any(t not in self.spreads for t in [short, middle, long]):
            return np.nan
        
        return 2 * self.spreads[middle] - self.spreads[short] - self.spreads[long]
    
    def to_dataframe(self):
        """Convert curve to pandas DataFrame for analysis"""
        import pandas as pd
        
        data = {
            'tenor': [t.value for t in self.spreads.keys()],
            'years': [t.years for t in self.spreads.keys()],
            'spread': list(self.spreads.values())
        }
        
        if self.dv01s:
            data['dv01'] = [self.dv01s.get(t, np.nan) for t in self.spreads.keys()]
        
        df = pd.DataFrame(data)
        df['observation_date'] = self.observation_date
        df['region'] = self.region.value
        df['market'] = self.market.value
        df['series'] = self.series
        
        return df