"""
Spread data models
Point-in-time market data for CDS indices
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class CDSSpreadData:
    """Point-in-time spread data for a CDS index"""
    index_id: str
    timestamp: datetime
    bid: float
    ask: float
    mid: float
    last: float
    volume: Optional[float] = None
    dv01: Optional[float] = None
    change_1d: Optional[float] = None
    change_pct_1d: Optional[float] = None
    
    @property
    def bid_ask_spread(self) -> float:
        """Calculate bid-ask spread in basis points"""
        return self.ask - self.bid
    
    @property
    def mid_last_diff(self) -> float:
        """Difference between mid and last price"""
        return abs(self.mid - self.last)
    
    @property
    def is_stale(self, hours: int = 24) -> bool:
        """Check if data is older than specified hours"""
        age = datetime.now() - self.timestamp
        return age.total_seconds() > hours * 3600
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage/serialization"""
        return {
            'index_id': self.index_id,
            'timestamp': self.timestamp.isoformat(),
            'bid': self.bid,
            'ask': self.ask,
            'mid': self.mid,
            'last': self.last,
            'volume': self.volume,
            'dv01': self.dv01,
            'change_1d': self.change_1d,
            'change_pct_1d': self.change_pct_1d,
            'bid_ask_spread': self.bid_ask_spread
        }