"""
Position and Strategy models
Trade representation and P&L calculations
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np
from .enums import Side


@dataclass
class Position:
    """Represents a position in a CDS index"""
    index_id: str
    side: Side
    notional: float
    entry_date: datetime
    entry_spread: float
    entry_dv01: float
    current_spread: Optional[float] = None
    current_dv01: Optional[float] = None
    exit_date: Optional[datetime] = None
    exit_spread: Optional[float] = None
    exit_dv01: Optional[float] = None
    
    @property
    def is_open(self) -> bool:
        """Check if position is still open"""
        return self.exit_date is None
    
    @property
    def dv01_exposure(self) -> float:
        """Current DV01 exposure (signed)"""
        dv01 = self.current_dv01 or self.entry_dv01
        return self.side.value * self.notional * dv01 / 10000
    
    @property
    def pnl_bps(self) -> float:
        """P&L in basis points"""
        if not self.is_open and self.exit_spread is not None:
            # Closed position
            return self.side.value * (self.entry_spread - self.exit_spread)
        elif self.current_spread is not None:
            # Open position with current mark
            return self.side.value * (self.entry_spread - self.current_spread)
        return 0.0
    
    @property
    def pnl_dollars(self) -> float:
        """P&L in dollars"""
        if not self.is_open and self.exit_dv01 is not None:
            # Use exit DV01 for closed positions
            dv01 = self.exit_dv01
        else:
            # Use current or entry DV01
            dv01 = self.current_dv01 or self.entry_dv01
        
        return self.pnl_bps * dv01 * self.notional / 10000
    
    @property
    def days_held(self) -> int:
        """Number of days position has been held"""
        end_date = self.exit_date or datetime.now()
        return (end_date - self.entry_date).days
    
    def mark_to_market(self, current_spread: float, current_dv01: float = None):
        """Update position with current market data"""
        self.current_spread = current_spread
        if current_dv01 is not None:
            self.current_dv01 = current_dv01
    
    def close(self, exit_date: datetime, exit_spread: float, exit_dv01: float = None):
        """Close the position"""
        self.exit_date = exit_date
        self.exit_spread = exit_spread
        if exit_dv01 is not None:
            self.exit_dv01 = exit_dv01
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            'index_id': self.index_id,
            'side': self.side.value,
            'notional': self.notional,
            'entry_date': self.entry_date.isoformat(),
            'entry_spread': self.entry_spread,
            'entry_dv01': self.entry_dv01,
            'current_spread': self.current_spread,
            'current_dv01': self.current_dv01,
            'exit_date': self.exit_date.isoformat() if self.exit_date else None,
            'exit_spread': self.exit_spread,
            'exit_dv01': self.exit_dv01,
            'is_open': self.is_open,
            'pnl_bps': self.pnl_bps,
            'pnl_dollars': self.pnl_dollars,
            'days_held': self.days_held
        }


@dataclass
class Strategy:
    """Represents a trading strategy with multiple positions"""
    name: str
    strategy_type: str  # '5s10s', 'compression', 'butterfly', etc.
    positions: List[Position]
    creation_date: datetime
    target_dv01: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Check if strategy has any open positions"""
        return any(p.is_open for p in self.positions)
    
    @property
    def net_dv01(self) -> float:
        """Net DV01 exposure across all positions"""
        return sum(p.dv01_exposure for p in self.positions)
    
    @property
    def total_pnl(self) -> float:
        """Total P&L in dollars"""
        return sum(p.pnl_dollars for p in self.positions)
    
    @property
    def total_pnl_bps(self) -> float:
        """Weighted average P&L in basis points"""
        total_weight = sum(abs(p.dv01_exposure) for p in self.positions)
        if total_weight == 0:
            return 0
        
        weighted_pnl = sum(p.pnl_bps * abs(p.dv01_exposure) for p in self.positions)
        return weighted_pnl / total_weight
    
    def is_dv01_neutral(self, tolerance: float = 100) -> bool:
        """Check if strategy is DV01 neutral within tolerance"""
        return abs(self.net_dv01) < tolerance
    
    def check_stop_loss(self) -> bool:
        """Check if stop loss has been hit"""
        if self.stop_loss is None:
            return False
        return self.total_pnl <= -abs(self.stop_loss)
    
    def check_take_profit(self) -> bool:
        """Check if take profit has been hit"""
        if self.take_profit is None:
            return False
        return self.total_pnl >= abs(self.take_profit)
    
    def add_position(self, position: Position):
        """Add a new position to the strategy"""
        self.positions.append(position)
    
    def close_all_positions(self, exit_date: datetime, exit_spreads: Dict[str, float], 
                           exit_dv01s: Dict[str, float] = None):
        """Close all open positions in the strategy"""
        for position in self.positions:
            if position.is_open and position.index_id in exit_spreads:
                exit_dv01 = exit_dv01s.get(position.index_id) if exit_dv01s else None
                position.close(exit_date, exit_spreads[position.index_id], exit_dv01)
    
    def get_position_summary(self) -> Dict:
        """Get summary of all positions"""
        summary = {
            'total_positions': len(self.positions),
            'open_positions': sum(1 for p in self.positions if p.is_open),
            'closed_positions': sum(1 for p in self.positions if not p.is_open),
            'long_positions': sum(1 for p in self.positions if p.side == Side.BUY),
            'short_positions': sum(1 for p in self.positions if p.side == Side.SELL),
            'total_notional': sum(p.notional for p in self.positions),
            'net_dv01': self.net_dv01,
            'total_pnl': self.total_pnl,
            'is_dv01_neutral': self.is_dv01_neutral()
        }
        return summary
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            'name': self.name,
            'strategy_type': self.strategy_type,
            'creation_date': self.creation_date.isoformat(),
            'target_dv01': self.target_dv01,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'metadata': self.metadata,
            'is_active': self.is_active,
            'net_dv01': self.net_dv01,
            'total_pnl': self.total_pnl,
            'position_count': len(self.positions),
            'positions': [p.to_dict() for p in self.positions]
        }