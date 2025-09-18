"""
CDS models package
Core data structures for CDS monitoring and trading
"""

from .enums import Region, Market, Tenor, Side
from .cds_index import CDSIndex, CDSIndexDefinition
from .spread_data import CDSSpreadData
from .curve import CDSCurve
from .position import Position, Strategy
from .database import CDSDatabase

__all__ = [
    'Region', 'Market', 'Tenor', 'Side',
    'CDSIndex', 'CDSIndexDefinition',
    'CDSSpreadData', 'CDSCurve',
    'Position', 'Strategy',
    'CDSDatabase'
]

__version__ = '1.0.0'
