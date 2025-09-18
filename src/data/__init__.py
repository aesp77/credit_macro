"""
Data access layer for CDS Monitor
"""

from .bloomberg_connector import BloombergCDSConnector
from .data_manager import CDSDataManager
from .cache import DataCache, SpreadCache, ReferenceDataCache, CachedFunction

__all__ = [
    'BloombergCDSConnector',
    'CDSDataManager',
    'DataCache',
    'SpreadCache',
    'ReferenceDataCache',
    'CachedFunction'
]

__version__ = '1.0.0'
