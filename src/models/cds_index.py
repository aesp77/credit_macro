"""
CDS Index model definitions
Core data structures for representing CDS indices
"""
from dataclasses import dataclass
from typing import Optional
from datetime import date
from .enums import Region, Market, Tenor


@dataclass
class CDSIndex:
    """Represents a CDS Index with Bloomberg identifiers"""
    ticker: str
    region: str  # Can be string or Region enum
    market: str  # Can be string or Market enum
    series: int
    tenor: str  # Can be string or Tenor enum
    full_ticker: str
    
    @property
    def bbg_ticker(self) -> str:
        """Returns the Bloomberg formatted ticker with Corp suffix"""
        return f"{self.full_ticker} Corp"
    
    @property
    def cbin_ticker(self) -> str:
        """Returns the CBIN INDEX ticker format"""
        return f"{self.full_ticker} CBIN INDEX"
    
    @property
    def unique_id(self) -> str:
        """Generate unique identifier for database storage"""
        return f"{self.region}_{self.market}_S{self.series}_{self.tenor}"


@dataclass
class CDSIndexDefinition:
    """Static definition of a CDS index with full metadata"""
    region: Region
    market: Market
    series: int
    tenor: Tenor
    ticker_base: str
    start_date: date
    maturity_date: date
    recovery_rate: float = 0.4
    coupon: Optional[float] = None
    notional: float = 1000000
    
    @property
    def ticker(self) -> str:
        """Generate the Bloomberg ticker based on region/market conventions"""
        if self.region == Region.EU:
            if self.market == Market.IG:
                return f"ITRX EUR CDSI S{self.series} {self.tenor.value}"
            elif self.market == Market.XO:
                return f"ITRX EUR XOVER S{self.series} {self.tenor.value}"
            elif self.market == Market.SNR_FIN:
                return f"ITRX EUR SNRFIN S{self.series} {self.tenor.value}"
            elif self.market == Market.SUB_FIN:
                return f"ITRX EUR SUBFIN S{self.series} {self.tenor.value}"
        elif self.region == Region.US:
            if self.market == Market.IG:
                return f"CDX IG {self.series} {self.tenor.value}"
            elif self.market == Market.HY:
                return f"CDX HY {self.series} {self.tenor.value}"
        
        # Fallback for other regions/markets
        return f"{self.ticker_base} S{self.series} {self.tenor.value}"
    
    @property
    def unique_id(self) -> str:
        """Generate unique identifier for database storage"""
        return f"{self.region.value}_{self.market.value}_S{self.series}_{self.tenor.value}"
    
    @property
    def bbg_ticker(self) -> str:
        """Bloomberg ticker with Corp suffix"""
        return f"{self.ticker} Corp"
    
    @property
    def cbin_ticker(self) -> str:
        """CBIN INDEX ticker format"""
        return f"{self.ticker} CBIN INDEX"
    
    def years_to_maturity(self, as_of_date: date = None) -> float:
        """Calculate years to maturity from given date"""
        if as_of_date is None:
            from datetime import date as dt
            as_of_date = dt.today()
        
        days_to_maturity = (self.maturity_date - as_of_date).days
        return days_to_maturity / 365.25