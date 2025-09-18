"""
Strategy Configuration System based on Excel DataVal Sheet
Parses the configuration options into structured enums and classes
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

# Configuration Enums based on your Excel options
class WeightType(Enum):
    BETA = "Beta"
    CARRY_NEUTRAL = "Carry neutral" 
    DV01 = "Dv01"
    EQUAL = "Equal"

class Region(Enum):
    EU = "EU"
    US = "US"
    FINANCIAL = "Financial"  # Special category

class Market(Enum):
    IG = "IG"
    HY = "HY"
    SUB_FIN = "Sub Fin"
    SEN_FIN = "Sen Fin"

class StrategyType(Enum):
    STEEPENER_3S5S = "3s5s"
    STEEPENER_3S7S = "3s7s" 
    STEEPENER_3S10S = "3s10s"
    STEEPENER_5S7S = "5s7s"
    STEEPENER_5S10S = "5s10s"
    STEEPENER_7S10S = "7s10s"

class RegressionType(Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"

class Tenor(Enum):
    Y3 = "3"
    Y5 = "5" 
    Y7 = "7"
    Y10 = "10"

class Side(Enum):
    BUY = "Buy"
    SELL = "Sell"

class LegPosition(Enum):
    LEG_1 = "LEG 1"
    LEG_2 = "LEG 2"  
    LEG_3 = "LEG 3"
    X_AXIS = "X AXIS"
    Y_LEG_1 = "Y LEG 1"  # For plotting/analysis

class ThirdLeg(Enum):
    YES = "Y"
    NO = "N"

# Bloomberg ticker mappings
TICKER_MAPPINGS = {
    ('EU', 'IG'): 'ITXEB',
    ('US', 'IG'): 'CDXIG', 
    ('EU', 'HY'): 'ITXEX',
    ('US', 'HY'): 'CXPHY',
    ('EU', 'Sub Fin'): 'ITXEU',
    ('US', 'Sub Fin'): 'ITXEU',  # Same ticker?
    ('EU', 'Sen Fin'): 'ITXES',
    ('US', 'Sen Fin'): 'ITXES',  # Same ticker?
}

@dataclass
class StrategyLeg:
    """Individual leg of a strategy"""
    tenor: Tenor
    side: Side
    size: Optional[float] = None
    position: Optional[LegPosition] = None
    
    def get_bloomberg_ticker(self, region: Region, market: Market, series: int) -> str:
        """Generate Bloomberg ticker for this leg"""
        base_ticker = TICKER_MAPPINGS.get((region.value, market.value))
        if not base_ticker:
            raise ValueError(f"No ticker mapping for {region.value} {market.value}")
        
        return f"{base_ticker}{self.tenor.value}{series} CBIN Index"

@dataclass 
class StrategyConfig:
    """Complete strategy configuration"""
    strategy_type: StrategyType
    weight_type: WeightType
    region: Region
    market: Market
    regression_type: RegressionType
    primary_leg: StrategyLeg  # Only specify the first leg
    series: int = 42
    third_leg: ThirdLeg = ThirdLeg.NO
    base_notional: float = 10_000_000  # Base notional for primary leg
    
    def __post_init__(self):
        """Auto-generate additional legs based on strategy type and weighting"""
        # Set primary leg size to base notional
        self.primary_leg.size = self.base_notional
        self.legs = self._calculate_all_legs()
    
    def _calculate_all_legs(self) -> List[StrategyLeg]:
        """Calculate all legs based on strategy type and weighting method"""
        legs = [self.primary_leg]
        
        # Get tenors for this strategy type
        tenor_pairs = self._get_strategy_tenors()
        
        if len(tenor_pairs) >= 2:
            # 2-leg strategy (steepener/flattener)
            second_tenor = tenor_pairs[1]
            second_side = Side.BUY if self.primary_leg.side == Side.SELL else Side.SELL
            
            # Calculate size based on weighting method
            second_size = self._calculate_leg_size(
                primary_tenor=tenor_pairs[0],
                target_tenor=second_tenor,
                primary_size=self.base_notional
            )
            
            second_leg = StrategyLeg(
                tenor=second_tenor,
                side=second_side,
                size=second_size,
                position=LegPosition.LEG_2
            )
            legs.append(second_leg)
            
        if len(tenor_pairs) >= 3 and self.third_leg == ThirdLeg.YES:
            # 3-leg strategy (butterfly)
            third_tenor = tenor_pairs[2]
            third_side = self.primary_leg.side  # Same side as primary for butterfly
            
            third_size = self._calculate_leg_size(
                primary_tenor=tenor_pairs[0],
                target_tenor=third_tenor, 
                primary_size=self.base_notional
            )
            
            third_leg = StrategyLeg(
                tenor=third_tenor,
                side=third_side,
                size=third_size,
                position=LegPosition.LEG_3
            )
            legs.append(third_leg)
            
        return legs
    
    def _get_strategy_tenors(self) -> List[Tenor]:
        """Get the tenors involved in this strategy type"""
        tenor_mapping = {
            StrategyType.STEEPENER_3S5S: [Tenor.Y3, Tenor.Y5],
            StrategyType.STEEPENER_3S7S: [Tenor.Y3, Tenor.Y7], 
            StrategyType.STEEPENER_3S10S: [Tenor.Y3, Tenor.Y10],
            StrategyType.STEEPENER_5S7S: [Tenor.Y5, Tenor.Y7],
            StrategyType.STEEPENER_5S10S: [Tenor.Y5, Tenor.Y10],
            StrategyType.STEEPENER_7S10S: [Tenor.Y7, Tenor.Y10],
        }
        
        # For butterfly trades, add middle tenor
        if self.third_leg == ThirdLeg.YES:
            if self.strategy_type == StrategyType.STEEPENER_3S7S:
                return [Tenor.Y3, Tenor.Y5, Tenor.Y7]  # 3s5s7s butterfly
            elif self.strategy_type == StrategyType.STEEPENER_5S10S:
                return [Tenor.Y5, Tenor.Y7, Tenor.Y10]  # 5s7s10s butterfly
        
        return tenor_mapping.get(self.strategy_type, [self.primary_leg.tenor])
    
    def _calculate_leg_size(self, primary_tenor: Tenor, target_tenor: Tenor, primary_size: float) -> float:
        """Calculate leg size based on weighting method"""
        
        if self.weight_type == WeightType.EQUAL:
            return primary_size  # Same notional for all legs
            
        elif self.weight_type == WeightType.DV01:
            # DV01-neutral weighting - this will be calculated with real Bloomberg DV01 data
            # For now, return primary_size as placeholder - actual calculation happens in strategy execution
            return primary_size  # Placeholder - will be recalculated with real DV01s
            
        elif self.weight_type == WeightType.CARRY_NEUTRAL:
            # Carry-neutral weighting - placeholder for now
            return primary_size  
            
        elif self.weight_type == WeightType.BETA:
            # Beta-weighted - placeholder for now  
            return primary_size  
            
        else:
            return primary_size
    
    def calculate_real_leg_sizes(self, bloomberg_connector) -> 'StrategyConfig':
        """
        Calculate actual leg sizes using real Bloomberg DV01 data
        This replaces the placeholder sizes with actual weighted sizes
        
        Args:
            bloomberg_connector: Bloomberg connector instance with get_dv01() method
            
        Returns:
            Updated StrategyConfig with real sizes
        """
        if self.weight_type != WeightType.DV01:
            return self  # No changes needed for non-DV01 strategies
            
        # Get real DV01s from Bloomberg
        updated_legs = []
        primary_dv01 = None
        
        for i, leg in enumerate(self.legs):
            ticker = leg.get_bloomberg_ticker(self.region, self.market, self.series)
            leg_dv01 = bloomberg_connector.get_dv01(ticker, notional=10_000_000)
            
            if i == 0:  # Primary leg
                primary_dv01 = leg_dv01
                leg.size = self.base_notional  # Keep primary at base notional
            else:
                # Calculate DV01-neutral size: primary_notional * (primary_dv01 / target_dv01)
                if leg_dv01 and leg_dv01 != 0:
                    leg.size = self.base_notional * (primary_dv01 / leg_dv01)
                else:
                    leg.size = self.base_notional  # Fallback
                    
            updated_legs.append(leg)
        
        # Return new config with updated sizes
        new_config = StrategyConfig(
            strategy_type=self.strategy_type,
            weight_type=self.weight_type,
            region=self.region,
            market=self.market,
            regression_type=self.regression_type,
            primary_leg=self.primary_leg,
            series=self.series,
            third_leg=self.third_leg,
            base_notional=self.base_notional
        )
        new_config.legs = updated_legs
        return new_config
    
    def get_strategy_summary(self) -> dict:
        """Get strategy summary for Streamlit display - no printing"""
        return {
            'name': self.get_strategy_name(),
            'type': self.strategy_type.value,
            'weight_method': self.weight_type.value,
            'region': self.region.value,
            'market': self.market.value,
            'legs': [
                {
                    'tenor': leg.tenor.value + 'Y',
                    'side': leg.side.value,
                    'size': leg.size,
                    'ticker': leg.get_bloomberg_ticker(self.region, self.market, self.series)
                }
                for leg in self.legs
            ],
            'total_legs': len(self.legs),
            'is_butterfly': self.third_leg == ThirdLeg.YES
        }
    
    def get_strategy_name(self) -> str:
        """Generate strategy name"""
        return f"{self.region.value}_{self.market.value}_{self.strategy_type.value}_{self.weight_type.value}"
    
    def get_all_tickers(self) -> List[str]:
        """Get Bloomberg tickers for all legs"""
        return [leg.get_bloomberg_ticker(self.region, self.market, self.series) 
                for leg in self.legs]

# Updated strategy templates - only specify primary leg
STRATEGY_TEMPLATES = {
    "5s10s_steepener_dv01": StrategyConfig(
        strategy_type=StrategyType.STEEPENER_5S10S,
        weight_type=WeightType.DV01,
        region=Region.EU,
        market=Market.IG,
        regression_type=RegressionType.LINEAR,
        primary_leg=StrategyLeg(tenor=Tenor.Y5, side=Side.SELL, position=LegPosition.LEG_1),
        base_notional=10_000_000
    ),
    
    "3s5s7s_butterfly_equal": StrategyConfig(
        strategy_type=StrategyType.STEEPENER_3S7S,
        weight_type=WeightType.EQUAL, 
        region=Region.US,
        market=Market.IG,
        regression_type=RegressionType.QUADRATIC,
        third_leg=ThirdLeg.YES,
        primary_leg=StrategyLeg(tenor=Tenor.Y3, side=Side.BUY, position=LegPosition.LEG_1),
        base_notional=10_000_000
    ),
    
    "5s10s_steepener_beta": StrategyConfig(
        strategy_type=StrategyType.STEEPENER_5S10S,
        weight_type=WeightType.BETA,
        region=Region.US,
        market=Market.HY,
        regression_type=RegressionType.LINEAR,
        primary_leg=StrategyLeg(tenor=Tenor.Y5, side=Side.SELL, position=LegPosition.LEG_1),
        base_notional=25_000_000
    )
}

def create_simple_strategy(
    strategy_type: str,
    weight_type: str,
    region: str = 'EU',
    market: str = 'IG',
    primary_tenor: str = '5',
    primary_side: str = 'Sell',
    base_notional: float = 10_000_000,
    series: int = 42,
    third_leg: bool = False
) -> StrategyConfig:
    """
    Simple factory function to create a strategy with just one leg specification
    
    Usage:
        # Create 5s10s DV01-neutral steepener
        strategy = create_simple_strategy('5s10s', 'Dv01', primary_side='Sell')
        
        # Create 3s5s7s equal-weighted butterfly  
        strategy = create_simple_strategy('3s7s', 'Equal', third_leg=True)
    """
    return StrategyConfig(
        strategy_type=StrategyType(strategy_type),
        weight_type=WeightType(weight_type),
        region=Region(region),
        market=Market(market),
        regression_type=RegressionType.LINEAR,
        primary_leg=StrategyLeg(
            tenor=Tenor(primary_tenor),
            side=Side(primary_side),
            position=LegPosition.LEG_1
        ),
        base_notional=base_notional,
        series=series,
        third_leg=ThirdLeg.YES if third_leg else ThirdLeg.NO
    )

def create_strategy_from_config(config_dict: dict) -> StrategyConfig:
    """Create strategy config from dictionary (e.g., from Excel dataval sheet)"""
    
    # Parse legs from config
    legs = []
    for i in range(1, 4):  # LEG 1, LEG 2, LEG 3
        tenor_key = f'leg{i}_tenor'
        side_key = f'leg{i}_side' 
        size_key = f'leg{i}_size'
        
        if tenor_key in config_dict and config_dict[tenor_key]:
            leg = StrategyLeg(
                tenor=Tenor(str(config_dict[tenor_key])),
                side=Side(config_dict.get(side_key, 'Buy')),
                size=config_dict.get(size_key),
                position=LegPosition(f'LEG {i}')
            )
            legs.append(leg)
    
    return StrategyConfig(
        strategy_type=StrategyType(config_dict.get('strategy_type', '5s10s')),
        weight_type=WeightType(config_dict.get('weight_type', 'Dv01')),
        region=Region(config_dict.get('region', 'EU')),
        market=Market(config_dict.get('market', 'IG')),
        regression_type=RegressionType(config_dict.get('regression', 'linear')),
        series=int(config_dict.get('series', 42)),
        third_leg=ThirdLeg(config_dict.get('third_leg', 'N')),
        legs=legs
    )

# Example usage - minimal output for testing
if __name__ == "__main__":
    # Test automatic leg generation - quiet mode for Streamlit
    strategy = create_simple_strategy('5s10s', 'Dv01', primary_side='Sell')
    summary = strategy.get_strategy_summary()
    
    # Only print essential info for validation
    print(f"Strategy: {summary['name']}")
    print(f"Legs: {summary['total_legs']}")
    for leg in summary['legs']:
        print(f"  {leg['side']} {leg['tenor']}: ${leg['size']:,.0f}")
        
    # Test butterfly
    butterfly = create_simple_strategy('3s7s', 'Equal', third_leg=True)
    b_summary = butterfly.get_strategy_summary()
    print(f"\nButterfly legs: {b_summary['total_legs']}")
    for leg in b_summary['legs']:
        print(f"  {leg['side']} {leg['tenor']}: ${leg['size']:,.0f}")