"""
Test script to verify all modules are properly set up
Run this from the credit_macro directory
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test models
        from src.models import (
            Region, Market, Tenor, Side,
            CDSIndex, CDSIndexDefinition,
            CDSSpreadData, CDSCurve,
            Position, Strategy,
            CDSDatabase
        )
        print("✓ Models imported successfully")
        
        # Test data layer
        from src.data import (
            BloombergCDSConnector,
            CDSDataManager,
            DataCache
        )
        print("✓ Data layer imported successfully")
        
        # Test utils
        from src.utils import setup_logger, get_logger
        print("✓ Utils imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without Bloomberg connection"""
    print("\nTesting basic functionality...")
    
    try:
        from src.models import Region, Market, Tenor, CDSIndex
        from src.models.database import CDSDatabase
        from datetime import datetime, date
        
        # Test enum
        region = Region.EU
        print(f"✓ Region enum: {region.value}")
        
        # Test CDSIndex
        index = CDSIndex(
            ticker="ITRX EUR CDSI S41 5Y",
            region="EU",
            market="IG",
            series=41,
            tenor="5Y",
            full_ticker="ITRX EUR CDSI S41 5Y"
        )
        print(f"✓ CDSIndex created: {index.bbg_ticker}")
        
        # Test database (will create a test database)
        db = CDSDatabase("test_cds.db")
        print("✓ Database initialized")
        db.close()
        
        # Clean up test database
        import os
        if os.path.exists("test_cds.db"):
            os.remove("test_cds.db")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_structures():
    """Test data structure creation"""
    print("\nTesting data structures...")
    
    try:
        from src.models import (
            CDSSpreadData, CDSCurve, Position, Strategy,
            Region, Market, Tenor, Side
        )
        from datetime import datetime
        
        # Test SpreadData
        spread = CDSSpreadData(
            index_id="EU_IG_S41_5Y",
            timestamp=datetime.now(),
            bid=50.0,
            ask=51.0,
            mid=50.5,
            last=50.7,
            dv01=4500
        )
        print(f"✓ SpreadData created: {spread.bid_ask_spread} bps bid-ask")
        
        # Test Curve
        curve = CDSCurve(
            region=Region.EU,
            market=Market.IG,
            series=41,
            observation_date=datetime.now(),
            spreads={
                Tenor.Y3: 45.0,
                Tenor.Y5: 50.0,
                Tenor.Y7: 57.0,
                Tenor.Y10: 65.0
            }
        )
        print(f"✓ Curve created: 5s10s slope = {curve.get_slope(Tenor.Y5, Tenor.Y10)} bps")
        
        # Test Position
        position = Position(
            index_id="EU_IG_S41_5Y",
            side=Side.BUY,
            notional=10000000,
            entry_date=datetime.now(),
            entry_spread=50.0,
            entry_dv01=4500,
            current_spread=49.0
        )
        print(f"✓ Position created: P&L = {position.pnl_bps} bps")
        
        # Test Strategy
        strategy = Strategy(
            name="Test_5s10s",
            strategy_type="5s10s",
            positions=[position],
            creation_date=datetime.now()
        )
        print(f"✓ Strategy created: Net DV01 = {strategy.net_dv01}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache():
    """Test caching functionality"""
    print("\nTesting cache...")
    
    try:
        from src.data.cache import DataCache
        import time
        
        cache = DataCache(expiry_minutes=0.01)  # 0.6 seconds for testing
        
        # Test set and get
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        print("✓ Cache set/get working")
        
        # Test expiry
        time.sleep(1)
        assert cache.get("test_key") is None
        print("✓ Cache expiry working")
        
        # Test stats
        stats = cache.get_stats()
        print(f"✓ Cache stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Cache test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("CDS Monitor Module Test Suite")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Data Structures", test_data_structures),
        ("Cache", test_cache)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, success in results if success)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Test Bloomberg connection with real data")
        print("2. Run: python -c 'from src.data import BloombergCDSConnector; c = BloombergCDSConnector()'")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)