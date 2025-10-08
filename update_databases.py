"""
CDS Monitor Database Update Script

This script updates both the raw historical spreads database and the TRS database
with the latest data from Bloomberg.

Usage:
    poetry run python update_databases.py [--days-back DAYS]
    
Arguments:
    --days-back: Number of days to look back for updates (default: 30)
    
Example:
    poetry run python update_databases.py --days-back 60
"""

import sys
import os
from pathlib import Path
import argparse
from datetime import datetime

# Add src to path for imports
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from models.database import CDSDatabase
from models.trs import TRSDatabaseBuilder


def update_databases(days_back: int = 30, verbose: bool = True):
    """
    Update both raw and TRS databases with latest data
    
    Args:
        days_back: Number of days to look back for updates
        verbose: Print detailed progress information
    """
    # Define database paths
    raw_db_path = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\raw\cds_indices_raw.db"
    trs_db_path = r"C:\source\repos\psc\packages\psc_csa_tools\credit_macro\data\processed\cds_trs.db"
    
    print("=" * 70)
    print(f"CDS Monitor Database Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Step 1: Update Raw Database
    print("\n[1/2] Updating Raw Historical Spreads Database...")
    print("-" * 70)
    
    try:
        raw_db = CDSDatabase(raw_db_path)
        raw_db.update_historical_data(days_back=days_back)
        raw_db.close()
        print("✓ Raw database updated successfully")
    except Exception as e:
        print(f"✗ Error updating raw database: {e}")
        return False
    
    # Step 2: Update TRS Database
    print("\n[2/2] Updating TRS (Total Return Swap) Database...")
    print("-" * 70)
    
    try:
        builder = TRSDatabaseBuilder(raw_db_path, trs_db_path)
        updated_count, failed_updates = builder.update_trs_database(
            days_back=days_back, 
            verbose=verbose
        )
        
        if failed_updates:
            print(f"⚠ Some series failed to update: {', '.join(failed_updates)}")
        else:
            print("✓ TRS database updated successfully")
            
    except Exception as e:
        print(f"✗ Error updating TRS database: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("✓ Database Update Complete!")
    print("=" * 70)
    print("\nBoth databases are now up to date with the latest CDS spreads.")
    print("You can now use the Streamlit app with current data.")
    print("\nTo launch the app, run:")
    print("  cd src/apps")
    print("  poetry run streamlit run streamlit_app.py")
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Update CDS Monitor databases with latest Bloomberg data'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=30,
        help='Number of days to look back for updates (default: 30)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed progress output'
    )
    
    args = parser.parse_args()
    
    success = update_databases(
        days_back=args.days_back,
        verbose=not args.quiet
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
