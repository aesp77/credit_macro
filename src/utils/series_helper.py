"""
Helper functions for CDS series management
Handles series rolls and date mappings
"""
from datetime import datetime, date, timedelta
from typing import List, Tuple, Dict
import pandas as pd


class CDSSeriesHelper:
    """
    Helper class for managing CDS series rolls and date ranges
    
    CDS indices roll twice a year:
    - March 20 (Spring roll)
    - September 20 (Fall roll)
    """
    
    # Roll dates (month, day)
    MARCH_ROLL = (3, 20)
    SEPTEMBER_ROLL = (9, 20)
    
    @staticmethod
    def get_roll_dates(year: int) -> List[date]:
        """Get the two roll dates for a given year"""
        return [
            date(year, CDSSeriesHelper.MARCH_ROLL[0], CDSSeriesHelper.MARCH_ROLL[1]),
            date(year, CDSSeriesHelper.SEPTEMBER_ROLL[0], CDSSeriesHelper.SEPTEMBER_ROLL[1])
        ]
    
    @staticmethod
    def get_current_series(index_type: str = 'ITRX_EUR', as_of_date: date = None) -> int:
        """
        Get the current on-the-run series number
        
        Args:
            index_type: Type of index ('ITRX_EUR', 'CDX_IG', etc.)
            as_of_date: Date to check (default: today)
            
        Returns:
            Current series number
        """
        if as_of_date is None:
            as_of_date = date.today()
        
        # Base series numbers as of known dates
        # These need to be updated based on actual market data
        base_series = {
            'ITRX_EUR': {'date': date(2025, 9, 20), 'series': 44},  # S44 starts Sept 2025
            'CDX_IG': {'date': date(2025, 9, 20), 'series': 43},    # CDX typically one behind
            'ITRX_XOVER': {'date': date(2025, 9, 20), 'series': 44}
        }
        
        if index_type not in base_series:
            raise ValueError(f"Unknown index type: {index_type}")
        
        base_info = base_series[index_type]
        base_date = base_info['date']
        base_series_num = base_info['series']
        
        # Calculate how many rolls have occurred between base_date and as_of_date
        if as_of_date >= base_date:
            # Count forward
            rolls = 0
            current_date = base_date
            while current_date <= as_of_date:
                next_roll = CDSSeriesHelper.get_next_roll_date(current_date)
                if next_roll <= as_of_date:
                    rolls += 1
                    current_date = next_roll
                else:
                    break
            return base_series_num + rolls
        else:
            # Count backward
            rolls = 0
            current_date = base_date
            while current_date > as_of_date:
                prev_roll = CDSSeriesHelper.get_previous_roll_date(current_date - timedelta(days=1))
                if prev_roll > as_of_date:
                    rolls += 1
                    current_date = prev_roll
                else:
                    break
            return base_series_num - rolls
    
    @staticmethod
    def get_next_roll_date(as_of_date: date) -> date:
        """Get the next roll date after the given date"""
        year = as_of_date.year
        
        # Check this year's rolls
        march_roll = date(year, CDSSeriesHelper.MARCH_ROLL[0], CDSSeriesHelper.MARCH_ROLL[1])
        sept_roll = date(year, CDSSeriesHelper.SEPTEMBER_ROLL[0], CDSSeriesHelper.SEPTEMBER_ROLL[1])
        
        if as_of_date < march_roll:
            return march_roll
        elif as_of_date < sept_roll:
            return sept_roll
        else:
            # Next year's March roll
            return date(year + 1, CDSSeriesHelper.MARCH_ROLL[0], CDSSeriesHelper.MARCH_ROLL[1])
    
    @staticmethod
    def get_previous_roll_date(as_of_date: date) -> date:
        """Get the previous roll date before the given date"""
        year = as_of_date.year
        
        # Check this year's rolls
        march_roll = date(year, CDSSeriesHelper.MARCH_ROLL[0], CDSSeriesHelper.MARCH_ROLL[1])
        sept_roll = date(year, CDSSeriesHelper.SEPTEMBER_ROLL[0], CDSSeriesHelper.SEPTEMBER_ROLL[1])
        
        if as_of_date > sept_roll:
            return sept_roll
        elif as_of_date > march_roll:
            return march_roll
        else:
            # Previous year's September roll
            return date(year - 1, CDSSeriesHelper.SEPTEMBER_ROLL[0], CDSSeriesHelper.SEPTEMBER_ROLL[1])
    
    @staticmethod
    def get_series_date_range(series: int, index_type: str = 'ITRX_EUR') -> Tuple[date, date]:
        """
        Get the date range for a specific series
        
        Args:
            series: Series number
            index_type: Type of index
            
        Returns:
            Tuple of (start_date, end_date) for the series
        """
        # Get current series and date
        current_series = CDSSeriesHelper.get_current_series(index_type)
        current_date = date.today()
        
        # Calculate how many rolls back
        rolls_back = current_series - series
        
        # Work backwards from current date
        start_date = current_date
        for _ in range(rolls_back):
            start_date = CDSSeriesHelper.get_previous_roll_date(start_date - timedelta(days=1))
        
        # End date is the next roll
        end_date = CDSSeriesHelper.get_next_roll_date(start_date)
        
        return start_date, end_date
    
    @staticmethod
    def get_series_for_date_range(start_date: date, end_date: date, 
                                  index_type: str = 'ITRX_EUR',
                                  current_series: int = None) -> List[Dict]:
        """
        Get all series that were on-the-run during a date range
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            index_type: Type of index
            current_series: Override current series number (optional)
            
        Returns:
            List of dictionaries with series info
        """
        if current_series is None:
            current_series = CDSSeriesHelper.get_current_series(index_type, end_date)
        
        series_list = []
        
        # Work backwards from end_date to find all relevant series
        current_date = end_date
        series_num = current_series
        
        while current_date >= start_date:
            # Get the date range for this series
            series_start = CDSSeriesHelper.get_previous_roll_date(current_date)
            series_end = CDSSeriesHelper.get_next_roll_date(current_date)
            
            # Add to list
            series_list.append({
                'series': series_num,
                'start_date': max(series_start, start_date),
                'end_date': min(series_end, end_date),
                'is_current': series_num == current_series
            })
            
            # Move to previous series
            current_date = series_start - timedelta(days=1)
            series_num -= 1
            
            # Stop if we've gone before our start date
            if series_start < start_date:
                break
        
        return list(reversed(series_list))
    
    @staticmethod
    def get_series_schedule(current_series: int, num_series: int = 4) -> pd.DataFrame:
        """
        Create a schedule of series with their date ranges
        
        Args:
            current_series: Current on-the-run series number
            num_series: Number of series to include (going backwards)
            
        Returns:
            DataFrame with series schedule
        """
        schedule = []
        
        for i in range(num_series):
            series = current_series - i
            
            # Calculate dates (approximate - 6 months per series)
            months_back = i * 6
            
            if i == 0:
                start_date = CDSSeriesHelper.get_previous_roll_date(date.today())
                end_date = CDSSeriesHelper.get_next_roll_date(date.today())
                status = "On-the-run"
            else:
                # Approximate dates
                end_date = date.today() - timedelta(days=months_back * 30)
                end_date = CDSSeriesHelper.get_next_roll_date(end_date)
                start_date = CDSSeriesHelper.get_previous_roll_date(end_date - timedelta(days=1))
                status = "Off-the-run"
            
            schedule.append({
                'Series': f'S{series}',
                'Start Date': start_date,
                'End Date': end_date,
                'Status': status
            })
        
        return pd.DataFrame(schedule)


def get_historical_series_data(connector, manager, lookback_days: int = 360, 
                              current_series: int = None, index_type: str = 'ITRX_EUR',
                              region: str = 'EU', market: str = 'IG', tenor: str = '5Y'):
    """
    Automatically fetch historical data for the appropriate series based on date range
    
    Args:
        connector: BloombergCDSConnector instance
        manager: CDSDataManager instance
        lookback_days: Number of days to look back
        current_series: Current series number (will auto-detect if None)
        index_type: Type of index
        region: Region (EU, US, etc.)
        market: Market (IG, HY, XO, etc.)
        tenor: Tenor (5Y, 10Y, etc.)
        
    Returns:
        Dictionary of historical data by series
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    # Auto-detect current series if not provided
    if current_series is None:
        helper = CDSSeriesHelper()
        current_series = helper.get_current_series(index_type, end_date.date())
        print(f"Auto-detected current series: S{current_series}")
    
    # Get list of series to fetch
    helper = CDSSeriesHelper()
    series_info = helper.get_series_for_date_range(
        start_date.date(), 
        end_date.date(),
        index_type,
        current_series
    )
    
    print(f"\nSeries to fetch for {lookback_days} days history:")
    for info in series_info:
        print(f"  S{info['series']}: {info['start_date']} to {info['end_date']} "
              f"{'(current)' if info['is_current'] else ''}")
    
    # Fetch data for each series
    historical_data = {}
    
    for info in series_info:
        series = info['series']
        ticker = connector.get_index_ticker(region, market, series, tenor)
        key = f'S{series}_{tenor}'
        
        print(f'\nLoading {key} ({ticker})...')
        
        # Fetch data only for the relevant date range of this series
        series_start = max(info['start_date'], start_date.date())
        series_end = min(info['end_date'], end_date.date())
        
        data = connector.get_historical_spreads(
            ticker,
            start_date=series_start,
            end_date=series_end
        )
        
        if not data.empty:
            # Add series metadata
            data['series'] = series
            data['is_on_the_run'] = info['is_current']
            historical_data[key] = data
            print(f'  Loaded {len(data)} days for S{series}')
        else:
            print(f'  No data available for S{series}')
    
    print(f'\nTotal series loaded: {len(historical_data)}')
    
    # Combine all series into one continuous time series
    if historical_data:
        all_data = pd.concat(historical_data.values(), sort=True)
        all_data = all_data.sort_index()
        print(f'Combined data: {len(all_data)} total days')
        
        return historical_data, all_data
    
    return historical_data, pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # Test series detection
    helper = CDSSeriesHelper()
    
    # Get current series
    current = helper.get_current_series('ITRX_EUR')
    print(f"Current ITRX EUR series: S{current}")
    
    # Get series schedule
    schedule = helper.get_series_schedule(current, num_series=4)
    print("\nSeries Schedule:")
    print(schedule)
    
    # Get series for a date range
    start = date.today() - timedelta(days=365)
    end = date.today()
    series_list = helper.get_series_for_date_range(start, end, 'ITRX_EUR', current)
    
    print(f"\nSeries active between {start} and {end}:")
    for s in series_list:
        print(f"  S{s['series']}: {s['start_date']} to {s['end_date']}")