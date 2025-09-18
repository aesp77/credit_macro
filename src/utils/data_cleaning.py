"""
Data cleaning utilities
Migrated from notebooks/04_pnl_reconstruction.ipynb
TO BE REVIEWED FOR REDUNDANCIES
"""

import pandas as pd
import numpy as np
from datetime import datetime



def clean_spread_data(data):
    '''Clean and fix spread data'''
    cleaned = data.copy()
    
    # Handle missing values
    cleaned = cleaned.fillna(method='ffill').fillna(method='bfill')
    
    # Fix zero spreads
    if 'px_last' in cleaned.columns:
        cleaned.loc[cleaned['px_last'] == 0, 'px_last'] = np.nan
        cleaned['px_last'] = cleaned['px_last'].fillna(method='ffill')
    
    return cleaned

# Clean all series
cleaned_data = {}
for key, data in historical_data.items():
    cleaned_data[key] = clean_spread_data(data)
    print(f'Cleaned {key}')
