# Data manipulation libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List
import os
import sys
from io import StringIO
import re

# For the bt.get() call specifically, you can wrap it with output suppression
def suppress_output(func, *args, **kwargs):
    """Run function with suppressed stdout/stderr"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        result = func(*args, **kwargs)
        return result
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def load_and_filter_data(file_path, tickers, start_date, end_date):
    # Convert single ticker (string) to a list if needed
    if isinstance(tickers, str):
        tickers = [tickers]
        
    df = pd.read_csv(file_path)

    # Filter columns to available tickers
    available_tickers = [ticker for ticker in tickers if ticker in df.columns]
    columns_to_keep = ['Date'] + available_tickers if 'Date' in df.columns else available_tickers
    df = df[columns_to_keep] if all(col in df.columns for col in columns_to_keep) else df

    # Handle Date column or index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        df = df.set_index('Date')
    else:
        df.index = pd.to_datetime(df.index)
        df = df.loc[start_date:end_date]

    return df.sort_index(axis=0).sort_index(axis=1)

# Load Tickers from the tickers.txt file
def load_tickers(file_path: str) -> List[str]:
    try:
        with open(file_path, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        # Remove duplicates while preserving order
        tickers = list(dict.fromkeys(tickers))
        print(f"Loaded {len(tickers)} unique tickers from {file_path}")
        return tickers
    except Exception as e:
        print(f"Error loading tickers: {str(e)}")
        return []
    
def clean_column_names(df):
    """Clean column names with better handling of camelCase and preserve specific acronyms"""
    cleaned_columns = []
    for col in df.columns:
        # First handle camelCase by inserting spaces before uppercase letters
        col = re.sub(r'([a-z])([A-Z])', r'\1 \2', col)
        # Replace underscores with spaces
        col = col.replace('_', ' ')
        # Convert to title case
        col = col.title()
        
        # Keep specific acronyms in uppercase
        acronyms = ['Eps', 'Roe', 'Roa', 'Cagr', 'Ttm', 'Ev/']
        for acronym in acronyms:
            col = col.replace(acronym, acronym.upper())
        
        cleaned_columns.append(col)
    df.columns = cleaned_columns
    return df


def export_to_excel(output_file, data_dict):
    """
    Export multiple DataFrames to Excel, merging with existing data if appropriate.
    
    Parameters:
    -----------
    output_file : str
        Path to the Excel file
    data_dict : dict
        Dictionary where keys are sheet names and values are DataFrames to export
    """
    import pandas as pd
    import os
    
    # Check if file exists for mode selection
    file_exists = os.path.exists(output_file)
    mode = 'a' if file_exists else 'w'
    
    # For each DataFrame in the dictionary
    for sheet_name, df in data_dict.items():
        df_copy = df.copy()
        
        # Handle date index for time series data (daily_quotes, monthly_quotes)
        if sheet_name in ['daily_quotes', 'monthly_quotes']:
            # Handle Date column if present
            if 'Date' in df_copy.columns:
                df_copy = df_copy.set_index('Date')
            
            # Ensure datetime index
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy.index = pd.to_datetime(df_copy.index)
            
            # If file exists, try to merge with existing data
            if file_exists:
                try:
                    # Read existing sheet
                    existing_df = pd.read_excel(output_file, sheet_name=sheet_name, index_col=0)
                    existing_df.index = pd.to_datetime(existing_df.index)
                    
                    # Merge with existing data
                    merged_df = pd.concat([existing_df, df_copy], axis=1)
                    
                    # Remove duplicate columns
                    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
                    
                    # Write to Excel
                    with pd.ExcelWriter(output_file, mode='a', engine='openpyxl', 
                                       if_sheet_exists='replace') as writer:
                        merged_df.to_excel(writer, sheet_name=sheet_name)
                    
                    print(f"Successfully merged data into '{sheet_name}' sheet")
                    continue
                    
                except ValueError:
                    # Sheet doesn't exist, will write directly
                    pass
        
        # For non-time series data or new sheets, write directly
        with pd.ExcelWriter(output_file, mode=mode, engine='openpyxl', 
                           if_sheet_exists='replace') as writer:
            df_copy.to_excel(writer, sheet_name=sheet_name, index=True)
        
        # Set mode to append for subsequent sheets
        mode = 'a'
        
        print(f"{'Created' if not file_exists else 'Updated'} sheet '{sheet_name}'")
    
    print(f"Successfully exported all data to {output_file}")