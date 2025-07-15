import yfinance as yf
import pandas as pd
import concurrent.futures
import time
from tqdm import tqdm
import os
from datetime import datetime
import json

def fetch_fundamentals(tickers, max_workers=10, batch_size=100, fetch_sectors=False):
    """
    Fetch fundamental data for multiple tickers with efficient batching and error handling
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    max_workers : int
        Maximum number of concurrent threads
    batch_size : int
        Number of tickers to process in each batch
    fetch_sectors : bool
        Whether to fetch sector weightings data (for funds/ETFs)
    """
    def fetch_single_ticker(ticker):
        for attempt in range(3):  # 3 retry attempts
            try:
                ticker_obj = yf.Ticker(ticker)
                fund_data = ticker_obj.info
                
                result = {
                    # Company Information
                    'Name': fund_data.get('longName', 'Unknown'),
                    'Sector': fund_data.get('sector', 'Unknown'),
                    'Industry': fund_data.get('industry', 'Unknown'),
                    'Country': fund_data.get('country', 'Unknown'),
                    'Website': fund_data.get('website', 'Unknown'),
                    
                    # Size Metrics
                    'Market Cap': fund_data.get('marketCap', 0),
                    'Enterprise Value': fund_data.get('enterpriseValue', None),
                    'Float Shares': fund_data.get('floatShares', None),
                    'Shares Outstanding': fund_data.get('sharesOutstanding', None),
                    
                    # Valuation Metrics
                    'P/E (trailing)': fund_data.get('trailingPE', None),
                    'P/E (forward)': fund_data.get('forwardPE', None),
                    'P/S': fund_data.get('priceToSalesTrailing12Months', None),
                    'P/B': fund_data.get('priceToBook', None),
                    'EV/EBITDA': fund_data.get('enterpriseToEbitda', None),
                    'EV/Revenue': fund_data.get('enterpriseToRevenue', None),
                    
                    # Financial Performance
                    'Gross Margin (%)': fund_data.get('grossMargins', None),
                    'EBITDA Margin (%)': fund_data.get('ebitdaMargins', None),
                    'Operating Margin (%)': fund_data.get('operatingMargins', None),
                    'Profit Margin (%)': fund_data.get('profitMargins', None),
                    'ROE': fund_data.get('returnOnEquity', None),
                    'ROA': fund_data.get('returnOnAssets', None),
                    
                    # Earnings & Revenue
                    'Revenue (TTM)': fund_data.get('totalRevenue', None),
                    'Revenue Growth (%)': fund_data.get('revenueGrowth', None),
                    'EPS (trailing)': fund_data.get('trailingEps', None),
                    'EPS (forward)': fund_data.get('forwardEps', None),
                    'Earnings Growth (%)': fund_data.get('earningsGrowth', None),
                    'Earnings Quarterly Growth (%)': fund_data.get('earningsQuarterlyGrowth', None),
                    
                    # Balance Sheet Metrics
                    'Total Cash': fund_data.get('totalCash', None),
                    'Total Debt': fund_data.get('totalDebt', None),
                    'Debt to Equity': fund_data.get('debtToEquity', None),
                    'Current Ratio': fund_data.get('currentRatio', None),
                    'Quick Ratio': fund_data.get('quickRatio', None),
                    'Book Value': fund_data.get('bookValue', None),
                    
                    # Cash Flow Metrics
                    'Free Cash Flow': fund_data.get('freeCashflow', None),
                    'Operating Cash Flow': fund_data.get('operatingCashflow', None),
                    
                    # Dividend Information
                    'Dividend Yield (%)': fund_data.get('dividendYield', 0) / 100 if fund_data.get('dividendYield') else 0,
                    'Dividend Rate (%)': fund_data.get('dividendRate', 0) / 100 if fund_data.get('dividendRate') else 0,
                    '5Y Avg Dividend Yield (%)': fund_data.get('fiveYearAvgDividendYield', 0) / 100 if fund_data.get('fiveYearAvgDividendYield') else 0,
                    'Payout Ratio (%)': fund_data.get('payoutRatio', 0) if fund_data.get('payoutRatio') else 0,
                    
                    # Analyst Metrics
                    'Price': fund_data.get('regularMarketPrice', None),
                    'Target Price': fund_data.get('targetMeanPrice', None),
                    'Target High': fund_data.get('targetHighPrice', None),
                    'Target Low': fund_data.get('targetLowPrice', None),
                    'Analyst Rating': fund_data.get('recommendationKey', 'Unknown'),
                    'Analyst Rating Value': fund_data.get('recommendationMean', None),
                    
                    # Trading Information
                    'Beta': fund_data.get('beta', None),
                    '52W High': fund_data.get('fiftyTwoWeekHigh', None),
                    '52W Low': fund_data.get('fiftyTwoWeekLow', None),
                    '50 Day Avg': fund_data.get('fiftyDayAverage', None),
                    '200 Day Avg': fund_data.get('twoHundredDayAverage', None),
                    'Short Ratio': fund_data.get('shortRatio', None),
                    'Short % of Float': fund_data.get('shortPercentOfFloat', None),
                }
                
                # Add sector weightings if requested
                if fetch_sectors:
                    try:
                        funds_data = ticker_obj.get_funds_data()
                        if funds_data is not None and hasattr(funds_data, 'sector_weightings'):
                            result['Sector Weightings'] = funds_data.sector_weightings
                        else:
                            result['Sector Weightings'] = {}
                    except Exception as e:
                        print(f"Error fetching sector data for {ticker}: {e}")
                        result['Sector Weightings'] = {}
                
                return ticker, result
            except Exception as e:
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(10)
                    continue
                print(f"Failed to fetch data for {ticker}: {e}")
                return ticker, None
    
    # Process tickers in batches to avoid overloading API
    results = []
    ticker_batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    
    for batch_num, batch in enumerate(ticker_batches):
        print(f"Processing batch {batch_num+1}/{len(ticker_batches)} ({len(batch)} tickers)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_futures = {executor.submit(fetch_single_ticker, ticker): ticker for ticker in batch}
            
            for future in tqdm(concurrent.futures.as_completed(batch_futures), total=len(batch)):
                ticker = batch_futures[future]
                try:
                    result = future.result()
                    if result[1]:  # If data was retrieved successfully
                        results.append(result)
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
        
        # Add delay between batches
        if batch_num < len(ticker_batches) - 1:
            time.sleep(3)
    
    # Create DataFrame from results
    fundamentals_df = pd.DataFrame(index=tickers)
    for ticker, data in results:
        if data:
            for key, value in data.items():
                # Store sector weightings as JSON if it's a dictionary
                if key == 'Sector Weightings' and isinstance(value, dict):
                    fundamentals_df.loc[ticker, key] = json.dumps(value)
                else:
                    fundamentals_df.loc[ticker, key] = value
    
    # Round numeric columns to improve readability
    numeric_cols = [
        'P/E (trailing)', 'P/E (forward)', 'P/S', 'P/B', 'EV/EBITDA',
        'EPS (trailing)', 'EPS (forward)', 'Market Cap', 'Enterprise Value', 'Float Shares', 'Shares Outstanding',
        'EV/Revenue', 'ROE', 'ROA', 'Revenue (TTM)', 'Total Cash',
        'Total Debt', 'Debt to Equity', 'Current Ratio', 'Quick Ratio',
        'Book Value', 'Free Cash Flow', 'Operating Cash Flow', 'Price', 'Target Price', 'Target High',
        'Target Low', 'Analyst Rating Value', 'Beta', '52W High', '52W Low',
        '50 Day Avg', '200 Day Avg', 'Short Ratio', 'Short % of Float'
    ]

    # Round numeric columns to improve readability
    percentage_cols = [
        'Dividend Yield (%)', '5Y Avg Dividend Yield (%)', 'Profit Margin (%)',
        'Operating Margin (%)', 'Gross Margin (%)', 'EBITDA Margin (%)',
        'Revenue Growth (%)', 'Earnings Growth (%)', 'Earnings Quarterly Growth (%)',
        'Payout Ratio (%)'
    ]


    # Apply rounding to explicitly defined numeric columns
    fundamentals_df[numeric_cols] = fundamentals_df[numeric_cols].apply(pd.to_numeric, errors='coerce').round(2)
    fundamentals_df[percentage_cols] = fundamentals_df[percentage_cols].apply(pd.to_numeric, errors='coerce').round(4)
    
    return fundamentals_df

def fetch_fundamental_data(tickers, date, data_file, fetch_sectors=False):
    """
    Fetch fundamental data for tickers, only updating those that need it.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols to fetch data for
    date : datetime
        Reference date for the data (typically today)
    data_file : str
        Path to the CSV file for storing/reading data
    fetch_sectors : bool
        Whether to fetch sector weightings data (for funds/ETFs)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing fundamental data for all tickers
    """
    
    skip_download = False
    tickers_to_update = []

    if os.path.exists(data_file):
        try:
            # Read the existing data
            existing_df = pd.read_csv(data_file)
            # Convert Date column to datetime for comparison only
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
            
            # Get max date per ticker in the existing data
            latest_dates = existing_df.groupby('Ticker')['Date'].max()
            
            # Check if all required tickers have data for the current date
            today_date = pd.to_datetime(date.strftime('%Y-%m-%d'))
            outdated_tickers = []
            
            # Find tickers with missing or outdated data
            for ticker in tickers:
                if ticker not in latest_dates.index:
                    outdated_tickers.append(f"{ticker} (missing)")
                    tickers_to_update.append(ticker)
                elif latest_dates[ticker] != today_date:
                    outdated_tickers.append(f"{ticker} (last date: {latest_dates[ticker].strftime('%Y-%m-%d')})")
                    tickers_to_update.append(ticker)
            
            # If fetch_sectors is True, check for tickers with empty sector data
            if fetch_sectors and 'Sector Weightings' in existing_df.columns:
                # Get the latest data for each ticker
                latest_data = existing_df.sort_values('Date', ascending=False).drop_duplicates('Ticker')
                
                # Find tickers with empty sector weightings
                empty_sector_tickers = []
                for ticker in tickers:
                    # Skip tickers already scheduled for update
                    if ticker in tickers_to_update:
                        continue
                    
                    # Check if ticker exists in latest_data and has empty sector weightings
                    ticker_row = latest_data[latest_data['Ticker'] == ticker]
                    if not ticker_row.empty:
                        sector_data = ticker_row['Sector Weightings'].values[0]
                        # Check if sector data is empty (either "{}" or empty dict)
                        if sector_data == "{}" or not sector_data or pd.isna(sector_data):
                            empty_sector_tickers.append(ticker)
                
                # Add tickers with empty sector data to update list
                if empty_sector_tickers:
                    print(f"Found {len(empty_sector_tickers)} tickers with empty sector data that will be updated:")
                    for ticker in empty_sector_tickers:
                        print(f"- {ticker} (empty sector data)")
                        tickers_to_update.append(ticker)
            
            if outdated_tickers:
                print(f"Found {len(outdated_tickers)} outdated or missing tickers:")
                for ticker in outdated_tickers:
                    print(f"- {ticker}")
                print(f"Will update the {len(tickers_to_update)} ticker(s)")
            elif not tickers_to_update:  # Only skip if no tickers need updating
                skip_download = True
                print(f"All tickers already have data for {date.strftime('%Y-%m-%d')}. Skipping download.")
                
        except Exception as e:
            print(f"Error checking existing data: {e}")
            tickers_to_update = tickers
            
    else:
        # If file doesn't exist, we need all tickers
        tickers_to_update = tickers

    # Only proceed with download if there are tickers to update
    if tickers_to_update:
        # Download fundamentals data only for the tickers that need updating
        print(f"Fetching data for {len(tickers_to_update)} ticker(s)")
        updated_fundamentals_df = fetch_fundamentals(tickers_to_update, fetch_sectors=fetch_sectors)
        updated_fundamentals_df.index.name = 'Ticker'
        
        # Add "Date" column with the current date at position 0
        updated_fundamentals_df.insert(0, "Date", date.strftime('%Y-%m-%d'))
        
        # If we have existing data, merge with it
        if os.path.exists(data_file):
            # Filter out the rows for tickers we're updating
            filtered_existing_df = existing_df[~existing_df['Ticker'].isin(tickers_to_update)]
            
            # Convert to DataFrame if it's a Series
            if isinstance(updated_fundamentals_df, pd.Series):
                updated_fundamentals_df = pd.DataFrame(updated_fundamentals_df).transpose()
            
            # Reset index to get 'Ticker' as a column
            updated_fundamentals_df = updated_fundamentals_df.reset_index()
            
            # Concatenate the filtered existing data with the updated data
            fundamentals_df = pd.concat([filtered_existing_df, updated_fundamentals_df], ignore_index=True)
        else:
            # If no existing file, use only the updated data
            fundamentals_df = updated_fundamentals_df.reset_index()
        
        # Ensure all dates are consistently formatted as strings before saving
        fundamentals_df['Date'] = pd.to_datetime(fundamentals_df['Date']).dt.strftime('%Y-%m-%d')
        
        # Sort the DataFrame by Ticker and Date before saving
        fundamentals_df = fundamentals_df.sort_values(['Ticker', 'Date'], ascending=[True, True])
        
        # Save the combined dataframe to a CSV file
        fundamentals_df.to_csv(data_file, index=False)
        
        # Display the fundamentals dataframe
        print(f"Updated data for {len(tickers_to_update)} ticker(s)")
        return fundamentals_df
    else:
        # Read the existing data and return it
        fundamentals_df = pd.read_csv(data_file)
        
        # Parse sector weightings back to dictionaries if needed
        if 'Sector Weightings' in fundamentals_df.columns:
            fundamentals_df['Sector Weightings'] = fundamentals_df['Sector Weightings'].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x and x != "{}" else {}
            )
            
        print(f"Using existing data from {data_file}")
        return fundamentals_df