from io import StringIO
from typing import Dict, List
from datetime import datetime, timedelta

# API Requests
import requests
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session():
    """Create a requests session with retry configuration"""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries, pool_maxsize=10))
    return session

def fetch_data(api_key: str, session: requests.Session, days_back: int = None, max_pages: int = None, records_per_page: int = None, request_timeout: int = None) -> Dict:
    """Fetch stock news with pagination"""
    # Use parameters or fall back to default values
    days_back = days_back if days_back is not None else 7
    max_pages = max_pages if max_pages is not None else 10
    records_per_page = records_per_page if records_per_page is not None else 1000
    request_timeout = request_timeout if request_timeout is not None else 10
    
    # API base URL
    API_BASE_URL = "https://financialmodelingprep.com/api/v3/stock_news"
    
    # Calculate date range
    today = datetime.now().date()
    week_ago = today - timedelta(days=days_back)
    
    all_data = []
    
    # Loop through max_pages with records_per_page records each
    for page in range(max_pages):
        url = API_BASE_URL
        params = {
            "apikey": api_key,
            "from": week_ago.strftime('%Y-%m-%d'),
            "to": today.strftime('%Y-%m-%d'),
            "limit": records_per_page,
            "page": page
        }
        
        try:
            print(f"Fetching page {page + 1}/{max_pages}...")
            response = session.get(url, params=params, timeout=request_timeout)
            response.raise_for_status()
            data = response.json()
            
            if not data:  # If no more data, break the loop
                print(f"No more data found at page {page + 1}. Stopping pagination.")
                break
                
            all_data.extend(data)
            print(f"Page {page + 1}: {len(data)} articles fetched")
            
        except Exception as e:
            print(f"Error fetching page {page + 1}: {e}")
            continue
    
    print(f"Total articles fetched: {len(all_data)}")
    return all_data