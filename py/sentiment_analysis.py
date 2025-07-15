# Import additional libraries for sentiment analysis
import polars as pl
import pandas as pd
from textblob import TextBlob
import re
from collections import defaultdict

# ===============================================================================
# SENTIMENT ANALYSIS FUNCTIONS
# ===============================================================================

def extract_stock_symbols(text, all_tickers=None, excluded_symbols=None):
    """Extract valid stock symbols from text, excluding common words"""
    if all_tickers is None:
        all_tickers = set()
    if excluded_symbols is None:
        excluded_symbols = {'AI', 'S', 'A', 'U', 'E', 'US', 'ET', 'TSXV', 'CODI', 'C'}
    
    symbols = re.findall(r'\b[A-Z]{1,5}\b', text)
    return [symbol for symbol in symbols 
            if symbol in all_tickers and symbol not in excluded_symbols]

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob and classify as bullish/bearish/neutral"""
    polarity = TextBlob(text).sentiment.polarity
    
    if polarity > 0.1:
        return 'bullish', polarity
    elif polarity < -0.1:
        return 'bearish', polarity
    else:
        return 'neutral', polarity

def calculate_stock_sentiment_metrics(df, all_tickers=None, excluded_symbols=None):
    """Calculate comprehensive sentiment metrics for each stock symbol"""
    if all_tickers is None:
        all_tickers = set()
    if excluded_symbols is None:
        excluded_symbols = {'AI', 'S', 'A', 'U', 'E', 'US', 'ET', 'TSXV', 'CODI', 'C'}
    
    stock_metrics = defaultdict(lambda: {
        'sentiment_scores': [],
        'bullish_count': 0,
        'bearish_count': 0,
        'neutral_count': 0,
        'total_articles': 0
    })
    
    # Process each news article
    for row in df.iter_rows(named=True):
        full_text = f"{row.get('title', '')} {row.get('text', '')}"
        mentioned_symbols = extract_stock_symbols(full_text, all_tickers, excluded_symbols)
        sentiment_type, sentiment_score = analyze_sentiment(full_text)
        
        # Update metrics for each mentioned symbol
        for symbol in mentioned_symbols:
            metrics = stock_metrics[symbol]
            metrics['sentiment_scores'].append(sentiment_score)
            metrics['total_articles'] += 1
            metrics[f'{sentiment_type}_count'] += 1
    
    # Calculate final metrics
    final_metrics = {}
    for symbol, data in stock_metrics.items():
        if data['total_articles'] > 0:
            total = data['total_articles']
            avg_sentiment = sum(data['sentiment_scores']) / len(data['sentiment_scores'])
            
            final_metrics[symbol] = {
                "articlesInLastWeek": total,
                "companyNewsScore": round((avg_sentiment + 1) / 2, 4),
                "sentiment": {
                    "bearishPercent": round(data['bearish_count'] / total, 4),
                    "bullishPercent": round(data['bullish_count'] / total, 4)
                },
                "averageSentimentScore": round(avg_sentiment, 4),
                "totalArticles": total
            }
    
    return final_metrics

# ===============================================================================
# SECTOR ANALYSIS & FUNDAMENTAL DATA INTEGRATION
# ===============================================================================

def calculate_sector_averages(sentiment_df, fundamentals_pandas):
    """Calculate sector-level sentiment averages"""
    sector_metrics = defaultdict(list)
    
    for row in sentiment_df.iter_rows(named=True):
        symbol = row['symbol']
        if symbol in fundamentals_pandas.index:
            sector = fundamentals_pandas.loc[symbol, 'Sector']
            sector_metrics[sector].append({
                'bullishPercent': row['bullishPercent'],
                'newsScore': row['companyNewsScore']
            })
    
    return {
        sector: {
            'sectorAverageBullishPercent': round(sum(m['bullishPercent'] for m in metrics) / len(metrics), 4),
            'sectorAverageNewsScore': round(sum(m['newsScore'] for m in metrics) / len(metrics), 4)
        }
        for sector, metrics in sector_metrics.items() if metrics
    }

def get_fundamental_value(symbol, column, default=0, fundamentals_df=None):
    """Safely get fundamental data value for a symbol"""
    if fundamentals_df is None:
        return default
    
    try:
        if symbol in fundamentals_df.index:
            value = fundamentals_df.loc[symbol, column]
            # Handle NaN values
            if pd.isna(value):
                return default
            return value
        else:
            return default
    except (KeyError, IndexError):
        return default

def process_sentiment_analysis(news_df, fundamentals_path='data/fundamentals_stock.csv', 
                             min_articles=3, min_news_score=0.45, display_results=True):
    """
    Process sentiment analysis by combining news data with fundamental stock data.
    
    Parameters:
    -----------
    news_df : pl.DataFrame
        News data with sentiment information
    fundamentals_path : str
        Path to fundamentals CSV file
    min_articles : int
        Minimum articles required in last week
    min_news_score : float
        Minimum company news score threshold
    display_results : bool
        Whether to display results
        
    Returns:
    --------
    dict : Contains comprehensive_screened, sector_summary, and fundamentals_pandas
    """
    import pandas as pd
    from IPython.display import display
    
    # Load fundamental data
    print("📊 Loading fundamental stock data...")
    fundamentals_df = pl.read_csv(fundamentals_path)
    fundamentals_pandas = fundamentals_df.to_pandas().set_index('Ticker')

    # Prepare ticker universe
    all_tickers = set(news_df['symbol'].to_list() + fundamentals_df['Ticker'].to_list())
    EXCLUDED_SYMBOLS = {'AI', 'S', 'A', 'U', 'E', 'US', 'ET', 'TSXV', 'CODI', 'C'}

    print(f"📈 Processing {len(fundamentals_df)} stocks across {len(all_tickers)} unique tickers")

    # Execute sentiment analysis
    print("🔍 Analyzing news sentiment by stock symbol...")
    sentiment_metrics = calculate_stock_sentiment_metrics(news_df, all_tickers, EXCLUDED_SYMBOLS)

    # Create sentiment dataframe
    sentiment_df = pl.DataFrame([{
        "symbol": symbol,
        "articlesInLastWeek": metrics["articlesInLastWeek"],
        "companyNewsScore": metrics["companyNewsScore"], 
        "bearishPercent": metrics["sentiment"]["bearishPercent"],
        "bullishPercent": metrics["sentiment"]["bullishPercent"],
        "averageSentimentScore": metrics["averageSentimentScore"],
        "totalArticles": metrics["totalArticles"]
    } for symbol, metrics in sentiment_metrics.items()]).sort(
        ["articlesInLastWeek", "companyNewsScore"], descending=[True, True]
    )

    # Add fundamental and sector data
    sector_averages = calculate_sector_averages(sentiment_df, fundamentals_pandas)
    
    # Helper function to get fundamental values with proper access to fundamentals_pandas
    def get_fund_value(symbol, column, default_val=0):
        return get_fundamental_value(symbol, column, default_val, fundamentals_pandas)
    
    sentiment_with_fundamentals = sentiment_df.with_columns([
        pl.col("symbol").map_elements(
            lambda x: sector_averages.get(get_fund_value(x, 'Sector', 'Unknown'), {}).get('sectorAverageBullishPercent', 0), 
            return_dtype=pl.Float64
        ).alias("sectorAverageBullishPercent"),
        pl.col("symbol").map_elements(
            lambda x: sector_averages.get(get_fund_value(x, 'Sector', 'Unknown'), {}).get('sectorAverageNewsScore', 0), 
            return_dtype=pl.Float64
        ).alias("sectorAverageNewsScore"),
        pl.col("symbol").map_elements(
            lambda x: get_fund_value(x, 'Sector', 'Unknown'), 
            return_dtype=pl.Utf8
        ).alias("sector"),
        pl.col("symbol").map_elements(
            lambda x: get_fund_value(x, 'Market Cap', 0.0), 
            return_dtype=pl.Float64
        ).alias("marketCap"),
        pl.col("symbol").map_elements(
            lambda x: get_fund_value(x, 'P/E (trailing)', 0.0), 
            return_dtype=pl.Float64
        ).alias("peRatio"),
        pl.col("symbol").map_elements(
            lambda x: get_fund_value(x, 'Price', 0.0), 
            return_dtype=pl.Float64
        ).alias("price")
    ])

    # Apply screening filters
    comprehensive_screened = sentiment_with_fundamentals.filter(
        (pl.col("articlesInLastWeek") >= min_articles) & 
        (pl.col("companyNewsScore") >= min_news_score)
    ).sort(["companyNewsScore", "articlesInLastWeek"], descending=[True, True])

    # Generate sector summary
    sector_summary = sentiment_with_fundamentals.filter(
        pl.col("sector") != "Unknown"
    ).group_by("sector").agg([
        pl.count("symbol").alias("stock_count"), 
        pl.mean("companyNewsScore").alias("avg_news_score"),
        pl.mean("bullishPercent").alias("avg_bullish_percent"), 
        pl.mean("articlesInLastWeek").alias("avg_articles"),
        pl.mean("marketCap").alias("avg_market_cap"), 
        pl.mean("peRatio").alias("avg_pe_ratio")
    ]).sort("avg_news_score", descending=True)

    # Display results
    if display_results:
        print(f"✅ Screened {len(comprehensive_screened)} qualifying stocks across {len(sector_averages)} sectors")
        print("\n📊 Top Sentiment Stocks:")
        display(comprehensive_screened.head())
        print("\n🏢 Sector Analysis:")
        display(sector_summary)

    return {
        'comprehensive_screened': comprehensive_screened,
        'sector_summary': sector_summary,
        'fundamentals_pandas': fundamentals_pandas,
        'sentiment_with_fundamentals': sentiment_with_fundamentals
    }