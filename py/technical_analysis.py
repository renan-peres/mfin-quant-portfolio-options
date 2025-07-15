import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from itertools import product
import warnings
import logging
import os
import sys
from io import StringIO
import matplotlib.pyplot as plt

import bt
import talib
from scipy import stats
from itertools import combinations
import random
import warnings
warnings.filterwarnings('ignore')


# Suppress warnings and logging
warnings.filterwarnings('ignore')
logging.getLogger('bt').setLevel(logging.CRITICAL + 1)
os.environ['BT_PROGRESS'] = 'False'

STRATEGY_NAMES = ['SMA_Cross_Signal', 'EMA_Cross_Signal', 'ADX_Trend_Signal', 'RSI_Signal']

# ===============================================================================
# TECHNICAL INDICATORS & SIGNALS
# ===============================================================================

def calculate_technical_indicators(prices_df, sma_short=50, sma_long=200, ema_short=26, ema_long=52):
    """Calculate SMA and EMA technical indicators with customizable timeperiods"""
    indicators = {}
    min_required = max(sma_long, ema_long) + 10
    
    for ticker in prices_df.columns:       
        close = prices_df[ticker].dropna()
        if len(close) < min_required:
            continue
            
        ticker_indicators = pd.DataFrame(index=close.index)
        ticker_indicators['Close'] = close
        
        try:
            ticker_indicators[f'SMA_{sma_short}'] = talib.SMA(close, timeperiod=sma_short)
            ticker_indicators[f'SMA_{sma_long}'] = talib.SMA(close, timeperiod=sma_long)
            ticker_indicators[f'EMA_{ema_short}'] = talib.EMA(close, timeperiod=ema_short)
            ticker_indicators[f'EMA_{ema_long}'] = talib.EMA(close, timeperiod=ema_long)
            indicators[ticker] = ticker_indicators
        except Exception:
            continue
    return indicators

def generate_trading_signals(indicators_dict, sma_short=50, sma_long=200, ema_short=26, ema_long=52):
    """Generate buy/sell signals based on SMA and EMA crossovers"""
    strategies = {}
    for ticker, indicators in indicators_dict.items():
        try:
            ticker_signals = pd.DataFrame(index=indicators.index)
            ticker_signals['SMA_Cross_Signal'] = np.where(
                indicators[f'SMA_{sma_short}'] > indicators[f'SMA_{sma_long}'], 1, 
                np.where(indicators[f'SMA_{sma_short}'] < indicators[f'SMA_{sma_long}'], -1, 0)
            )
            ticker_signals['EMA_Cross_Signal'] = np.where(
                indicators[f'EMA_{ema_short}'] > indicators[f'EMA_{ema_long}'], 1, 
                np.where(indicators[f'EMA_{ema_short}'] < indicators[f'EMA_{ema_long}'], -1, 0)
            )
            strategies[ticker] = ticker_signals
        except Exception:
            continue
    return strategies

# ===============================================================================
# BACKTESTING UTILITIES
# ===============================================================================

def safe_get_stat(stats, strategy_col, stat_names, default=0):
    """Safely extract statistics with fallback options"""
    for stat_name in stat_names:
        if stat_name in stats.index:
            value = stats.loc[stat_name, strategy_col]
            return value if not pd.isna(value) else default
    return default

def run_backtest_silent(backtest):
    """Run backtest while suppressing output"""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = sys.stderr = StringIO()
        return bt.run(backtest)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

def extract_performance_metrics(result, strategy_name):
    """Extract key performance metrics from backtest result"""
    try:
        stats = result.stats
        return {
            'result': result,
            'total_return': safe_get_stat(stats, strategy_name, ['total_return']),
            'sharpe_ratio': safe_get_stat(stats, strategy_name, ['daily_sharpe', 'monthly_sharpe', 'yearly_sharpe']),
            'max_drawdown': safe_get_stat(stats, strategy_name, ['max_drawdown']),
            'volatility': safe_get_stat(stats, strategy_name, ['daily_vol', 'monthly_vol', 'yearly_vol']),
            'cagr': safe_get_stat(stats, strategy_name, ['cagr'])
        }
    except Exception:
        return None

# ===============================================================================
# OPTIMIZATION ENGINE - CONSOLIDATED
# ===============================================================================

def calculate_performance_metrics(returns):
    """Calculate all performance metrics from returns array"""
    if len(returns) < 5:
        return None
        
    total_return = np.prod(1 + returns) - 1
    volatility = np.std(returns) * np.sqrt(252)
    mean_return = np.mean(returns) * 252
    
    if volatility <= 1e-10:
        return None
        
    sharpe = mean_return / volatility
    
    # Sortino ratio
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        downside_deviation = np.std(negative_returns) * np.sqrt(252)
        sortino = mean_return / downside_deviation if downside_deviation > 1e-10 else sharpe
    else:
        sortino = sharpe if mean_return > 0 else 0
    
    # Drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative / running_max) - 1
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
    
    # CAGR
    years = len(returns) / 252
    cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    if not all(np.isfinite([sortino, sharpe, cagr])):
        return None
        
    return {
        'cagr': cagr,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'total_return': total_return
    }

def optimize_strategy_parameters(data, parameter_ranges, strategy_type):
    """Optimize parameters for a single strategy type"""
    results = []
    
    if strategy_type == 'SMA_Cross_Signal':
        combinations = product(parameter_ranges['short_periods'], parameter_ranges['long_periods'])
        for short_period, long_period in combinations:
            if short_period >= long_period:
                continue
                
            # Calculate moving averages
            ma_short = data.rolling(short_period).mean()
            ma_long = data.rolling(long_period).mean()
            
            # Create target weights
            target_weights = ma_long.copy()
            target_weights[ma_short > ma_long] = 1.0
            target_weights[ma_short <= ma_long] = -1.0
            target_weights = target_weights.fillna(0)
            
            # Calculate returns
            returns = data.pct_change().fillna(0)
            strategy_returns = (returns * target_weights.shift(1)).fillna(0)
            valid_returns = strategy_returns[~np.isnan(strategy_returns)]
            
            metrics = calculate_performance_metrics(valid_returns)
            if metrics:
                metrics.update({
                    'strategy_type': strategy_type,
                    'short_period': short_period,
                    'long_period': long_period
                })
                results.append(metrics)
                
    elif strategy_type == 'EMA_Cross_Signal':
        combinations = product(parameter_ranges['short_periods'], parameter_ranges['long_periods'])
        for short_period, long_period in combinations:
            if short_period >= long_period:
                continue
                
            # Calculate exponential moving averages
            ma_short = data.ewm(span=short_period).mean()
            ma_long = data.ewm(span=long_period).mean()
            
            # Create target weights
            target_weights = ma_long.copy()
            target_weights[ma_short > ma_long] = 1.0
            target_weights[ma_short <= ma_long] = -1.0
            target_weights = target_weights.fillna(0)
            
            # Calculate returns
            returns = data.pct_change().fillna(0)
            strategy_returns = (returns * target_weights.shift(1)).fillna(0)
            valid_returns = strategy_returns[~np.isnan(strategy_returns)]
            
            metrics = calculate_performance_metrics(valid_returns)
            if metrics:
                metrics.update({
                    'strategy_type': strategy_type,
                    'short_period': short_period,
                    'long_period': long_period
                })
                results.append(metrics)
    
    return results

def optimize_single_ticker_parameters(args):
    """Optimized single ticker parameter optimization"""
    ticker, ticker_data, parameter_ranges = args
    
    if len(ticker_data) < 250:
        return ticker, None
    
    data = ticker_data[ticker].dropna()
    all_results = []
    
    # Optimize both strategies
    for strategy in ['SMA_Cross_Signal', 'EMA_Cross_Signal']:
        strategy_results = optimize_strategy_parameters(data, parameter_ranges[strategy], strategy)
        all_results.extend(strategy_results)
    
    if not all_results:
        return ticker, None
    
    results_df = pd.DataFrame(all_results)
    best_idx = results_df['sortino_ratio'].idxmax()
    best_params = results_df.loc[best_idx].to_dict()
    
    return ticker, {
        'best_strategy': best_params['strategy_type'],
        'best_params': best_params,
        'strategy_type': best_params['strategy_type'],
        'results_df': results_df
    }

def find_optimal_portfolio_with_parameter_optimization(quotes, min_cagr=0.0, max_volatility=0.3, max_stocks=10, n_jobs=None, heatmap_metric='sortino', parameter_ranges=None):
    """Optimized portfolio optimization with consolidated logic"""
    if n_jobs is None:
        n_jobs = min(mp.cpu_count() - 1, len(quotes.columns))
    
    if parameter_ranges is None:
        parameter_ranges = {
            'SMA_Cross_Signal': {
                'short_periods': list(range(10, 45, 5)),
                'long_periods': list(range(60, 200, 20))
            },
            'EMA_Cross_Signal': {
                'short_periods': list(range(10, 30, 5)),
                'long_periods': list(range(35, 65, 5))
            }
        }
    
    # Prepare data for parallel processing
    ticker_data_list = []
    for ticker in quotes.columns:
        ticker_data = quotes[[ticker]].dropna()
        if len(ticker_data) >= 250:
            ticker_data_list.append((ticker, ticker_data, parameter_ranges))
    
    # Parallel optimization
    all_optimization_results = {}
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_ticker = {
            executor.submit(optimize_single_ticker_parameters, args): args[0] 
            for args in ticker_data_list
        }
        for future in as_completed(future_to_ticker):
            ticker, result = future.result()
            if result is not None:
                all_optimization_results[ticker] = result
    
    if not all_optimization_results:
        return None
    
    # Filter and select top performers
    filter_data = [{
        'ticker': ticker,
        'cagr': results['best_params']['cagr'],
        'volatility': results['best_params']['volatility'],
        'sharpe_ratio': results['best_params']['sharpe_ratio'],
        'sortino_ratio': results['best_params']['sortino_ratio']
    } for ticker, results in all_optimization_results.items()]
    
    filter_df = pd.DataFrame(filter_data)
    qualified_mask = (filter_df['cagr'] >= min_cagr) & (filter_df['volatility'] <= max_volatility)
    qualified_df = filter_df[qualified_mask]
    
    if qualified_df.empty:
        return None
    
    top_df = qualified_df.nlargest(max_stocks, 'sortino_ratio')
    top_tickers = top_df['ticker'].tolist()
    
    # Generate portfolio weights and signals
    portfolio_weights = pd.DataFrame(index=quotes.index, columns=quotes.columns).fillna(0)
    optimized_signals = {}
    magnitude = 1.0 / len(top_tickers)
    
    for ticker in top_tickers:
        params = all_optimization_results[ticker]['best_params']
        ticker_data = quotes[ticker].dropna()
        
        # Calculate moving averages based on strategy type
        if params['strategy_type'] == 'SMA_Cross_Signal':
            ma_short = ticker_data.rolling(params['short_period']).mean()
            ma_long = ticker_data.rolling(params['long_period']).mean()
        else:  # EMA_Cross_Signal
            ma_short = ticker_data.ewm(span=params['short_period']).mean()
            ma_long = ticker_data.ewm(span=params['long_period']).mean()
        
        # Create target weights
        target_weights = ma_long.copy()
        target_weights[ma_short > ma_long] = magnitude
        target_weights[ma_short <= ma_long] = -magnitude
        target_weights = target_weights.fillna(0)
        
        # Store signals and weights
        ticker_signals = pd.DataFrame(index=ticker_data.index)
        ticker_signals[params['strategy_type']] = np.where(target_weights > 0, 1, 
                                                         np.where(target_weights < 0, -1, 0))
        optimized_signals[ticker] = ticker_signals
        
        reindexed_weights = target_weights.reindex(quotes.index, method='ffill').fillna(0)
        portfolio_weights[ticker] = reindexed_weights
    
    # Calculate portfolio performance
    portfolio_returns = []
    for i in range(1, len(quotes)):
        daily_return = 0
        for ticker in top_tickers:
            if ticker in quotes.columns:
                try:
                    current_price = quotes[ticker].iloc[i]
                    previous_price = quotes[ticker].iloc[i-1]
                    price_return = (current_price / previous_price) - 1
                    weight = portfolio_weights[ticker].iloc[i-1]
                    daily_return += price_return * weight
                except (IndexError, KeyError, ZeroDivisionError):
                    continue
        portfolio_returns.append(daily_return)
    
    portfolio_returns = np.array(portfolio_returns)
    portfolio_returns = portfolio_returns[~np.isnan(portfolio_returns)]
    
    # Calculate portfolio statistics
    if len(portfolio_returns) > 0:
        portfolio_total_return = np.prod(1 + portfolio_returns) - 1
        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)
        portfolio_sharpe = (np.mean(portfolio_returns) * 252) / (portfolio_volatility + 1e-10)
        portfolio_cumulative = np.cumprod(1 + portfolio_returns)
        portfolio_running_max = np.maximum.accumulate(portfolio_cumulative)
        portfolio_drawdown = (portfolio_cumulative / portfolio_running_max) - 1
        portfolio_max_drawdown = np.min(portfolio_drawdown)
        years = len(portfolio_returns) / 252
        portfolio_cagr = (1 + portfolio_total_return) ** (1/years) - 1 if years > 0 else 0
    else:
        portfolio_cagr = portfolio_sharpe = portfolio_volatility = portfolio_max_drawdown = 0
        portfolio_total_return = 0
    
    portfolio_stats = {
        'cagr': portfolio_cagr,
        'total_return': portfolio_total_return,
        'sharpe_ratio': portfolio_sharpe,
        'volatility': portfolio_volatility,
        'max_drawdown': portfolio_max_drawdown
    }
    
    # Create summary results
    optimization_summary = [{
        'Ticker': ticker,
        'Strategy': all_optimization_results[ticker]['best_params']['strategy_type'],
        'Best_Sharpe': all_optimization_results[ticker]['best_params']['sharpe_ratio'],
        'Best_Sortino': all_optimization_results[ticker]['best_params']['sortino_ratio'],
        'CAGR': all_optimization_results[ticker]['best_params']['cagr'],
        'Max_Drawdown': all_optimization_results[ticker]['best_params']['max_drawdown'],
        'Volatility': all_optimization_results[ticker]['best_params']['volatility'],
        'Short_Period': all_optimization_results[ticker]['best_params']['short_period'],
        'Long_Period': all_optimization_results[ticker]['best_params']['long_period']
    } for ticker in top_tickers]
    
    best_strategies_dict = {ticker: {
        'strategy': all_optimization_results[ticker]['best_params']['strategy_type'],
        'cagr': all_optimization_results[ticker]['best_params']['cagr'],
        'total_return': all_optimization_results[ticker]['best_params'].get('total_return', 0),
        'max_drawdown': all_optimization_results[ticker]['best_params']['max_drawdown'],
        'sharpe_ratio': all_optimization_results[ticker]['best_params']['sharpe_ratio'],
        'sortino_ratio': all_optimization_results[ticker]['best_params']['sortino_ratio'],
        'volatility': all_optimization_results[ticker]['best_params']['volatility']
    } for ticker in top_tickers}
    
    return {
        'portfolio_stats': portfolio_stats,
        'portfolio_weights': portfolio_weights,
        'optimization_summary': pd.DataFrame(optimization_summary),
        'best_strategies_df': pd.DataFrame.from_dict(best_strategies_dict, orient='index'),
        'selected_tickers': top_tickers,
        'optimized_signals': optimized_signals,
        'all_optimization_results': all_optimization_results,
        'processing_time': 'Fast parallel processing completed',
        'heatmap_metric': heatmap_metric,
        'parameter_ranges': parameter_ranges
    }

# ===============================================================================
# PLOTTING FUNCTIONS
# ===============================================================================

def plot_optimization_heatmaps(optimization_results, metric=None):
    """Simplified heatmap plotting with enhanced titles"""
    import seaborn as sns
    
    if metric is None:
        metric = optimization_results.get('heatmap_metric', 'sortino')
    
    metric_mapping = {'cagr': 'cagr', 'sharpe': 'sharpe_ratio', 'sortino': 'sortino_ratio'}
    metric_column = metric_mapping.get(metric, 'sortino_ratio')
    metric_display = metric.title()
    
    for ticker in optimization_results['selected_tickers']:
        if ticker not in optimization_results['all_optimization_results']:
            continue
        
        results = optimization_results['all_optimization_results'][ticker]
        results_df = results['results_df']
        strategy_type = results['strategy_type']
        best_params = results['best_params']
        
        strategy_filtered_df = results_df[results_df['strategy_type'] == strategy_type].copy()
        valid_results_df = strategy_filtered_df[
            (~strategy_filtered_df[metric_column].isna()) & 
            (~strategy_filtered_df[metric_column].isin([np.inf, -np.inf]))
        ].copy()
        
        if valid_results_df.empty:
            continue
        
        plt.figure(figsize=(12, 8))
        
        pivot_table = valid_results_df.pivot_table(
            values=metric_column, 
            index='long_period', 
            columns='short_period',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', 
                   square=False, linewidths=0.5)
        
        # Invert y-axis
        plt.gca().invert_yaxis()
        
        # Enhanced title with best parameters and metrics
        best_cagr = best_params['cagr']
        best_sharpe = best_params['sharpe_ratio']
        best_sortino = best_params['sortino_ratio']
        
        title_line1 = f'{ticker} - {strategy_type} Parameter Optimization ({metric_display}-Based)'
        title_line2 = f'Best Parameters: {best_params["short_period"]}/{best_params["long_period"]}'
        title_line3 = f'(CAGR: {best_cagr:.1%}, Sharpe: {best_sharpe:.3f}, Sortino: {best_sortino:.3f})'
        
        plt.title(f'{title_line1}\n{title_line2}\n{title_line3}',
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.xlabel('Short Period')
        plt.ylabel('Long Period')
        
        # Highlight best combination
        best_short = best_params['short_period']
        best_long = best_params['long_period']
        if best_short in pivot_table.columns and best_long in pivot_table.index:
            short_idx = list(pivot_table.columns).index(best_short)
            long_idx = list(pivot_table.index).index(best_long)
            rect = plt.Rectangle((short_idx, long_idx), 1, 1, 
                               fill=False, edgecolor='red', linewidth=3)
            plt.gca().add_patch(rect)
        
        plt.tight_layout()
        plt.show()
        
def plot_complete_strategy_analysis(optimization_results, quotes, training_set, test_set, cols=2, rows=None):
    """Complete function to generate and plot optimized trading signals for full dataset"""
    selected_tickers = optimization_results['selected_tickers']
    optimization_summary = optimization_results['optimization_summary']
    
    # Generate signals for full dataset
    full_technical_indicators = {}
    full_trading_signals = {}
    
    for ticker in selected_tickers:
        if ticker in optimization_results['all_optimization_results']:
            params = optimization_results['all_optimization_results'][ticker]['best_params']
            strategy_type = params['strategy_type']
            ticker_full_data = quotes[[ticker]].dropna()
            
            ticker_indicators = pd.DataFrame(index=ticker_full_data.index)
            ticker_indicators['Close'] = ticker_full_data[ticker]
            
            if strategy_type == 'SMA_Cross_Signal':
                short_period = params['short_period']
                long_period = params['long_period']
                ticker_indicators[f'SMA_{short_period}'] = ticker_full_data[ticker].rolling(short_period).mean()
                ticker_indicators[f'SMA_{long_period}'] = ticker_full_data[ticker].rolling(long_period).mean()
                
                ticker_signals = pd.DataFrame(index=ticker_full_data.index)
                position = np.where(
                    ticker_indicators[f'SMA_{short_period}'] > ticker_indicators[f'SMA_{long_period}'], 1,
                    np.where(ticker_indicators[f'SMA_{short_period}'] < ticker_indicators[f'SMA_{long_period}'], -1, 0)
                )
                ticker_signals['SMA_Cross_Signal'] = position
                
                position_series = pd.Series(position, index=ticker_full_data.index)
                signal_changes = position_series.diff()
                ticker_signals['Buy_Signal'] = np.where(signal_changes == 2, 1, 0)
                ticker_signals['Sell_Signal'] = np.where(signal_changes == -2, 1, 0)
                ticker_signals['Buy_Signal'] = np.where(
                    (signal_changes == 1) & (position_series == 1), 1, ticker_signals['Buy_Signal']
                )
                ticker_signals['Sell_Signal'] = np.where(
                    (signal_changes == -1) & (position_series == -1), 1, ticker_signals['Sell_Signal']
                )
                
            else:  # EMA_Cross_Signal
                short_period = params['short_period']
                long_period = params['long_period']
                ticker_indicators[f'EMA_{short_period}'] = ticker_full_data[ticker].ewm(span=short_period).mean()
                ticker_indicators[f'EMA_{long_period}'] = ticker_full_data[ticker].ewm(span=long_period).mean()
                
                ticker_signals = pd.DataFrame(index=ticker_full_data.index)
                position = np.where(
                    ticker_indicators[f'EMA_{short_period}'] > ticker_indicators[f'EMA_{long_period}'], 1,
                    np.where(ticker_indicators[f'EMA_{short_period}'] < ticker_indicators[f'EMA_{long_period}'], -1, 0)
                )
                ticker_signals['EMA_Cross_Signal'] = position
                
                position_series = pd.Series(position, index=ticker_full_data.index)
                signal_changes = position_series.diff()
                ticker_signals['Buy_Signal'] = np.where(signal_changes == 2, 1, 0)
                ticker_signals['Sell_Signal'] = np.where(signal_changes == -2, 1, 0)
                ticker_signals['Buy_Signal'] = np.where(
                    (signal_changes == 1) & (position_series == 1), 1, ticker_signals['Buy_Signal']
                )
                ticker_signals['Sell_Signal'] = np.where(
                    (signal_changes == -1) & (position_series == -1), 1, ticker_signals['Sell_Signal']
                )
            
            full_technical_indicators[ticker] = ticker_indicators
            full_trading_signals[ticker] = ticker_signals
    
    # Plot the results with minimal plotting function
    if not selected_tickers:
        return full_technical_indicators, full_trading_signals
    
    cols = min(cols, len(selected_tickers))
    if rows is None:
        rows = (len(selected_tickers) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
    axes_flat = [axes] if len(selected_tickers) == 1 else axes.flatten()
    
    for idx, ticker in enumerate(selected_tickers):
        ax = axes_flat[idx]
        try:
            full_price_data = quotes[ticker].dropna()
            training_data = training_set[ticker].dropna() if ticker in training_set.columns else pd.Series(dtype=float)
            test_data = test_set[ticker].dropna() if ticker in test_set.columns else pd.Series(dtype=float)
            
            indicators = full_technical_indicators[ticker]
            signals = full_trading_signals[ticker]
            
            strategy_info = optimization_summary[optimization_summary['Ticker'] == ticker].iloc[0].to_dict() if not optimization_summary[optimization_summary['Ticker'] == ticker].empty else {}
            
            # Add background colors
            if not training_data.empty:
                ax.axvspan(training_data.index[0], training_data.index[-1], 
                        alpha=0.1, color='#93cbf9', label='Training Period', zorder=0)
            if not test_data.empty:
                ax.axvspan(test_data.index[0], test_data.index[-1], 
                        alpha=0.05, color='white', label='Test Period', zorder=0)
            
            # Plot price
            ax.plot(full_price_data.index, full_price_data.values, label='Price', color='black', linewidth=2, zorder=1)
            
            # Plot indicators and signals
            strategy = strategy_info.get('Strategy', 'SMA_Cross_Signal')
            short_period = strategy_info.get('Short_Period', 20)
            long_period = strategy_info.get('Long_Period', 50)
            
            if strategy == 'SMA_Cross_Signal':
                sma_short_col = f'SMA_{short_period}'
                sma_long_col = f'SMA_{long_period}'
                if sma_short_col in indicators.columns and sma_long_col in indicators.columns:
                    ax.plot(indicators.index, indicators[sma_short_col], 
                        label=f'SMA {short_period}', alpha=0.8, color='blue', linewidth=1.5)
                    ax.plot(indicators.index, indicators[sma_long_col], 
                        label=f'SMA {long_period}', alpha=0.8, color='orange', linewidth=1.5)
            elif strategy == 'EMA_Cross_Signal':
                ema_short_col = f'EMA_{short_period}'
                ema_long_col = f'EMA_{long_period}'
                if ema_short_col in indicators.columns and ema_long_col in indicators.columns:
                    ax.plot(indicators.index, indicators[ema_short_col], 
                        label=f'EMA {short_period}', alpha=0.8, color='green', linewidth=1.5)
                    ax.plot(indicators.index, indicators[ema_long_col], 
                        label=f'EMA {long_period}', alpha=0.8, color='red', linewidth=1.5)
            
            # Plot signals
            buy_signals = signals.get('Buy_Signal', pd.Series(dtype=float, index=signals.index))
            sell_signals = signals.get('Sell_Signal', pd.Series(dtype=float, index=signals.index))
            
            buy_count = sell_count = 0
            buy_dates = buy_signals[buy_signals == 1].index
            for date in buy_dates:
                if date in indicators.index:
                    try:
                        y_val = indicators.loc[date, f'SMA_{short_period}'] if strategy == 'SMA_Cross_Signal' else indicators.loc[date, f'EMA_{short_period}']
                        ax.scatter(date, y_val, color='green', marker='^', s=120, alpha=0.9, zorder=6)
                        buy_count += 1
                    except (KeyError, IndexError):
                        continue
            
            sell_dates = sell_signals[sell_signals == 1].index
            for date in sell_dates:
                if date in indicators.index:
                    try:
                        y_val = indicators.loc[date, f'SMA_{short_period}'] if strategy == 'SMA_Cross_Signal' else indicators.loc[date, f'EMA_{short_period}']
                        ax.scatter(date, y_val, color='red', marker='v', s=120, alpha=0.9, zorder=6)
                        sell_count += 1
                    except (KeyError, IndexError):
                        continue
            
            signal_info = f"Signals: {buy_count} Long, {sell_count} Short"
            title_parts = [f'{ticker} - {strategy}', signal_info]
            ax.set_title('\n'.join(title_parts), fontsize=10, weight='bold')
            ax.set_ylabel('Price', fontsize=10)
            ax.legend(fontsize=12, loc='upper left')
            ax.grid(True, alpha=0.3, zorder=0)
            ax.tick_params(axis='x', rotation=45, labelsize=12)
            
            if not training_data.empty and not test_data.empty:
                ax.axvline(x=training_data.index[-1], color='red', linestyle='--', 
                        alpha=0.7, linewidth=1.5, label='Train/Test Split', zorder=2)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting {ticker}\n{str(e)}', ha='center', va='center', 
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Hide unused subplots
    for idx in range(len(selected_tickers), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return full_technical_indicators, full_trading_signals

def plot_portfolio_performance(optimization_summary, cols=2, rows=1):
    """Plot portfolio performance analysis with descriptive statistics table and risk-return profile"""
    stats_data = []
    metrics = ['CAGR', 'Volatility', 'Best_Sharpe', 'Best_Sortino', 'Max_Drawdown']
    
    for metric in metrics:
        if metric in optimization_summary.columns:
            col_data = optimization_summary[metric]
            display_name = metric.replace('_', ' ')
            if metric == 'Best_Sharpe':
                display_name = 'Sharpe Ratio'
            elif metric == 'Best_Sortino':
                display_name = 'Sortino Ratio'
            
            stats_data.append({
                'Metric': display_name,
                'Mean': col_data.mean(),
                'Min': col_data.min(),
                'Max': col_data.max(),
                'Count': col_data.count()
            })
    
    stats_df = pd.DataFrame(stats_data)
    
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    # Descriptive Statistics Table
    ax1 = axes[0]
    ax1.axis('tight')
    ax1.axis('off')
    
    asset_count = len(optimization_summary)
    ax1.set_title(f'Portfolio Performance Statistics\nAssets: {asset_count}', 
                  fontsize=14, weight='bold', pad=20)

    table_data = []
    for _, row in stats_df.iterrows():
        metric = row['Metric']
        if metric in ['CAGR', 'Volatility', 'Max Drawdown']:
            formatted_row = [metric, f"{row['Mean']:.2%}", f"{row['Min']:.2%}", f"{row['Max']:.2%}"]
        else:
            formatted_row = [metric, f"{row['Mean']:.3f}", f"{row['Min']:.3f}", f"{row['Max']:.3f}"]
        table_data.append(formatted_row)
    
    table = ax1.table(
        cellText=table_data,
        colLabels=['Metric', 'Mean', 'Min', 'Max'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    num_table_columns = 4
    for i in range(num_table_columns):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i, (_, row) in enumerate(stats_df.iterrows(), 1):
        metric = row['Metric']
        if metric in ['Sharpe Ratio', 'Sortino Ratio']:
            color = '#E8F5E8'
        elif metric == 'CAGR':
            color = '#FFF9C4'
        elif metric in ['Volatility', 'Max Drawdown']:
            color = '#FFEBEE'
        else:
            color = 'white'
        
        for j in range(num_table_columns):
            table[(i, j)].set_facecolor(color)
    
    # Risk-Return Profile
    if len(axes) > 1:
        ax2 = axes[1]
        if all(col in optimization_summary.columns for col in ['Volatility', 'CAGR', 'Best_Sortino']):
            scatter = ax2.scatter(optimization_summary['Volatility'] * 100, 
                                optimization_summary['CAGR'] * 100,
                                c=optimization_summary['Best_Sortino'], 
                                cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black')
            
            ax2.set_title('Risk-Return Profile\n(Color = Sortino Ratio)', fontsize=14, weight='bold')
            ax2.set_xlabel('Volatility (%)')
            ax2.set_ylabel('CAGR (%)')
            ax2.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Sortino Ratio', rotation=270, labelpad=20)
            
            if 'Ticker' in optimization_summary.columns:
                for i, ticker in enumerate(optimization_summary['Ticker']):
                    ax2.annotate(ticker, 
                               (optimization_summary['Volatility'].iloc[i] * 100, 
                                optimization_summary['CAGR'].iloc[i] * 100),
                               xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Risk-Return Profile\nRequires: Volatility, CAGR, Best_Sortino columns',
                    ha='center', va='center', transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    for idx in range(2, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return stats_df

def create_overfitting_analysis_report(optimization_results, test_set):
    """
    Create a comprehensive overfitting analysis report
    
    Args:
        optimization_results: Results from training set optimization
        test_set: Out-of-sample test data
    
    Returns:
        dict: Comprehensive overfitting analysis results
    """
    
    # Initialize analysis results
    analysis = {
        'summary': {},
        'detailed_results': {},
        'statistical_tests': {},
        'recommendations': []
    }
    
    try:
        # Validate inputs
        if not optimization_results or not isinstance(optimization_results, dict):
            raise ValueError("Invalid optimization_results - must be a non-empty dictionary")
        
        if test_set is None or test_set.empty:
            raise ValueError("Invalid test_set - must be a non-empty DataFrame")
        
        if 'selected_tickers' not in optimization_results or not optimization_results['selected_tickers']:
            raise ValueError("No selected tickers found in optimization results")
        
        # Extract in-sample metrics
        in_sample_metrics = _extract_in_sample_metrics(optimization_results)
        
        # Calculate out-of-sample metrics
        out_sample_metrics = _calculate_out_sample_performance(optimization_results, test_set)
        
        # Perform statistical tests
        statistical_results = _perform_overfitting_tests(in_sample_metrics, out_sample_metrics)
        
        # Summary statistics
        analysis['summary'] = {
            'num_strategies_tested': len(optimization_results['selected_tickers']),
            'in_sample_mean_sharpe': np.mean(in_sample_metrics['sharpe_ratios']) if in_sample_metrics['sharpe_ratios'] else 0,
            'out_sample_mean_sharpe': np.mean(out_sample_metrics['sharpe_ratios']) if out_sample_metrics['sharpe_ratios'] else 0,
            'in_sample_mean_sortino': np.mean(in_sample_metrics['sortino_ratios']) if in_sample_metrics['sortino_ratios'] else 0,
            'out_sample_mean_sortino': np.mean(out_sample_metrics['sortino_ratios']) if out_sample_metrics['sortino_ratios'] else 0,
        }
        
        # Statistical test results
        analysis['statistical_tests'] = statistical_results
        
        # Generate recommendations based on results
        analysis['recommendations'] = _generate_overfitting_recommendations(statistical_results)
        
        # Detailed breakdown by ticker
        analysis['detailed_results'] = _create_detailed_ticker_analysis(
            optimization_results, test_set, in_sample_metrics, out_sample_metrics
        )
        
    except Exception as e:
        print(f"⚠️ Error in overfitting analysis: {str(e)}")
        analysis['error'] = f"Error in overfitting analysis: {str(e)}"
        # Return fallback analysis
        analysis.update({
            'summary': {
                'num_strategies_tested': 0,
                'in_sample_mean_sharpe': 0,
                'out_sample_mean_sharpe': 0,
                'in_sample_mean_sortino': 0,
                'out_sample_mean_sortino': 0
            },
            'statistical_tests': {
                'sharpe_t_stat': 0.0,
                'sharpe_p_value': 1.0,
                'sortino_t_stat': 0.0,
                'sortino_p_value': 1.0,
                'is_overfitted': True
            },
            'recommendations': ["⚠️ Analysis failed - manual review required"],
            'detailed_results': {}
        })
    
    return analysis


def _extract_in_sample_metrics(optimization_results):
    """Extract in-sample performance metrics from optimization results"""
    in_sample_metrics = {
        'sharpe_ratios': [],
        'sortino_ratios': [],
        'cagr_values': [],
        'max_drawdowns': [],
        'volatilities': []
    }
    
    try:
        if 'all_optimization_results' not in optimization_results:
            print("⚠️ No all_optimization_results found - using empty metrics")
            return in_sample_metrics
        
        for ticker, results in optimization_results['all_optimization_results'].items():
            if not isinstance(results, dict) or 'best_params' not in results:
                print(f"⚠️ Invalid results structure for {ticker}")
                continue
                
            best_params = results['best_params']
            
            # Safely extract metrics with defaults
            in_sample_metrics['sharpe_ratios'].append(best_params.get('sharpe_ratio', 0.0))
            in_sample_metrics['sortino_ratios'].append(best_params.get('sortino_ratio', 0.0))
            in_sample_metrics['cagr_values'].append(best_params.get('cagr', 0.0))
            in_sample_metrics['max_drawdowns'].append(best_params.get('max_drawdown', 0.0))
            in_sample_metrics['volatilities'].append(best_params.get('volatility', 0.0))
    
    except Exception as e:
        print(f"⚠️ Error extracting in-sample metrics: {e}")
    
    return in_sample_metrics


def _calculate_out_sample_performance(optimization_results, test_set):
    """Calculate out-of-sample performance using optimized parameters"""
    out_sample_metrics = {
        'sharpe_ratios': [],
        'sortino_ratios': [],
        'cagr_values': [],
        'max_drawdowns': [],
        'volatilities': []
    }
    
    try:
        if 'selected_tickers' not in optimization_results:
            print("⚠️ No selected_tickers found")
            return out_sample_metrics
        
        for ticker in optimization_results['selected_tickers']:
            try:
                if ticker not in optimization_results.get('all_optimization_results', {}):
                    print(f"⚠️ No optimization results for {ticker}")
                    continue
                    
                best_params = optimization_results['all_optimization_results'][ticker]['best_params']
                
                # Get test data for this ticker
                if ticker not in test_set.columns:
                    print(f"⚠️ {ticker} not in test set columns")
                    continue
                    
                ticker_test_data = test_set[ticker].dropna()
                
                if len(ticker_test_data) < 10:
                    print(f"⚠️ Insufficient test data for {ticker} ({len(ticker_test_data)} rows)")
                    continue
                
                # Apply the optimized strategy to test data
                strategy_returns = _apply_strategy_to_test_data(ticker_test_data, best_params)
                
                if strategy_returns is not None and len(strategy_returns) > 0:
                    test_metrics = calculate_performance_metrics(strategy_returns)
                    
                    if test_metrics:
                        out_sample_metrics['sharpe_ratios'].append(test_metrics['sharpe_ratio'])
                        out_sample_metrics['sortino_ratios'].append(test_metrics['sortino_ratio'])
                        out_sample_metrics['cagr_values'].append(test_metrics['cagr'])
                        out_sample_metrics['max_drawdowns'].append(test_metrics['max_drawdown'])
                        out_sample_metrics['volatilities'].append(test_metrics['volatility'])
                    else:
                        print(f"⚠️ Could not calculate test metrics for {ticker}")
                else:
                    print(f"⚠️ Could not apply strategy to test data for {ticker}")
                    
            except Exception as e:
                print(f"⚠️ Error processing {ticker}: {e}")
                continue
    
    except Exception as e:
        print(f"⚠️ Error calculating out-sample performance: {e}")
    
    return out_sample_metrics


def _apply_strategy_to_test_data(ticker_data, best_params):
    """Improved strategy application with better error handling"""
    try:
        if not isinstance(best_params, dict):
            print("⚠️ Invalid best_params - must be dictionary")
            return None
        
        strategy_type = best_params.get('strategy_type', 'SMA_Cross_Signal')
        short_period = best_params.get('short_period', 20)
        long_period = best_params.get('long_period', 50)
        
        # Ensure we have enough data
        min_required = max(short_period, long_period) + 50  # More buffer
        if len(ticker_data) < min_required:
            print(f"⚠️ Insufficient data for periods {short_period}/{long_period} (need {min_required}, have {len(ticker_data)})")
            return None
        
        # Apply strategy
        if strategy_type == 'SMA_Cross_Signal':
            ma_short = ticker_data.rolling(short_period).mean()
            ma_long = ticker_data.rolling(long_period).mean()
        else:  # EMA_Cross_Signal
            ma_short = ticker_data.ewm(span=short_period).mean()
            ma_long = ticker_data.ewm(span=long_period).mean()
        
        # Generate signals (more conservative)
        signals = np.where(ma_short > ma_long, 1, -1)
        
        # Calculate returns
        returns = ticker_data.pct_change().fillna(0)
        strategy_returns = (returns * pd.Series(signals, index=returns.index).shift(1)).fillna(0)
        
        # Filter out invalid returns more aggressively
        valid_returns = strategy_returns.values
        valid_returns = valid_returns[~np.isnan(valid_returns)]
        valid_returns = valid_returns[np.isfinite(valid_returns)]
        valid_returns = valid_returns[np.abs(valid_returns) < 0.5]  # Remove extreme outliers
        
        return valid_returns if len(valid_returns) > 20 else None  # Increased minimum
        
    except Exception as e:
        print(f"⚠️ Error applying strategy: {e}")
        return None

def _perform_overfitting_tests(in_sample_metrics, out_sample_metrics):
    """Perform statistical tests to detect overfitting"""
    results = {
        'sharpe_t_stat': 0.0,
        'sharpe_p_value': 1.0,
        'sortino_t_stat': 0.0,
        'sortino_p_value': 1.0,
        'is_overfitted': True,
        'sharpe_degradation': 0.0,
        'sortino_degradation': 0.0
    }
    
    try:
        # Sharpe ratio t-test
        if (len(in_sample_metrics['sharpe_ratios']) > 0 and 
            len(out_sample_metrics['sharpe_ratios']) > 0):
            
            in_sharpe = [x for x in in_sample_metrics['sharpe_ratios'] if np.isfinite(x)]
            out_sharpe = [x for x in out_sample_metrics['sharpe_ratios'] if np.isfinite(x)]
            
            if len(in_sharpe) > 0 and len(out_sharpe) > 0:
                try:
                    sharpe_t_stat, sharpe_p_value = stats.ttest_ind(
                        in_sharpe, out_sharpe, alternative='greater'
                    )
                    
                    # Ensure values are numeric and finite
                    if np.isfinite(sharpe_t_stat) and np.isfinite(sharpe_p_value):
                        results['sharpe_t_stat'] = float(sharpe_t_stat)
                        results['sharpe_p_value'] = float(sharpe_p_value)
                except Exception as e:
                    print(f"⚠️ Error in Sharpe t-test: {e}")
        
        # Sortino ratio t-test
        if (len(in_sample_metrics['sortino_ratios']) > 0 and 
            len(out_sample_metrics['sortino_ratios']) > 0):
            
            in_sortino = [x for x in in_sample_metrics['sortino_ratios'] if np.isfinite(x)]
            out_sortino = [x for x in out_sample_metrics['sortino_ratios'] if np.isfinite(x)]
            
            if len(in_sortino) > 0 and len(out_sortino) > 0:
                try:
                    sortino_t_stat, sortino_p_value = stats.ttest_ind(
                        in_sortino, out_sortino, alternative='greater'
                    )
                    
                    # Ensure values are numeric and finite
                    if np.isfinite(sortino_t_stat) and np.isfinite(sortino_p_value):
                        results['sortino_t_stat'] = float(sortino_t_stat)
                        results['sortino_p_value'] = float(sortino_p_value)
                except Exception as e:
                    print(f"⚠️ Error in Sortino t-test: {e}")
        
        # Determine if strategy is overfitted
        results['is_overfitted'] = (
            results.get('sharpe_p_value', 1) < 0.05 or 
            results.get('sortino_p_value', 1) < 0.05
        )
        
        # Calculate degradation metrics
        if (len(in_sample_metrics['sharpe_ratios']) > 0 and 
            len(out_sample_metrics['sharpe_ratios']) > 0):
            
            in_sample_mean_sharpe = np.mean([x for x in in_sample_metrics['sharpe_ratios'] if np.isfinite(x)])
            out_sample_mean_sharpe = np.mean([x for x in out_sample_metrics['sharpe_ratios'] if np.isfinite(x)])
            
            if np.isfinite(in_sample_mean_sharpe) and np.isfinite(out_sample_mean_sharpe) and abs(in_sample_mean_sharpe) > 1e-10:
                results['sharpe_degradation'] = float(
                    (in_sample_mean_sharpe - out_sample_mean_sharpe) / abs(in_sample_mean_sharpe)
                )
        
        if (len(in_sample_metrics['sortino_ratios']) > 0 and 
            len(out_sample_metrics['sortino_ratios']) > 0):
            
            in_sample_mean_sortino = np.mean([x for x in in_sample_metrics['sortino_ratios'] if np.isfinite(x)])
            out_sample_mean_sortino = np.mean([x for x in out_sample_metrics['sortino_ratios'] if np.isfinite(x)])
            
            if np.isfinite(in_sample_mean_sortino) and np.isfinite(out_sample_mean_sortino) and abs(in_sample_mean_sortino) > 1e-10:
                results['sortino_degradation'] = float(
                    (in_sample_mean_sortino - out_sample_mean_sortino) / abs(in_sample_mean_sortino)
                )
    
    except Exception as e:
        print(f"⚠️ Error in statistical tests: {e}")
    
    return results


def _generate_overfitting_recommendations(statistical_results):
    """Generate recommendations based on overfitting analysis"""
    recommendations = []
    
    try:
        if statistical_results.get('is_overfitted', False):
            recommendations.append(
                "⚠️ OVERFITTING DETECTED: The strategy shows significant performance "
                "degradation from in-sample to out-of-sample testing."
            )
            
            # Specific recommendations based on degradation levels
            sharpe_deg = statistical_results.get('sharpe_degradation', 0)
            if sharpe_deg > 0.3:
                recommendations.append(
                    f"📉 Severe Sharpe ratio degradation ({sharpe_deg:.1%}). "
                    "Consider simplifying the strategy or using fewer parameters."
                )
            elif sharpe_deg > 0.1:
                recommendations.append(
                    f"📉 Moderate Sharpe ratio degradation ({sharpe_deg:.1%}). "
                    "Consider walk-forward optimization or parameter constraints."
                )
            
            recommendations.append(
                "🔄 Recommended actions:\n"
                "  • Use walk-forward optimization\n"
                "  • Reduce parameter search space\n"
                "  • Implement parameter stability constraints\n"
                "  • Consider ensemble methods\n"
                "  • Increase out-of-sample period"
            )
        else:
            recommendations.append(
                "✅ NO SIGNIFICANT OVERFITTING: The strategy maintains reasonable "
                "performance from in-sample to out-of-sample testing."
            )
            
            recommendations.append(
                "💡 Continue monitoring:\n"
                "  • Track performance over time\n"
                "  • Consider additional validation periods\n"
                "  • Monitor parameter stability"
            )
    
    except Exception as e:
        print(f"⚠️ Error generating recommendations: {e}")
        recommendations = ["⚠️ Could not generate recommendations - manual review required"]
    
    return recommendations


def _create_detailed_ticker_analysis(optimization_results, test_set, in_sample_metrics, out_sample_metrics):
    """Create detailed analysis for each ticker"""
    detailed_results = {}
    
    try:
        selected_tickers = optimization_results.get('selected_tickers', [])
        all_opt_results = optimization_results.get('all_optimization_results', {})
        
        for i, ticker in enumerate(selected_tickers):
            try:
                if (i < len(in_sample_metrics['sharpe_ratios']) and 
                    i < len(out_sample_metrics['sharpe_ratios']) and
                    ticker in all_opt_results):
                    
                    in_sample_sharpe = in_sample_metrics['sharpe_ratios'][i]
                    out_sample_sharpe = out_sample_metrics['sharpe_ratios'][i]
                    
                    # Calculate degradation safely
                    sharpe_degradation = 0.0
                    if abs(in_sample_sharpe) > 1e-10:
                        sharpe_degradation = (in_sample_sharpe - out_sample_sharpe) / abs(in_sample_sharpe)
                    
                    detailed_results[ticker] = {
                        'in_sample_sharpe': float(in_sample_sharpe),
                        'out_sample_sharpe': float(out_sample_sharpe),
                        'sharpe_degradation': float(sharpe_degradation) if np.isfinite(sharpe_degradation) else 0.0,
                        'in_sample_sortino': float(in_sample_metrics['sortino_ratios'][i]),
                        'out_sample_sortino': float(out_sample_metrics['sortino_ratios'][i]),
                        'strategy_params': all_opt_results[ticker]['best_params']
                    }
            except Exception as e:
                print(f"⚠️ Error analyzing {ticker}: {e}")
                continue
    
    except Exception as e:
        print(f"⚠️ Error creating detailed analysis: {e}")
    
    return detailed_results


def run_overfitting_analysis(optimization_results, test_set):
    """
    Main function to run overfitting analysis - use this in your notebook
    
    This function performs comprehensive statistical tests to detect overfitting
    by comparing in-sample vs out-of-sample performance using t-tests.
    
    Args:
        optimization_results: Results from training set optimization
        test_set: Out-of-sample test data
    
    Returns:
        dict: Comprehensive overfitting analysis results including:
            - summary: Basic statistics
            - statistical_tests: t-test results and p-values
            - recommendations: Actionable advice
            - detailed_results: Per-ticker breakdown
    """
    print("🔍 Running Overfitting Analysis...")
    print("=" * 50)
    
    # Validate inputs
    if not optimization_results:
        print("❌ Error: No optimization results provided")
        return {
            'error': 'No optimization results provided',
            'statistical_tests': {'is_overfitted': True, 'sharpe_p_value': 1.0, 'sortino_p_value': 1.0}
        }
    
    if test_set is None or test_set.empty:
        print("❌ Error: No test set provided")
        return {
            'error': 'No test set provided',
            'statistical_tests': {'is_overfitted': True, 'sharpe_p_value': 1.0, 'sortino_p_value': 1.0}
        }
    
    # Create comprehensive analysis
    analysis = create_overfitting_analysis_report(optimization_results, test_set)
    
    if 'error' in analysis:
        print(f"❌ Error in analysis: {analysis['error']}")
        return analysis
    
    # Display summary
    summary = analysis.get('summary', {})
    print(f"📊 SUMMARY STATISTICS")
    print(f"Strategies tested: {summary.get('num_strategies_tested', 0)}")
    print(f"In-sample mean Sharpe: {summary.get('in_sample_mean_sharpe', 0):.3f}")
    print(f"Out-sample mean Sharpe: {summary.get('out_sample_mean_sharpe', 0):.3f}")
    print(f"In-sample mean Sortino: {summary.get('in_sample_mean_sortino', 0):.3f}")
    print(f"Out-sample mean Sortino: {summary.get('out_sample_mean_sortino', 0):.3f}")
    
    # Display statistical test results
    stats_results = analysis.get('statistical_tests', {})
    print(f"\n📈 STATISTICAL TEST RESULTS")
    
    sharpe_t_stat = stats_results.get('sharpe_t_stat', 0.0)
    sharpe_p_value = stats_results.get('sharpe_p_value', 1.0)
    sortino_t_stat = stats_results.get('sortino_t_stat', 0.0)
    sortino_p_value = stats_results.get('sortino_p_value', 1.0)
    
    print(f"Sharpe t-statistic: {sharpe_t_stat:.3f}")
    print(f"Sharpe p-value: {sharpe_p_value:.3f}")
    print(f"Sortino t-statistic: {sortino_t_stat:.3f}")
    print(f"Sortino p-value: {sortino_p_value:.3f}")
    
    print(f"\n🎯 OVERFITTING STATUS: {'DETECTED' if stats_results.get('is_overfitted') else 'NOT DETECTED'}")
    
    if sharpe_p_value < 0.05:
        print(f"   ⚠️ Sharpe ratio shows significant degradation (p={sharpe_p_value:.3f})")
    else:
        print(f"   ✅ Sharpe ratio degradation not significant (p={sharpe_p_value:.3f})")
        
    if sortino_p_value < 0.05:
        print(f"   ⚠️ Sortino ratio shows significant degradation (p={sortino_p_value:.3f})")
    else:
        print(f"   ✅ Sortino ratio degradation not significant (p={sortino_p_value:.3f})")
    
    # Display recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in analysis.get('recommendations', []):
        print(f"  {rec}")
    
    # Display detailed results
    print(f"\n📋 DETAILED TICKER ANALYSIS:")
    detailed_results = analysis.get('detailed_results', {})
    if detailed_results:
        for ticker, details in detailed_results.items():
            print(f"\n{ticker}:")
            print(f"  In-sample Sharpe: {details.get('in_sample_sharpe', 0):.3f}")
            print(f"  Out-sample Sharpe: {details.get('out_sample_sharpe', 0):.3f}")
            print(f"  Degradation: {details.get('sharpe_degradation', 0):.1%}")
            strategy_params = details.get('strategy_params', {})
            print(f"  Strategy: {strategy_params.get('strategy_type', 'Unknown')}")
            print(f"  Periods: {strategy_params.get('short_period', 'N/A')}/{strategy_params.get('long_period', 'N/A')}")
    else:
        print("  No detailed results available")
    
    return analysis


def find_robust_strategies_with_stock_rotation(quotes_data, available_tickers, max_iterations=5, target_stocks=2, parameter_iterations=None):
    """Improved stock rotation with better data validation"""
    
    print("🎯 ENHANCED STRATEGY SELECTION WITH STOCK ROTATION & OVERFITTING PROTECTION")
    print("=" * 80)
    
    # Pre-filter tickers with sufficient data
    min_required_data = 1000  # Minimum trading days
    valid_tickers = []
    
    for ticker in available_tickers:
        ticker_data = quotes_data[ticker].dropna()
        if len(ticker_data) >= min_required_data:
            valid_tickers.append(ticker)
        else:
            print(f"⚠️ Excluding {ticker} - insufficient data ({len(ticker_data)} days)")
    
    print(f"📊 Valid tickers after filtering: {len(valid_tickers)}/{len(available_tickers)}")
    
    if len(valid_tickers) < target_stocks:
        print(f"❌ Not enough valid tickers ({len(valid_tickers)}) for target stocks ({target_stocks})")
        return None
    
    # Generate combinations from valid tickers only
    try:
        all_combinations = list(combinations(valid_tickers, target_stocks))
        if len(all_combinations) > 15:  # Reduced from 20 for faster processing
            random.seed(42)
            stock_combinations = random.sample(all_combinations, 15)
            print(f"Testing 15 random combinations out of {len(all_combinations)} possible")
        else:
            stock_combinations = all_combinations
            print(f"Testing all {len(stock_combinations)} combinations")
    except Exception as e:
        print(f"❌ Error generating combinations: {e}")
        return None
    
    # Default parameter configurations
    if parameter_iterations is None:
        parameter_iterations = [
            {'name': 'Conservative', 'ranges': {
                'SMA_Cross_Signal': {'short_periods': [20, 30, 40, 50], 'long_periods': [100, 120, 140, 160]},
                'EMA_Cross_Signal': {'short_periods': [12, 20, 26], 'long_periods': [50, 60, 70]}}},
            {'name': 'Ultra-Conservative', 'ranges': {
                'SMA_Cross_Signal': {'short_periods': [20, 50], 'long_periods': [100, 150]},
                'EMA_Cross_Signal': {'short_periods': [12, 26], 'long_periods': [50, 70]}}},
            {'name': 'Minimal-SMA', 'ranges': {
                'SMA_Cross_Signal': {'short_periods': [20], 'long_periods': [100]}}},
            {'name': 'Minimal-EMA', 'ranges': {
                'EMA_Cross_Signal': {'short_periods': [12], 'long_periods': [50]}}}
        ]
    
    def force_two_stock_selection_internal(successful_combos):
        """Force selection of exactly 2 stocks from successful combinations"""
        print("🔄 FORCING 2-STOCK PORTFOLIO SELECTION...")
        best_two_stock_result, best_sharpe = None, -999
        
        for combo_data in successful_combos:
            try:
                combination = combo_data.get('combination', [])
                param_config_name = combo_data.get('param_config', '')
                param_config_ranges = next((config['ranges'] for config in parameter_iterations if config['name'] == param_config_name), None)
                
                if param_config_ranges is None:
                    print(f"  ⚠️ Could not find parameter config '{param_config_name}', skipping...")
                    continue
                
                print(f"\n🧪 Re-testing 2-stock portfolio: {combination} ({param_config_name})")
                combo_quotes = quotes_data[combination].dropna()
                train_data = combo_quotes.iloc[:int(0.8 * len(combo_quotes))]
                
                two_stock_results = find_optimal_portfolio_with_parameter_optimization(
                    quotes=train_data, max_stocks=2, n_jobs=1, heatmap_metric='sharpe', parameter_ranges=param_config_ranges)
                
                if not two_stock_results:
                    print(f"  ⚠️ No optimization results for {combination}")
                    continue
                
                selected_count = len(two_stock_results.get('selected_tickers', []))
                portfolio_sharpe = two_stock_results.get('portfolio_stats', {}).get('sharpe_ratio', 0)
                
                print(f"  📊 Selected {selected_count} stocks: {two_stock_results.get('selected_tickers', [])}")
                print(f"  📈 Portfolio Sharpe: {portfolio_sharpe:.3f}")
                
                if selected_count == 2 and portfolio_sharpe > best_sharpe:
                    best_two_stock_result = {
                        'optimization_results': two_stock_results, 'combination': combination, 'config': param_config_name,
                        'sharpe': portfolio_sharpe, 'stability_ratio': combo_data.get('stability_ratio', 0), 'param_ranges': param_config_ranges}
                    best_sharpe = portfolio_sharpe
                    print(f"  ✅ New best 2-stock portfolio!")
            except Exception as e:
                print(f"  ❌ Error testing {combination}: {e}")
        return best_two_stock_result
    
    best_results, best_stability, all_attempts, successful_two_stock_combos = None, 0, [], []
    
    print(f"📊 Available tickers: {available_tickers}")
    print(f"Creating combinations of {target_stocks} stocks...")
    
    # Generate stock combinations (limit to 20 for performance)
    try:
        all_combinations = list(combinations(available_tickers, target_stocks))
        if len(all_combinations) > 20:
            random.seed(42)
            stock_combinations = random.sample(all_combinations, 20)
            print(f"Testing 20 random combinations out of {len(all_combinations)} possible")
        else:
            stock_combinations = all_combinations
            print(f"Testing all {len(stock_combinations)} combinations")
    except Exception as e:
        print(f"❌ Error generating combinations: {e}")
        return None
    
    # Main optimization loop
    for combo_idx, stock_combo in enumerate(stock_combinations):
        print(f"\n🔄 STOCK COMBINATION {combo_idx + 1}/{len(stock_combinations)}: {list(stock_combo)}")
        print("-" * 60)
        
        try:
            combo_quotes = quotes_data[list(stock_combo)].dropna()
            if len(combo_quotes) < 500:
                print(f"⚠️ Insufficient data for {stock_combo} ({len(combo_quotes)} rows)")
                continue
            
            combo_best_results, combo_best_stability = None, 0
            
            for iteration, param_config in enumerate(parameter_iterations):
                if iteration >= max_iterations: break
                
                print(f"  🔄 ITERATION {iteration + 1}: {param_config['name']} Parameters")
                
                attempt_record = {
                    'stock_combo': list(stock_combo), 'param_config': param_config['name'], 'is_overfitted': True,
                    'p_value': 1.0, 'train_sharpe': 0.0, 'test_sharpe': 0.0, 'selected_tickers': list(stock_combo),
                    'stability_ratio': 0.0, 'error': None
                }
                
                try:
                    total_rows = len(combo_quotes)
                    train_size = int(0.8 * total_rows)
                    training_set, test_set = combo_quotes.iloc[:train_size], combo_quotes.iloc[train_size:]
                    
                    optimization_results = find_optimal_portfolio_with_parameter_optimization(
                        quotes=training_set, max_stocks=target_stocks, n_jobs=1, heatmap_metric='sharpe', parameter_ranges=param_config['ranges'])
                    
                    if not optimization_results:
                        print(f"    ⚠️ No optimization results in iteration {iteration + 1}")
                        attempt_record['error'] = "No optimization results"
                        all_attempts.append(attempt_record)
                        continue
                    
                    if 'selected_tickers' in optimization_results:
                        attempt_record['selected_tickers'] = optimization_results['selected_tickers']
                    
                    overfitting_analysis = run_overfitting_analysis(optimization_results, test_set)
                    
                    if overfitting_analysis and 'statistical_tests' in overfitting_analysis:
                        is_overfitted = overfitting_analysis['statistical_tests'].get('is_overfitted', True)
                        sharpe_p_value = overfitting_analysis['statistical_tests'].get('sharpe_p_value', 1.0)
                        attempt_record.update({'is_overfitted': is_overfitted, 'p_value': sharpe_p_value})
                        
                        if 'summary' in overfitting_analysis:
                            summary = overfitting_analysis['summary']
                            attempt_record.update({
                                'train_sharpe': summary.get('in_sample_mean_sharpe', 0.0),
                                'test_sharpe': summary.get('out_sample_mean_sharpe', 0.0)})
                    else:
                        print("    ⚠️ Overfitting analysis failed - assuming overfitted")
                        is_overfitted, sharpe_p_value = True, 1.0
                    
                    print(f"    📊 Holdout Test: {'❌ OVERFITTED' if is_overfitted else '✅ STABLE'} (p={sharpe_p_value:.3f})")
                    
                    if not is_overfitted:
                        print("    🔄 Running walk-forward validation...")
                        wf_results = walk_forward_validation(combo_quotes, param_config['ranges'], validation_periods=3)
                        
                        if wf_results:
                            stable_periods = sum(1 for result in wf_results if not result.get('is_overfitted', True))
                            stability_ratio = stable_periods / len(wf_results)
                        else:
                            stability_ratio = 0
                        
                        print(f"    📈 Walk-Forward Stability: {stable_periods if wf_results else 0}/{len(wf_results) if wf_results else 0} ({stability_ratio:.1%})")
                        attempt_record.update({'stability_ratio': stability_ratio, 'walk_forward_results': wf_results})
                        
                        # Track successful 2-stock combinations
                        if stability_ratio >= 0.5 and len(optimization_results.get('selected_tickers', [])) == 2:
                            successful_two_stock_combos.append({
                                'combination': list(stock_combo), 'selected_tickers': optimization_results['selected_tickers'],
                                'stability_ratio': stability_ratio, 'param_config': param_config['name'],
                                'optimization_results': optimization_results, 'overfitting_analysis': overfitting_analysis, 'wf_results': wf_results})
                            print(f"    ✅ 2-STOCK SUCCESS! Added to successful combinations list.")
                        
                        # Check for validation success
                        if stability_ratio >= 0.6:
                            print("    ✅ STRATEGY VALIDATION PASSED!")
                            return {
                                'optimization_results': optimization_results, 'overfitting_analysis': overfitting_analysis,
                                'walk_forward_results': wf_results, 'stability_ratio': stability_ratio, 'iteration': iteration + 1,
                                'parameter_config': param_config['name'], 'stock_combination': list(stock_combo),
                                'validation_status': 'PASSED', 'all_attempts': all_attempts + [attempt_record],
                                'successful_two_stock_combos': successful_two_stock_combos}
                        elif stability_ratio > best_stability:
                            print(f"    📈 New best overall stability ratio: {stability_ratio:.1%}")
                            best_results = {
                                'optimization_results': optimization_results, 'overfitting_analysis': overfitting_analysis,
                                'walk_forward_results': wf_results, 'stability_ratio': stability_ratio, 'iteration': iteration + 1,
                                'parameter_config': param_config['name'], 'stock_combination': list(stock_combo),
                                'validation_status': 'BEST_AVAILABLE', 'all_attempts': all_attempts + [attempt_record],
                                'successful_two_stock_combos': successful_two_stock_combos}
                            best_stability = stability_ratio
                        
                        if stability_ratio > combo_best_stability:
                            combo_best_results = {
                                'optimization_results': optimization_results, 'overfitting_analysis': overfitting_analysis,
                                'walk_forward_results': wf_results, 'stability_ratio': stability_ratio, 'parameter_config': param_config['name']}
                            combo_best_stability = stability_ratio
                    else:
                        print("    ❌ Failed holdout test - trying simpler parameters...")
                        
                except Exception as e:
                    print(f"    ⚠️ Error in iteration {iteration + 1}: {e}")
                    attempt_record['error'] = str(e)
                
                all_attempts.append(attempt_record)
            
            # Summary for this stock combination
            if combo_best_results:
                print(f"  📊 Best for {stock_combo}: {combo_best_stability:.1%} stability ({combo_best_results.get('parameter_config', 'Unknown')})")
            else:
                print(f"  📊 No stable strategy found for {stock_combo}")
                
        except Exception as e:
            print(f"⚠️ Error processing stock combination {stock_combo}: {e}")
            continue
    
    # Print summary of all attempts
    print(f"\n📊 SUMMARY OF ALL ATTEMPTS:")
    print("-" * 60)
    sorted_attempts = sorted([a for a in all_attempts if a.get('stability_ratio', 0) > 0], key=lambda x: x.get('stability_ratio', 0), reverse=True)
    
    print("Top 10 best attempts:")
    for i, attempt in enumerate(sorted_attempts[:10]):
        status = "✅ STABLE" if not attempt.get('is_overfitted', True) else "❌ OVERFITTED"
        stability = attempt.get('stability_ratio', 0)
        error_text = f" (Error: {attempt['error'][:50]}...)" if attempt.get('error') else ""
        print(f"{i+1:2d}. {attempt['stock_combo']} | {attempt['param_config']} | {status} | Stability: {stability:.1%}{error_text}")
    
    # Return results based on priority
    if successful_two_stock_combos:
        print(f"\n🎯 FOUND {len(successful_two_stock_combos)} SUCCESSFUL 2-STOCK COMBINATIONS!")
        best_two_stock = max(successful_two_stock_combos, key=lambda x: x['stability_ratio'])
        print(f"Selected best 2-stock combination:")
        print(f"Combination: {best_two_stock['combination']}")
        print(f"Selected tickers: {best_two_stock['selected_tickers']}")
        print(f"Stability: {best_two_stock['stability_ratio']:.1%}")
        print(f"Config: {best_two_stock['param_config']}")
        
        return {
            'optimization_results': best_two_stock['optimization_results'], 'overfitting_analysis': best_two_stock['overfitting_analysis'],
            'walk_forward_results': best_two_stock['wf_results'], 'stability_ratio': best_two_stock['stability_ratio'],
            'parameter_config': best_two_stock['param_config'], 'stock_combination': best_two_stock['combination'],
            'validation_status': 'TWO_STOCK_SUCCESS', 'all_attempts': all_attempts, 'successful_two_stock_combos': successful_two_stock_combos}
    
    # Try forcing 2-stock selection if we have successful combos
    if best_results and len(successful_two_stock_combos) > 0:
        print("\n🎯 ATTEMPTING TO FORCE 2-STOCK SELECTION FROM SUCCESSFUL COMBINATIONS...")
        forced_result = force_two_stock_selection_internal(successful_two_stock_combos)
        
        if forced_result and len(forced_result.get('optimization_results', {}).get('selected_tickers', [])) == 2:
            print(f"\n✅ SUCCESSFULLY FORCED 2-STOCK SELECTION!")
            print(f"Combination: {forced_result['combination']}")
            print(f"Selected tickers: {forced_result['optimization_results']['selected_tickers']}")
            print(f"Portfolio Sharpe: {forced_result['sharpe']:.3f}")
            
            return {
                'optimization_results': forced_result['optimization_results'], 'overfitting_analysis': None,
                'walk_forward_results': None, 'stability_ratio': forced_result['stability_ratio'],
                'parameter_config': forced_result['config'], 'stock_combination': forced_result['combination'],
                'validation_status': 'FORCED_TWO_STOCK', 'all_attempts': all_attempts, 'successful_two_stock_combos': successful_two_stock_combos}
    
    # Return best results if any found
    if best_results:
        print(f"\n🎯 RETURNING BEST AVAILABLE STRATEGY:")
        print(f"Stock Combination: {best_results['stock_combination']}")
        print(f"Stability: {best_results['stability_ratio']:.1%}")
        print(f"Parameter Config: {best_results['parameter_config']}")
        return best_results
    else:
        print("\n❌ NO VIABLE STRATEGIES FOUND - Consider:")
        print("  • Expanding stock universe further\n  • Using different technical indicators\n  • Implementing ensemble methods\n  • Reducing target number of stocks")
        return None


def walk_forward_validation(quotes_data, parameter_ranges, is_overfitted=None, sharpe_p_value=None, 
                          overfitting_analysis=None, selected_tickers=None, risk_free_rate=None, 
                          validation_periods=4, is_full_analysis=False):
    """Improved walk-forward validation with better data handling"""
    
    if quotes_data is None or quotes_data.empty:
        print("⚠️ No quotes data provided for walk-forward validation")
        return []
    
    if not parameter_ranges:
        print("⚠️ No parameter ranges provided for walk-forward validation")
        return []
    
    total_length = len(quotes_data)
    
    # Ensure minimum periods have enough data
    min_required_per_period = 250  # Minimum trading days per period
    max_possible_periods = total_length // (min_required_per_period * 2)  # Need 2x for train/test
    
    if max_possible_periods < 2:
        print(f"⚠️ Insufficient data for walk-forward validation ({total_length} rows)")
        return []
    
    # Use smaller number of periods if data is limited
    validation_periods = min(validation_periods, max_possible_periods, 3)
    period_length = total_length // (validation_periods + 1)  # +1 to ensure we don't exceed data
    
    print(f"\n🔄 Running walk-forward validation with {validation_periods} periods (period length: {period_length} days)...")
    
    all_results = []
    
    for i in range(validation_periods):
        try:
            # Training window - use 80% of available data for each period
            train_start = i * (period_length // 2)
            train_end = train_start + int(period_length * 1.6)  # Larger training window
            
            # Test window - remaining 20%
            test_start = train_end
            test_end = min(test_start + (period_length // 2), total_length)
            
            # Ensure we have minimum data
            if (train_end - train_start) < min_required_per_period or (test_end - test_start) < 50:
                print(f"   ⚠️ Skipping period {i+1} - insufficient data")
                continue
                
            train_data = quotes_data.iloc[train_start:train_end]
            test_data = quotes_data.iloc[test_start:test_end]
            
            # Ensure we don't go beyond available data
            if test_end >= total_length:
                test_data = quotes_data.iloc[test_start:]
                if len(test_data) < 50:
                    print(f"   ⚠️ Skipping period {i+1} - test data too small")
                    continue
            
            print(f"   Period {i+1}: Train {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} days)")
            print(f"   Period {i+1}: Test {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} days)")

            # Rest of the function remains the same...
            period_results = find_optimal_portfolio_with_parameter_optimization(
                quotes=train_data,
                max_stocks=2,
                n_jobs=1,
                heatmap_metric='sharpe',
                parameter_ranges=parameter_ranges
            )
            
            if not period_results:
                print(f"   ⚠️ No optimization results for period {i+1}")
                all_results.append({
                    'period': i + 1,
                    'train_sharpe': 0.0,
                    'test_sharpe': 0.0,
                    'is_overfitted': True,
                    'p_value': 1.0,
                    'selected_tickers': [],
                    'error': 'No optimization results'
                })
                continue
            
            validation_analysis = run_overfitting_analysis(period_results, test_data)
            
            result = {
                'period': i + 1,
                'train_sharpe': 0.0,
                'test_sharpe': 0.0,
                'is_overfitted': True,
                'p_value': 1.0,
                'selected_tickers': period_results.get('selected_tickers', [])
            }
            
            if validation_analysis and 'summary' in validation_analysis:
                summary = validation_analysis['summary']
                result.update({
                    'train_sharpe': summary.get('in_sample_mean_sharpe', 0.0),
                    'test_sharpe': summary.get('out_sample_mean_sharpe', 0.0)
                })
            
            if validation_analysis and 'statistical_tests' in validation_analysis:
                tests = validation_analysis['statistical_tests']
                result.update({
                    'is_overfitted': tests.get('is_overfitted', True),
                    'p_value': tests.get('sharpe_p_value', 1.0)
                })
            
            all_results.append(result)
            
        except Exception as e:
            print(f"   ⚠️ Error in period {i+1}: {e}")
            all_results.append({
                'period': i + 1,
                'train_sharpe': 0.0,
                'test_sharpe': 0.0,
                'is_overfitted': True,
                'p_value': 1.0,
                'selected_tickers': [],
                'error': str(e)
            })
    
    # If basic call, return just the results list (backwards compatibility)
    if not is_full_analysis:
        return all_results
    
    # Full analysis mode - create comprehensive results
    validation_results = {
        'wf_results': all_results,
        'stability_ratio': 0.0,
        'final_status': '',
        'recommendations': [],
        'metrics': {}
    }
    
    # Analyze walk-forward results
    print("\n📊 WALK-FORWARD VALIDATION RESULTS:")
    print("-" * 50)
    stable_periods = 0
    total_periods = len(all_results)
    
    for result in all_results:
        if 'error' in result:
            print(f"Period {result['period']}: ❌ ERROR - {result['error']}")
            continue
            
        status = "❌ OVERFITTED" if result['is_overfitted'] else "✅ STABLE"
        print(f"Period {result['period']}: {status}")
        print(f"  Train Sharpe: {result['train_sharpe']:.3f}")
        print(f"  Test Sharpe: {result['test_sharpe']:.3f}")
        print(f"  Tickers: {result['selected_tickers']}")
        
        if not result['is_overfitted']:
            stable_periods += 1
    
    stability_ratio = stable_periods / total_periods if total_periods > 0 else 0
    validation_results['stability_ratio'] = stability_ratio
    
    print(f"\n📈 STABILITY SUMMARY:")
    print(f"Stable periods: {stable_periods}/{total_periods} ({stability_ratio:.1%})")
    
    if stability_ratio >= 0.6:
        print("✅ Strategy shows reasonable stability across time periods")
    else:
        print("⚠️ Strategy remains unstable - consider further simplification")
    
    # Final Strategy Validation Summary
    print("\n" + "="*60)
    print("📋 FINAL STRATEGY VALIDATION SUMMARY")
    print("="*60)
    
    # Strategy assessment
    if is_overfitted:
        validation_results['final_status'] = 'NEEDS_IMPROVEMENT'
        print(f"🎯 STRATEGY STATUS: NEEDS IMPROVEMENT")
        print(f"Current Status: OVERFITTED (p-value: {sharpe_p_value:.4f})")
        print(f"\n⚠️ REQUIRED ACTIONS:")
        recommendations = [
            "❌ Do not deploy current strategy",
            "🔧 Review walk-forward validation results", 
            "📉 Further reduce parameter complexity",
            "🎯 Focus on stable strategies only",
            "📊 Consider longer out-of-sample testing"
        ]
        for rec in recommendations:
            print(f"  {rec}")
        validation_results['recommendations'] = recommendations
    else:
        validation_results['final_status'] = 'READY_FOR_DEPLOYMENT'
        print(f"🎯 STRATEGY STATUS: READY FOR DEPLOYMENT")
        print(f"✅ Passed overfitting tests (p-value: {sharpe_p_value:.4f})")
        print(f"\n💡 DEPLOYMENT RECOMMENDATIONS:")
        recommendations = [
            "✅ Strategy appears robust",
            "📊 Monitor performance closely",
            "🎯 Consider paper trading first", 
            "📈 Set up regular re-optimization",
            "⚖️ Implement risk management"
        ]
        for rec in recommendations:
            print(f"  {rec}")
        validation_results['recommendations'] = recommendations
    
    # Store metrics safely
    try:
        validation_results['metrics'] = {
            'selected_tickers': selected_tickers if selected_tickers else [],
            'portfolio_size': len(selected_tickers) if selected_tickers else 0,
            'training_sharpe': overfitting_analysis.get('summary', {}).get('in_sample_mean_sharpe', 0.0) if overfitting_analysis else 0.0,
            'testing_sharpe': overfitting_analysis.get('summary', {}).get('out_sample_mean_sharpe', 0.0) if overfitting_analysis else 0.0,
            'risk_free_rate': risk_free_rate if risk_free_rate else 0.0,
            'p_value': sharpe_p_value if sharpe_p_value else 1.0,
            'is_overfitted': is_overfitted if is_overfitted is not None else True
        }
        
        print(f"\n📊 STRATEGY METRICS:")
        print(f"Selected tickers: {validation_results['metrics']['selected_tickers']}")
        print(f"Portfolio size: {validation_results['metrics']['portfolio_size']} stocks")
        print(f"Training Sharpe: {validation_results['metrics']['training_sharpe']:.3f}")
        print(f"Testing Sharpe: {validation_results['metrics']['testing_sharpe']:.3f}")
        print(f"Risk-free rate: {validation_results['metrics']['risk_free_rate']:.3%}")
    except Exception as e:
        print(f"⚠️ Error storing metrics: {e}")
        validation_results['metrics'] = {}
    
    return all_results if not is_overfitted else validation_results


def walk_forward_validation_check(quotes_data, parameter_ranges, is_overfitted, sharpe_p_value, 
                          overfitting_analysis, selected_tickers, risk_free_rate, 
                          validation_periods=4):
    """
    Implement walk-forward optimization to reduce overfitting and provide final strategy assessment
    
    Parameters:
    -----------
    quotes_data : pd.DataFrame
        Historical price data
    parameter_ranges : dict
        Parameter ranges for optimization
    is_overfitted : bool
        Whether the strategy is overfitted
    sharpe_p_value : float
        P-value from overfitting test
    overfitting_analysis : dict
        Results from overfitting analysis
    selected_tickers : list
        Selected ticker symbols
    risk_free_rate : float
        Risk-free rate for calculations
    validation_periods : int
        Number of periods for walk-forward validation
        
    Returns:
    --------
    dict : Comprehensive validation results and recommendations
    """
    
    # Initialize results dictionary
    validation_results = {
        'wf_results': [],
        'stability_ratio': 0.0,
        'final_status': '',
        'recommendations': [],
        'metrics': {}
    }
    
    # Run walk-forward validation if overfitting detected
    if is_overfitted:
        print("\n🔄 Running walk-forward validation (this may take a few minutes...)")
        
        total_length = len(quotes_data)
        period_length = total_length // validation_periods
        all_results = []
        
        for i in range(validation_periods - 1):  # Leave last period for final test
            # Training window
            train_start = i * period_length
            train_end = (i + 2) * period_length  # 2 periods for training
            
            # Test window
            test_start = train_end
            test_end = min(test_start + period_length, total_length)
            
            if test_end <= test_start:
                continue
                
            train_data = quotes_data.iloc[train_start:train_end]
            test_data = quotes_data.iloc[test_start:test_end]
            
            print(f"   Period {i+1}: Train {train_data.index[0]} to {train_data.index[-1]}")
            print(f"   Period {i+1}: Test {test_data.index[0]} to {test_data.index[-1]}")
            
            # Optimize on training period
            try:
                period_results = find_optimal_portfolio_with_parameter_optimization(
                    quotes=train_data,
                    max_stocks=2,
                    n_jobs=2,  # Reduced for stability
                    heatmap_metric='sharpe',
                    parameter_ranges=parameter_ranges
                )
                
                # Test on validation period
                validation_analysis = run_overfitting_analysis(period_results, test_data)
                
                all_results.append({
                    'period': i + 1,
                    'train_sharpe': validation_analysis['summary']['in_sample_mean_sharpe'],
                    'test_sharpe': validation_analysis['summary']['out_sample_mean_sharpe'],
                    'is_overfitted': validation_analysis['statistical_tests']['is_overfitted'],
                    'selected_tickers': period_results['selected_tickers']
                })
                
            except Exception as e:
                print(f"   ⚠️ Error in period {i+1}: {e}")
                continue
        
        validation_results['wf_results'] = all_results
        
        # Analyze walk-forward results
        print("\n📊 WALK-FORWARD VALIDATION RESULTS:")
        print("-" * 50)
        stable_periods = 0
        total_periods = len(all_results)
        
        for result in all_results:
            status = "❌ OVERFITTED" if result['is_overfitted'] else "✅ STABLE"
            print(f"Period {result['period']}: {status}")
            print(f"  Train Sharpe: {result['train_sharpe']:.3f}")
            print(f"  Test Sharpe: {result['test_sharpe']:.3f}")
            print(f"  Tickers: {result['selected_tickers']}")
            
            if not result['is_overfitted']:
                stable_periods += 1
        
        stability_ratio = stable_periods / total_periods if total_periods > 0 else 0
        validation_results['stability_ratio'] = stability_ratio
        
        print(f"\n📈 STABILITY SUMMARY:")
        print(f"Stable periods: {stable_periods}/{total_periods} ({stability_ratio:.1%})")
        
        if stability_ratio >= 0.6:
            print("✅ Strategy shows reasonable stability across time periods")
        else:
            print("⚠️ Strategy remains unstable - consider further simplification")
    else:
        print("\n✅ Skipping walk-forward validation - no overfitting detected")
        validation_results['stability_ratio'] = 1.0  # Assume stable if not overfitted
    
    # Final Strategy Validation Summary
    print("\n" + "="*60)
    print("📋 FINAL STRATEGY VALIDATION SUMMARY")
    print("="*60)
    
    # Strategy assessment
    if is_overfitted:
        validation_results['final_status'] = 'NEEDS_IMPROVEMENT'
        print(f"🎯 STRATEGY STATUS: NEEDS IMPROVEMENT")
        print(f"Current Status: OVERFITTED (p-value: {sharpe_p_value:.4f})")
        print(f"\n⚠️ REQUIRED ACTIONS:")
        recommendations = [
            "❌ Do not deploy current strategy",
            "🔧 Review walk-forward validation results", 
            "📉 Further reduce parameter complexity",
            "🎯 Focus on stable strategies only",
            "📊 Consider longer out-of-sample testing"
        ]
        for rec in recommendations:
            print(f"  {rec}")
        validation_results['recommendations'] = recommendations
    else:
        validation_results['final_status'] = 'READY_FOR_DEPLOYMENT'
        print(f"🎯 STRATEGY STATUS: READY FOR DEPLOYMENT")
        print(f"✅ Passed overfitting tests (p-value: {sharpe_p_value:.4f})")
        print(f"\n💡 DEPLOYMENT RECOMMENDATIONS:")
        recommendations = [
            "✅ Strategy appears robust",
            "📊 Monitor performance closely",
            "🎯 Consider paper trading first", 
            "📈 Set up regular re-optimization",
            "⚖️ Implement risk management"
        ]
        for rec in recommendations:
            print(f"  {rec}")
        validation_results['recommendations'] = recommendations
    
    # Store metrics
    validation_results['metrics'] = {
        'selected_tickers': selected_tickers,
        'portfolio_size': len(selected_tickers),
        'training_sharpe': overfitting_analysis['summary']['in_sample_mean_sharpe'],
        'testing_sharpe': overfitting_analysis['summary']['out_sample_mean_sharpe'],
        'risk_free_rate': risk_free_rate,
        'p_value': sharpe_p_value,
        'is_overfitted': is_overfitted
    }
    
    print(f"\n📊 STRATEGY METRICS:")
    print(f"Selected tickers: {selected_tickers}")
    print(f"Portfolio size: {len(selected_tickers)} stocks")
    print(f"Training Sharpe: {overfitting_analysis['summary']['in_sample_mean_sharpe']:.3f}")
    print(f"Testing Sharpe: {overfitting_analysis['summary']['out_sample_mean_sharpe']:.3f}")
    print(f"Risk-free rate: {risk_free_rate:.3%}")
    
    return validation_results

def run_complete_strategy_optimization(quotes, available_tickers, max_stocks=2, 
                                     custom_parameter_iterations=None, max_iterations=4):
    """
    Complete strategy optimization with automatic fallback handling
    
    Returns a standardized result structure regardless of success/failure paths
    """
    print("🔄 Running robust strategy optimization...")
    
    # Default parameters if none provided
    if custom_parameter_iterations is None:
        custom_parameter_iterations = [
            {'name': 'Conservative', 'ranges': {
                'SMA_Cross_Signal': {'short_periods': [20, 30, 50], 'long_periods': [100, 120, 150]},
                'EMA_Cross_Signal': {'short_periods': [12, 20, 26], 'long_periods': [50, 60, 70]}}}
        ]
    
    # Try main optimization
    robust_strategy_results = find_robust_strategies_with_stock_rotation(
        quotes, available_tickers, max_iterations=max_iterations, 
        target_stocks=max_stocks, parameter_iterations=custom_parameter_iterations
    )
    
    if robust_strategy_results:
        return _process_successful_results(robust_strategy_results)
    else:
        return _handle_fallback_strategy(quotes, available_tickers, max_stocks)

def _process_successful_results(robust_strategy_results):
    """Process successful optimization results"""
    optimization_results = robust_strategy_results['optimization_results']
    overfitting_analysis = robust_strategy_results.get('overfitting_analysis')
    
    # Extract key metrics
    result = {
        'optimization_results': optimization_results,
        'optimization_summary': optimization_results['optimization_summary'],
        'selected_tickers': optimization_results['selected_tickers'],
        'stability_ratio': robust_strategy_results.get('stability_ratio', 0.5),
        'is_overfitted': overfitting_analysis['statistical_tests'].get('is_overfitted', False) if overfitting_analysis else False,
        'sharpe_p_value': overfitting_analysis['statistical_tests'].get('sharpe_p_value', 1.0) if overfitting_analysis else 0.1,
        'strategy_type': 'ROBUST_MULTI_STOCK'
    }
    
    print(f"✅ Strategy found: {result['selected_tickers']}")
    display(result['optimization_summary'])
    
    # Plot heatmaps
    try:
        plot_optimization_heatmaps(optimization_results, metric='sharpe')
    except Exception as e:
        print(f"⚠️ Heatmap error: {e}")
    
    return result

def _handle_fallback_strategy(quotes, available_tickers, max_stocks):
    """Handle fallback to single-stock strategy"""
    print("⚠️ Testing single-stock fallback...")
    
    minimal_params = {'SMA_Cross_Signal': {'short_periods': [20], 'long_periods': [100]}}
    best_single_stock = None
    best_single_stability = 0
    
    for ticker in available_tickers[:10]:
        try:
            single_quotes = quotes[[ticker]].dropna()
            if len(single_quotes) < 500:
                continue
                
            train_size = int(0.8 * len(single_quotes))
            optimization_results = find_optimal_portfolio_with_parameter_optimization(
                quotes=single_quotes.iloc[:train_size], max_stocks=1, n_jobs=2, 
                parameter_ranges=minimal_params
            )
            
            overfitting_analysis = run_overfitting_analysis(optimization_results, single_quotes.iloc[train_size:])
            is_overfitted = overfitting_analysis['statistical_tests'].get('is_overfitted', True) if overfitting_analysis else True
            
            if not is_overfitted:
                wf_results = walk_forward_validation(single_quotes, minimal_params, validation_periods=3)
                stability_ratio = sum(1 for r in wf_results if not r.get('is_overfitted', True)) / len(wf_results) if wf_results else 0
                
                if stability_ratio > best_single_stability:
                    best_single_stock = {
                        'ticker': ticker, 'optimization_results': optimization_results,
                        'overfitting_analysis': overfitting_analysis, 'stability_ratio': stability_ratio
                    }
                    best_single_stability = stability_ratio
        except Exception:
            continue
    
    # Process results
    if best_single_stock and best_single_stability >= 0.5:
        result = {
            'optimization_results': best_single_stock['optimization_results'],
            'optimization_summary': best_single_stock['optimization_results']['optimization_summary'],
            'selected_tickers': [best_single_stock['ticker']],
            'stability_ratio': best_single_stability,
            'is_overfitted': False,
            'sharpe_p_value': best_single_stock['overfitting_analysis']['statistical_tests'].get('sharpe_p_value', 1.0),
            'strategy_type': 'SINGLE_STOCK_FALLBACK'
        }
        
        print(f"✅ Fallback strategy: {result['selected_tickers'][0]}")
        display(result['optimization_summary'])
        return result
    else:
        # Ultimate fallback
        return {
            'optimization_results': None,
            'optimization_summary': None,
            'selected_tickers': available_tickers[:max_stocks],
            'stability_ratio': 0.0,
            'is_overfitted': True,
            'sharpe_p_value': 0.001,
            'strategy_type': 'BASIC_FALLBACK'
        }