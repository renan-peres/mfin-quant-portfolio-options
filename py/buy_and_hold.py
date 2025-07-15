
"""
Portfolio Management Utilities Module
Provides comprehensive functions for portfolio analysis, optimization and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize, fmin
from datetime import datetime, timedelta
import os
from typing import List
import math
import plotly.graph_objects as go

#################################################
# PART 1: CORE PORTFOLIO CALCULATIONS
#################################################

def calculate_portfolio_stats(w_risky, er_risky, std_dev_risky, risk_free_rate):
    """Calculate portfolio statistics based on weight of risky assets"""
    w_rf = 1 - w_risky
    er_portfolio = w_risky * er_risky + w_rf * risk_free_rate
    std_dev_portfolio = w_risky * std_dev_risky
    return er_portfolio, std_dev_portfolio

def calculate_utility(er, std_dev, risk_aversion):
    """Calculate utility based on mean-variance utility function"""
    return er - 0.5 * risk_aversion * std_dev**2

def standard_deviation(weights, cov_matrix):
    """Calculate portfolio volatility (standard deviation)"""
    return np.sqrt(weights.T @ cov_matrix @ weights)
    
def expected_return(weights, returns_df):
    """Calculate annualized portfolio return"""
    return np.sum(returns_df.mean() * weights) * 252

def portfolio_variance(returns, weights=None):
    """Calculate the variance of a portfolio"""
    if weights is None:
        weights = np.ones(len(returns.columns)) / len(returns.columns)
    covariance_matrix = np.cov(returns.T)
    return np.dot(weights, np.dot(covariance_matrix, weights))

def sharpe_ratio(weights, returns_df, cov_matrix, risk_free_rate):
    """Calculate portfolio Sharpe ratio"""
    return (expected_return(weights, returns_df) - risk_free_rate) / standard_deviation(weights, cov_matrix)

def neg_sharpe_ratio(weights, returns_df, cov_matrix, risk_free_rate):
    """Negative Sharpe ratio for minimization"""
    return -sharpe_ratio(weights, returns_df, cov_matrix, risk_free_rate)

def portfolio_stats(weights, returns, cov_matrix):
    """Calculate portfolio statistics from weights, returns, and covariance matrix"""
    portfolio_return = np.dot(weights, returns)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_std

def calc_cov_matrix(assets):
    """Calculate covariance matrix from asset data dictionary"""
    cov = np.zeros((2, 2))
    cov[0, 0] = assets['Bond']['std'] ** 2
    cov[1, 1] = assets['Equity']['std'] ** 2
    cov[0, 1] = assets['Bond']['corr_equity'] * assets['Bond']['std'] * assets['Equity']['std']
    cov[1, 0] = cov[0, 1]
    return cov

#################################################
# PART 2: DATA PROCESSING FUNCTIONS
#################################################

def download_stock_data(symbols, start_date, end_date):
    """Download stock price data using yfinance"""
    return yf.download(symbols, start_date, end_date)['Adj Close']

def calculate_daily_returns(prices):
    """Calculate daily log returns from stock prices"""
    return np.log(prices / prices.shift(1))

def calculate_monthly_returns(daily_returns):
    """Calculate monthly returns from daily returns"""
    return np.exp(daily_returns.groupby(lambda date: date.month).sum()) - 1

def calculate_annual_returns(daily_returns):
    """Calculate annual returns from daily returns"""
    return np.exp(daily_returns.groupby(lambda date: date.year).sum()) - 1

def annual_returns(symbols, start_date, end_date):
    """Calculate annual returns directly from symbol list and date range"""
    df = yf.download(symbols, start_date, end_date)['Adj Close']
    log_rets = np.log(df / df.shift(1))
    return np.exp(log_rets.groupby(log_rets.index.year).sum()) - 1

#################################################
# PART 3: CAPITAL ALLOCATION FUNCTIONS
#################################################

def generate_allocation_table(er_risky, std_dev_risky, risk_free_rate, risk_aversion_index):
    """Generate table of allocations with different weights for capital allocation"""
    weights = np.arange(0, 1.1, 0.1)
    results = []
    
    for w in weights:
        er, std_dev = calculate_portfolio_stats(w, er_risky, std_dev_risky, risk_free_rate)
        utility = calculate_utility(er, std_dev, risk_aversion_index)
        er_indiff = utility + 0.5 * risk_aversion_index * std_dev**2
        
        results.append({
            'Weight': w,
            'Weight_Pct': f"{w:.0%}",
            'Expected_Return': er,
            'Standard_Deviation': std_dev,
            'Utility': utility,
            'ER_Indifference': er_indiff
        })
    
    results_df = pd.DataFrame(results)
    
    # Find max utility row
    max_utility_idx = results_df['Utility'].idxmax()
    
    # Generate chart data
    w_fine = np.linspace(0, 1, 100)
    chart_data = []
    
    for w in w_fine:
        er, std_dev = calculate_portfolio_stats(w, er_risky, std_dev_risky, risk_free_rate)
        utility = calculate_utility(er, std_dev, risk_aversion_index)
        chart_data.append({
            'Weight': w,
            'Expected_Return': er,
            'Standard_Deviation': std_dev,
            'Utility': utility
        })
    
    chart_df = pd.DataFrame(chart_data)
    
    return results_df, chart_df, max_utility_idx

def calculate_data_for_ojs(rf_rate, er_risky, std_dev_risky, market_view, risk_score):
    """Calculate all capital allocation data for OJS visualization"""
    try:
        # Type checking and conversion
        rf_rate = float(rf_rate)
        er_risky = float(er_risky)
        std_dev_risky = float(std_dev_risky)
        market_view = float(market_view)
        risk_score = float(risk_score)
        
        risk_aversion_index = market_view * (1 - risk_score/100)
        
        # Calculate optimal weight analytically
        optimal_weight = (er_risky - rf_rate) / (risk_aversion_index * std_dev_risky**2)
        optimal_weight = max(0, min(1, optimal_weight))
        
        # Calculate expected return and standard deviation for optimal weight
        er_optimal, std_dev_optimal = calculate_portfolio_stats(
            optimal_weight, er_risky, std_dev_risky, rf_rate
        )
        
        # Calculate utility for optimal weight
        utility_optimal = calculate_utility(er_optimal, std_dev_optimal, risk_aversion_index)
        
        # Calculate Sharpe ratio
        sharpe_ratio_val = (er_risky - rf_rate) / std_dev_risky
        
        # Create data tables
        results_df, chart_df, max_utility_idx = generate_allocation_table(
            er_risky, std_dev_risky, rf_rate, risk_aversion_index
        )
        
        # Convert to JSON for OJS
        allocation_data = results_df.to_dict(orient='records')
        chart_data = chart_df.to_dict(orient='records')
        
        return {
            'allocation_data': allocation_data,
            'chart_data': chart_data,
            'optimal_weight': float(optimal_weight),
            'er_optimal': float(er_optimal),
            'std_dev_optimal': float(std_dev_optimal),
            'utility_optimal': float(utility_optimal),
            'sharpe_ratio': float(sharpe_ratio_val),
            'max_utility_idx': int(max_utility_idx),
            'risk_aversion_index': float(risk_aversion_index)
        }
    except Exception as e:
        print(f"Error in calculate_data_for_ojs: {str(e)}")
        # Return default values in case of error
        return {
            'allocation_data': [],
            'chart_data': [],
            'optimal_weight': 0.5,
            'er_optimal': 0.043,
            'std_dev_optimal': 0,
            'utility_optimal': 0.043,
            'sharpe_ratio': 0,
            'max_utility_idx': 0,
            'risk_aversion_index': 10,
            'error': str(e)
        }

#################################################
# PART 4: PORTFOLIO OPTIMIZATION FUNCTIONS
#################################################

def min_variance_portfolio(returns, cov_matrix, risk_free_rate):
    """Calculate minimum variance portfolio"""
    n_assets = len(returns)
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_weights = np.ones(n_assets) / n_assets
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    result = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
                     init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result['success']:
        portfolio_return, portfolio_std = portfolio_stats(result['x'], returns, cov_matrix)
        return {
            'Return': portfolio_return,
            'Risk': portfolio_std,
            'Weights': result['x'].tolist(),
            'Sharpe': (portfolio_return - risk_free_rate) / portfolio_std  # Default T-Bill rate
        }
    else:
        return None

def max_sharpe_portfolio(returns, cov_matrix, risk_free_rate):
    """Calculate maximum Sharpe ratio portfolio"""
    n_assets = len(returns)
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_weights = np.ones(n_assets) / n_assets
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    def negative_sharpe_ratio(weights):
        p_ret, p_std = portfolio_stats(weights, returns, cov_matrix)
        return -(p_ret - risk_free_rate) / p_std
    
    result = minimize(negative_sharpe_ratio, init_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    if result['success']:
        portfolio_return, portfolio_std = portfolio_stats(result['x'], returns, cov_matrix)
        return {
            'Return': portfolio_return,
            'Risk': portfolio_std,
            'Weights': result['x'].tolist(),
            'Sharpe': (portfolio_return - risk_free_rate) / portfolio_std
        }
    else:
        return None

def optimal_complete_portfolio(max_sharpe_portfolio, risk_free_rate, risk_aversion):
    """Calculate optimal complete portfolio including risk-free asset"""
    # Calculate y (weight of risky portfolio)
    sharpe = max_sharpe_portfolio['Sharpe']
    expected_return = max_sharpe_portfolio['Return']
    std_dev = max_sharpe_portfolio['Risk']
    
    y = (expected_return - risk_free_rate) / (risk_aversion * std_dev**2)
    y = max(0, min(1, y))  # Constrain to [0,1]
    
    # Calculate weights in the complete portfolio
    bond_weight = y * max_sharpe_portfolio['Weights'][0]
    equity_weight = y * max_sharpe_portfolio['Weights'][1]
    rf_weight = 1 - y
    
    # Calculate expected return and std dev of complete portfolio
    er_complete = y * expected_return + (1-y) * risk_free_rate
    std_dev_complete = y * std_dev
    
    return {
        'y': y,
        'rf_weight': rf_weight,
        'bond_weight': bond_weight,
        'equity_weight': equity_weight,
        'er_complete': er_complete,
        'std_dev_complete': std_dev_complete,
        'sharpe': sharpe
    }

def optimize_portfolio(returns, initial_weights, rf_rate):
    """Optimize portfolio for maximum Sharpe ratio using fmin"""
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(initial_weights)))
    optimized = fmin(lambda x: -sharpe_ratio(x, returns, returns.cov() * 252, rf_rate), 
                     initial_weights, disp=False)
    return optimized

def calculate_min_variance_portfolio(
    tickers, 
    log_returns_df, 
    cov_matrix, 
    risk_free_rate,
    initial_weights,
    bounds,
    constraints,
    min_weight_threshold=0.0001,
    verbose=True
):
    """
    Calculate the minimum variance portfolio by minimizing standard deviation.
    
    Args:
        tickers: List of asset tickers
        log_returns_df: DataFrame of log returns
        cov_matrix: Covariance matrix
        risk_free_rate: Risk-free rate for Sharpe calculation
        initial_weights: Initial weights for optimization
        bounds: Bounds for optimization
        constraints: Constraints for optimization
        min_weight_threshold: Minimum weight to consider an asset part of the portfolio
        verbose: Whether to print results
        
    Returns:
        Dict with portfolio information
    """
    # Minimize standard deviation
    min_var_results = minimize(
        fun=standard_deviation,
        x0=initial_weights,
        args=(cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Calculate the minimum variance portfolio metrics
    min_var_weights = min_var_results.x
    min_var_return = expected_return(min_var_weights, log_returns_df)
    min_var_volatility = standard_deviation(min_var_weights, cov_matrix)
    min_var_sharpe = sharpe_ratio(min_var_weights, log_returns_df, cov_matrix, risk_free_rate)
    
    # Filter out tickers with negligible weights
    min_var_filtered_tickers = []
    min_var_filtered_weights = []
    
    for ticker, weight in zip(tickers, min_var_weights):
        if weight >= min_weight_threshold:
            min_var_filtered_tickers.append(ticker)
            min_var_filtered_weights.append(weight)
    
    # Sort tickers and weights in descending order of weights
    min_var_sorted_indices = np.argsort(min_var_filtered_weights)[::-1]
    min_var_sorted_tickers = [min_var_filtered_tickers[i] for i in min_var_sorted_indices]
    min_var_sorted_weights = [min_var_filtered_weights[i] for i in min_var_sorted_indices]
    
    if verbose:
        print("\nMinimum Variance Portfolio Weights:")
        for ticker, weight in zip(min_var_sorted_tickers, min_var_sorted_weights):
            print(f"{ticker}: {weight:.4f}")
        
        print(f"Number of assets in minimum variance portfolio: {len(min_var_filtered_tickers)}")
        
        print("\nMinimum Variance Portfolio:")
        print(f"Expected Annual Return: {min_var_return:.4f}")
        print(f"Expected Volatility: {min_var_volatility:.4f}")
        print(f"Sharpe Ratio: {min_var_sharpe:.4f}")
    
    return {
        "tickers": min_var_sorted_tickers,
        "weights": min_var_sorted_weights,
        "return": min_var_return,
        "volatility": min_var_volatility,
        "sharpe": min_var_sharpe,
        "raw_weights": min_var_weights
    }

def select_diversified_portfolio(
    tickers,
    log_returns_df,
    cov_matrix,
    fundamentals_df,
    risk_free_rate,
    initial_weights,
    bounds,
    constraints,
    min_assets=2,
    max_assets=10,
    max_asset_per_sector=1,
    max_allocation_weight=0.4,
    min_allocation_weight=0.05
):
    """
    Improved diversified portfolio selection that better maximizes Sharpe ratio
    while respecting constraints through mathematical optimization.
    """
    from scipy.optimize import minimize
    import warnings
    
    # Suppress scipy optimization warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")
        
        # Create sector mapping
        sector_map = {}
        unique_sectors = fundamentals_df['Sector'].unique()
        for i, sector in enumerate(unique_sectors):
            sector_tickers = fundamentals_df[fundamentals_df['Sector'] == sector].index.tolist()
            sector_map[i] = [j for j, ticker in enumerate(tickers) if ticker in sector_tickers]
        
        def neg_sharpe_with_constraints(weights):
            # Calculate portfolio metrics
            portfolio_return = expected_return(weights, log_returns_df)
            portfolio_vol = standard_deviation(weights, cov_matrix)
            
            if portfolio_vol == 0:
                return 1e10  # Avoid division by zero
                
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            
            # Add penalty for constraint violations
            penalty = 0
            
            # Count non-zero assets (using a small threshold)
            active_threshold = min_allocation_weight * 0.1  # 10% of minimum weight
            non_zero_assets = np.sum(weights > active_threshold)
            
            if non_zero_assets < min_assets:
                penalty += 10000 * (min_assets - non_zero_assets)
            if non_zero_assets > max_assets:
                penalty += 10000 * (non_zero_assets - max_assets)
            
            # Check sector constraints - much stricter penalty
            for sector_indices in sector_map.values():
                sector_assets = np.sum([weights[i] > active_threshold for i in sector_indices])
                if sector_assets > max_asset_per_sector:
                    penalty += 50000 * (sector_assets - max_asset_per_sector)  # Very high penalty
            
            return -sharpe + penalty
        
        # Set up optimization
        n_assets = len(tickers)
        opt_bounds = [(0, max_allocation_weight)] * n_assets
        
        # Multiple random starts to find global optimum
        best_result = None
        best_sharpe = -np.inf
        
        for seed in range(15):  # Try more starting points
            np.random.seed(seed)
            
            # Create a random starting point that respects sector constraints
            random_weights = np.zeros(n_assets)
            
            # Select one asset per sector randomly, up to min_assets
            selected_sectors = np.random.choice(len(sector_map), 
                                              min(min_assets, len(sector_map)), 
                                              replace=False)
            
            for sector_idx in selected_sectors:
                sector_indices = sector_map[sector_idx]
                if sector_indices:  # If sector has assets
                    selected_asset = np.random.choice(sector_indices)
                    random_weights[selected_asset] = 1.0
            
            # If we need more assets and have remaining sectors
            remaining_assets_needed = min_assets - len(selected_sectors)
            if remaining_assets_needed > 0:
                # Add more assets from unused sectors
                unused_sectors = [i for i in range(len(sector_map)) if i not in selected_sectors]
                if unused_sectors:
                    additional_sectors = np.random.choice(unused_sectors, 
                                                        min(remaining_assets_needed, len(unused_sectors)), 
                                                        replace=False)
                    for sector_idx in additional_sectors:
                        sector_indices = sector_map[sector_idx]
                        if sector_indices:
                            selected_asset = np.random.choice(sector_indices)
                            random_weights[selected_asset] = 1.0
            
            # Normalize weights
            if np.sum(random_weights) > 0:
                random_weights = random_weights / np.sum(random_weights)
            else:
                # Fallback: equal weights for first min_assets
                random_weights[:min_assets] = 1.0 / min_assets
            
            try:
                result = minimize(
                    neg_sharpe_with_constraints,
                    random_weights,
                    method='SLSQP',
                    bounds=opt_bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
                )
                
                if result.success:
                    # Check if solution actually respects constraints
                    weights = result.x
                    active_threshold = min_allocation_weight * 0.1
                    
                    # Verify sector constraints
                    sector_violation = False
                    for sector_indices in sector_map.values():
                        sector_assets = np.sum([weights[i] > active_threshold for i in sector_indices])
                        if sector_assets > max_asset_per_sector:
                            sector_violation = True
                            break
                    
                    if not sector_violation:  # Only accept if constraints are satisfied
                        current_sharpe = sharpe_ratio(weights, log_returns_df, cov_matrix, risk_free_rate)
                        if current_sharpe > best_sharpe:
                            best_sharpe = current_sharpe
                            best_result = result
                        
            except Exception as e:
                continue
        
        if best_result is None:
            print("Optimization failed to find a solution that respects all constraints")
            return None
        
        optimal_weights = best_result.x
        
        # Clean up the solution
        final_tickers = []
        final_weights = []
        
        active_threshold = min_allocation_weight * 0.5
        for i, weight in enumerate(optimal_weights):
            if weight >= active_threshold:
                final_tickers.append(tickers[i])
                final_weights.append(weight)
        
        # Renormalize weights
        final_weights = np.array(final_weights)
        if len(final_weights) > 0:
            final_weights = final_weights / np.sum(final_weights)
            
            # Ensure minimum weight constraint
            below_min = final_weights < min_allocation_weight
            if np.any(below_min):
                # Remove assets below minimum
                keep_indices = ~below_min
                final_tickers = [final_tickers[i] for i in range(len(final_tickers)) if keep_indices[i]]
                final_weights = final_weights[keep_indices]
                
                # Renormalize
                if len(final_weights) > 0:
                    final_weights = final_weights / np.sum(final_weights)
        
        # Calculate final portfolio metrics
        if len(final_tickers) > 0:
            final_weights_full = np.zeros(len(tickers))
            for i, ticker in enumerate(final_tickers):
                ticker_index = tickers.index(ticker)
                final_weights_full[ticker_index] = final_weights[i]
            
            portfolio_return = expected_return(final_weights_full, log_returns_df)
            portfolio_vol = standard_deviation(final_weights_full, cov_matrix)
            portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            
            # Final validation
            sectors_used = [fundamentals_df.loc[ticker, 'Sector'] for ticker in final_tickers]
            sector_counts = pd.Series(sectors_used).value_counts()
            
            print(f"Final portfolio:")
            print(f"Assets: {len(final_tickers)}")
            print(f"Sectors: {len(set(sectors_used))}")
            print(f"Max weight: {max(final_weights):.2%}")
            print(f"Min weight: {min(final_weights):.2%}")
            print(f"Sharpe ratio: {portfolio_sharpe:.4f}")
            
            # Check for constraint violations
            if any(count > max_asset_per_sector for count in sector_counts):
                print("WARNING: Sector constraint violated!")
                print("Sector distribution:", dict(sector_counts))
            
            for ticker, weight in zip(final_tickers, final_weights):
                sector = fundamentals_df.loc[ticker, 'Sector']
                print(f"{ticker}: {weight:.2%} ({sector})")
            
            return {
                "tickers": final_tickers,
                "weights": final_weights.tolist(),
                "return": portfolio_return,
                "volatility": portfolio_vol,
                "sharpe": portfolio_sharpe,
                "max_asset_per_sector": max_asset_per_sector,
                "min_assets": min_assets,
                "max_assets": max_assets,
                "min_allocation_weight": min_allocation_weight
            }
        
        return None

#################################################
# PART 5: ASSET CLASS ALLOCATION FUNCTIONS
#################################################

def generate_efficient_frontier(returns, cov_matrix, n_points=100):
    """Generate points along the efficient frontier"""
    target_returns = np.linspace(min(returns), max(returns), n_points)
    efficient_frontier = []
    
    n_assets = len(returns)
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_weights = np.ones(n_assets) / n_assets
    
    for target in target_returns:
        # Minimize volatility for each target return
        constraint = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # weights sum to 1
                      {'type': 'eq', 'fun': lambda w: np.dot(w, returns) - target})  # return = target
        
        result = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
                         init_weights, method='SLSQP', bounds=bounds, constraints=constraint)
        
        if result['success']:
            portfolio_return, portfolio_std = portfolio_stats(result['x'], returns, cov_matrix)
            efficient_frontier.append({
                'Return': portfolio_return,
                'Risk': portfolio_std,
                'Weights': result['x'].tolist()
            })
    
    return efficient_frontier

def calculate_allocations(bond_return, equity_return, rf_rate, 
                         bond_std, equity_std, correlation, risk_aversion=1):
    """Calculate portfolio allocations for asset class allocation"""
    # Update asset data
    assets = {
        'Bond': {'return': bond_return, 'std': bond_std, 'corr_equity': correlation},
        'Equity': {'return': equity_return, 'std': equity_std, 'corr_bond': correlation},
        'T-Bill': {'return': rf_rate, 'std': 0.001}
    }
    
    # Calculate covariance matrix
    cov_matrix = calc_cov_matrix(assets)
    returns = np.array([assets['Bond']['return'], assets['Equity']['return']])
    
    # Generate allocations for different weights
    weights_list = []
    for bond_weight in range(0, 101, 10):  # 0% to 100% in 10% increments
        equity_weight = 100 - bond_weight
        weights = np.array([bond_weight/100, equity_weight/100])
        
        er, std_dev = portfolio_stats(weights, returns, cov_matrix)
        sharpe = (er - rf_rate) / std_dev
        
        weights_list.append({
            'bond_weight': bond_weight,
            'equity_weight': equity_weight,
            'expected_return': er,
            'std_dev': std_dev,
            'sharpe': sharpe
        })
    
    # Calculate minimum variance portfolio
    min_var = min_variance_portfolio(returns, cov_matrix, rf_rate)
    
    # Calculate maximum Sharpe ratio portfolio
    max_sharpe = max_sharpe_portfolio(returns, cov_matrix, rf_rate)
    
    # Calculate optimal complete portfolio
    complete = optimal_complete_portfolio(max_sharpe, rf_rate, risk_aversion)
    
    # Calculate efficient frontier points
    frontier_points = generate_efficient_frontier(returns, cov_matrix)
    
    return {
        'allocations': weights_list,
        'min_variance': min_var,
        'max_sharpe': max_sharpe,
        'complete_portfolio': complete,
        'efficient_frontier': frontier_points
    }

#################################################
# PART 6: VISUALIZATION FUNCTIONS
#################################################

def generate_efficient_frontier_plot(
    tickers, log_returns_df, cov_matrix, risk_free_rate, 
    initial_weights, bounds, constraints, fundamentals_df,
    min_var_return, min_var_volatility, min_var_sharpe,
    optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio,
    filtered_tickers, filtered_weights, num_points=50, show_plot=True, fig_size=(12, 8)
):
    """
    Generate and plot efficient frontier with key portfolio points
    """
    # Define a function to minimize variance given a target return
    def min_variance_given_return(target_return):
        target_constraints = [
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'eq', 'fun': lambda weights: expected_return(weights, log_returns_df) - target_return}
        ]

        result = minimize(
            fun=standard_deviation,
            x0=initial_weights,
            args=(cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=target_constraints
        )

        if not result['success']:
            return None

        weights = result['x']
        volatility = standard_deviation(weights, cov_matrix)
        actual_return = expected_return(weights, log_returns_df)

        return {
            'weights': weights,
            'volatility': volatility,
            'return': actual_return
        }

    # Generate a range of target returns
    min_return = min_var_return * 0.8  # Start below min_var for complete frontier
    max_return = np.max(log_returns_df.mean() * 252) * 0.9  # Slightly less than max asset return
    target_returns = np.linspace(min_return, max_return, num_points)

    # Calculate efficient frontier portfolios
    efficient_portfolios = []
    for target in target_returns:
        portfolio = min_variance_given_return(target)
        if portfolio is not None:
            efficient_portfolios.append(portfolio)

    # Extract returns and volatilities for plotting
    ef_returns = [p['return'] for p in efficient_portfolios]
    ef_volatilities = [p['volatility'] for p in efficient_portfolios]

    # Calculate the pure mathematical max Sharpe ratio portfolio without sector constraints
    pure_optimized = minimize(
        neg_sharpe_ratio,
        initial_weights,
        args=(log_returns_df, cov_matrix, risk_free_rate),
        method='SLSQP',
        constraints=constraints,
        bounds=bounds
    )
    pure_optimal_weights = pure_optimized.x
    pure_optimal_return = expected_return(pure_optimal_weights, log_returns_df)
    pure_optimal_volatility = standard_deviation(pure_optimal_weights, cov_matrix)
    pure_optimal_sharpe = sharpe_ratio(pure_optimal_weights, log_returns_df, cov_matrix, risk_free_rate)

    # Create the figure
    fig = plt.figure(figsize=fig_size)
    
    # Plot the efficient frontier
    plt.plot(ef_volatilities, ef_returns, 'b-', linewidth=3, label='Efficient Frontier')

    # Plot individual assets
    asset_returns = [expected_return(np.eye(1, len(tickers), i)[0], log_returns_df) for i in range(len(tickers))]
    asset_volatilities = [standard_deviation(np.eye(1, len(tickers), i)[0], cov_matrix) for i in range(len(tickers))]
    plt.scatter(asset_volatilities, asset_returns, marker='o', s=100, color='darkblue', alpha=0.5, 
                label='Individual Assets')

    # Plot key portfolios
    plt.scatter(pure_optimal_volatility, pure_optimal_return, marker='*', s=300, color='orange', 
                label=f'Pure Max Sharpe Portfolio (SR: {pure_optimal_sharpe:.2f})')
    plt.scatter(optimal_portfolio_volatility, optimal_portfolio_return, marker='*', s=300, color='red',
                label=f'Constrained Portfolio (SR: {optimal_sharpe_ratio:.2f})')
    plt.scatter(min_var_volatility, min_var_return, marker='P', s=200, color='green',
                label=f'Min Variance Portfolio (SR: {min_var_sharpe:.2f})')

    # Add capital market lines
    plt.plot([0, pure_optimal_volatility*1.5],
             [risk_free_rate, risk_free_rate + pure_optimal_sharpe*pure_optimal_volatility*1.5],
             color='orange', linestyle='--', linewidth=2, label='True Capital Market Line')
    plt.plot([0, optimal_portfolio_volatility*1.5],
             [risk_free_rate, risk_free_rate + optimal_sharpe_ratio*optimal_portfolio_volatility*1.5],
             color='red', linestyle='--', linewidth=1, label='Constrained Portfolio Line')

    # Add labels and formatting
    plt.title('Efficient Frontier', fontsize=16)
    plt.xlabel('Expected Volatility (Standard Deviation)', fontsize=14)
    plt.ylabel('Expected Return', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=10)
    
    # Format axes as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Add annotations
    plt.annotate(f'Sharpe: {pure_optimal_sharpe:.2f}\nReturn: {pure_optimal_return:.2%}\nRisk: {pure_optimal_volatility:.2%}',
                 xy=(pure_optimal_volatility, pure_optimal_return),
                 xytext=(-120, -30),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='orange'),
                 bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.8))

    plt.annotate(f'Sharpe: {optimal_sharpe_ratio:.2f}\nReturn: {optimal_portfolio_return:.2%}\nRisk: {optimal_portfolio_volatility:.2%}',
                 xy=(optimal_portfolio_volatility, optimal_portfolio_return),
                 xytext=(20, 80),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='red'),
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

    plt.annotate(f'Sharpe: {min_var_sharpe:.2f}\nReturn: {min_var_return:.2%}\nRisk: {min_var_volatility:.2%}',
                 xy=(min_var_volatility, min_var_return),
                 xytext=(-80, -80),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='green'),
                 bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.5))

    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    # Calculate pure portfolio sector data
    pure_top_holdings = []
    pure_top_indices = np.argsort(pure_optimal_weights)[::-1][:5]
    for i in pure_top_indices:
        if pure_optimal_weights[i] >= 0.01:  # Only show positions > 1%
            ticker = tickers[i]
            sector = fundamentals_df.loc[ticker, 'Sector']
            pure_top_holdings.append({
                'ticker': ticker,
                'weight': pure_optimal_weights[i],
                'sector': sector
            })
            
    # Count number of sectors in pure mathematical portfolio
    pure_sectors = set()
    for i, w in enumerate(pure_optimal_weights):
        if w >= 0.01:  # Only count meaningful positions
            pure_sectors.add(fundamentals_df.loc[tickers[i], 'Sector'])
            
    # Prepare output
    result = {
        'efficient_frontier': {
            'returns': ef_returns,
            'volatilities': ef_volatilities
        },
        'pure_portfolio': {
            'weights': pure_optimal_weights,
            'return': pure_optimal_return,
            'volatility': pure_optimal_volatility,
            'sharpe': pure_optimal_sharpe,
            'top_holdings': pure_top_holdings,
            'unique_sectors': len(pure_sectors),
            'assets_above_1pct': sum(pure_optimal_weights >= 0.01)
        },
        'diversified_portfolio': {
            'tickers': filtered_tickers,
            'weights': filtered_weights,
            'return': optimal_portfolio_return,
            'volatility': optimal_portfolio_volatility,
            'sharpe': optimal_sharpe_ratio,
            'unique_sectors': len(set(fundamentals_df.loc[filtered_tickers, 'Sector'])),
            'max_weight': max(filtered_weights)
        },
        'figure': fig
    }
    
    return result

def plot_efficient_frontier_interactive(
    selected_tickers,
    log_returns_df,
    risk_free_rate,
    optimal_portfolio_return,
    optimal_portfolio_volatility,
    min_var_return,
    min_var_volatility,
    optimal_sharpe_ratio,
    min_var_sharpe,
    optimal_weights=None,
    min_var_weights=None,
    pure_optimal_return=None,
    pure_optimal_volatility=None,
    pure_optimal_sharpe=None,
    max_allocation_weight=0.4,
    num_points=100,
    show_plot=True
):
    """
    Plot an interactive efficient frontier using Plotly
    """
    selected_log_returns = log_returns_df[selected_tickers]
    selected_cov = selected_log_returns.cov() * 252
    n = len(selected_tickers)
    initial_weights = np.full(n, 1 / n)
    bounds = [(0, max_allocation_weight)] * n
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    def expected_return(weights):
        return np.sum(weights * selected_log_returns.mean()) * 252

    def standard_deviation(weights):
        return np.sqrt(weights.T @ selected_cov @ weights)

    def min_var_given_return(target_return):
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: expected_return(w) - target_return}
        ]
        res = minimize(
            fun=standard_deviation,
            x0=initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        if not res.success:
            return None
        port_return = expected_return(res.x)
        port_vol = standard_deviation(res.x)
        port_sharpe = (port_return - risk_free_rate) / port_vol
        tooltip = "<br>".join(
            f"{selected_tickers[i]}: {res.x[i]:.2%}"
            for i in range(n) if res.x[i] > 0.001
        )
        return {
            "return": port_return,
            "risk": port_vol,
            "sharpe": port_sharpe,
            "tooltip": tooltip,
        }

    # Efficient-frontier curve
    mean_returns = selected_log_returns.mean() * 252
    min_ret = min_var_return * 0.95
    max_ret = mean_returns.max() * 1.05
    target_returns = np.linspace(min_ret, max_ret, num_points)

    ef_points = [pt for r in target_returns if (pt := min_var_given_return(r))]
    if not ef_points:
        raise RuntimeError("No points could be computed for the efficient frontier.")

    ef_returns = [pt["return"] for pt in ef_points]
    ef_vols = [pt["risk"] for pt in ef_points]
    ef_sharpes = [pt["sharpe"] for pt in ef_points]
    ef_tooltips = [pt["tooltip"] for pt in ef_points]

    # Capital-Market Line
    x_start = math.floor(min_var_volatility * 20) / 20
    cml_slope = optimal_sharpe_ratio
    cml_x = [0, optimal_portfolio_volatility * 1.5]
    cml_y = [risk_free_rate + optimal_sharpe_ratio * x for x in cml_x]

    # Create Plotly figure
    fig = go.Figure()

    # Frontier scatter
    fig.add_trace(go.Scatter(
        x=ef_vols, y=ef_returns,
        mode="markers",
        marker=dict(
            size=10,
            color=ef_sharpes,
            colorscale="Viridis",
            showscale=False,
            line=dict(width=1, color="black")
        ),
        name="Frontier Portfolios",
        text=[f"Sharpe: {s:.2f}<br>{t}" for s, t in zip(ef_sharpes, ef_tooltips)],
        hovertemplate="Return: %{y:.2%}<br>Risk: %{x:.2%}<br>%{text}<extra></extra>",
    ))

    # Frontier line
    fig.add_trace(go.Scatter(
        x=ef_vols, y=ef_returns,
        mode="lines", line=dict(color="blue", width=3),
        name="Efficient Frontier"
    ))

    # CML
    fig.add_trace(go.Scatter(
        x=cml_x, y=cml_y,
        mode="lines", line=dict(color="red", width=2, dash="dash"),
        name="Capital Market Line",
        hoverinfo="skip"
    ))

    # Tooltips
    def tooltip_from_weights(wts):
        if wts is None:
            return ""
        
        # Safely generate tooltip text by ensuring index is in range for both lists
        valid_length = min(len(selected_tickers), len(wts))
        return "<br>".join(f"{selected_tickers[i]}: {wts[i]:.2%}"
                          for i in range(valid_length) if i < len(wts) and wts[i] > 0.001)

    # Optimal (tangency) portfolio
    opt_tooltip = tooltip_from_weights(optimal_weights) if optimal_weights is not None else ""
    fig.add_trace(go.Scatter(
        x=[optimal_portfolio_volatility], y=[optimal_portfolio_return],
        mode="markers", marker=dict(size=18, color="red",
                                    line=dict(width=2, color="black")),
        name="Optimal Portfolio",
        hovertemplate=f"Return: %{{y:.2%}}<br>Risk: %{{x:.2%}}<br>"
                      f"Sharpe: {optimal_sharpe_ratio:.2f}<br>{opt_tooltip}<extra></extra>"
    ))

    # Min-var portfolio
    mv_tooltip = tooltip_from_weights(min_var_weights) if min_var_weights is not None else ""
    fig.add_trace(go.Scatter(
        x=[min_var_volatility], y=[min_var_return],
        mode="markers", marker=dict(size=18, color="green",
                                    line=dict(width=2, color="black")),
        name="Min Variance Portfolio",
        hovertemplate=f"Return: %{{y:.2%}}<br>Risk: %{{x:.2%}}<br>"
                      f"Sharpe: {min_var_sharpe:.2f}<br>{mv_tooltip}<extra></extra>"
    ))

    # Annotations
    # Optional "pure" optimal
    if all(v is not None for v in
           (pure_optimal_return, pure_optimal_volatility, pure_optimal_sharpe)):
        fig.add_annotation(
            x=pure_optimal_volatility, y=pure_optimal_return,
            text=f"<b>Pure Optimal</b><br>"
                 f"Return: {pure_optimal_return:.2%}<br>"
                 f"Risk: {pure_optimal_volatility:.2%}<br>"
                 f"Sharpe: {pure_optimal_sharpe:.2f}<br>",
            font=dict(color="black"), 
            showarrow=True, arrowhead=2, arrowwidth=1, arrowcolor="orange",
            ax=-120, ay=-30,
            bgcolor="lightyellow", bordercolor="orange", opacity=0.8
        )

    # Tangency portfolio
    fig.add_annotation(
        x=optimal_portfolio_volatility, y=optimal_portfolio_return,
        text=f"<b>Optimal Portfolio</b><br>"
             f"Return: {optimal_portfolio_return:.2%}<br>"
             f"Risk: {optimal_portfolio_volatility:.2%}<br>"
             f"Sharpe: {optimal_sharpe_ratio:.2f}<br>",
        showarrow=True, arrowhead=2, arrowwidth=1, arrowcolor="red",
        ax=20, ay=80,
        bgcolor="yellow", bordercolor="red", opacity=0.6
    )

    # Minimum-variance portfolio
    fig.add_annotation(
        x=min_var_volatility, y=min_var_return,
        text=f"<b>Min Variance</b><br>"
             f"Return: {min_var_return:.2%}<br>"
             f"Risk: {min_var_volatility:.2%}<br>"
             f"Sharpe: {min_var_sharpe:.2f}<br>",
        showarrow=True, arrowhead=2, arrowwidth=1, arrowcolor="green",
        ax=-80, ay=-80,
        bgcolor="lightgreen", bordercolor="green", opacity=0.6
    )

    # Layout / axes
    all_x = ef_vols + [optimal_portfolio_volatility, min_var_volatility]
    all_y = ef_returns + [optimal_portfolio_return, min_var_return]
    x_min = max(0, min(all_x) - 0.01)
    x_max = max(all_x) + 0.01
    y_min = max(0, min(all_y) - 0.01)
    y_max = max(all_y) + 0.01

    # Updated layout for responsive full-screen display
    fig.update_layout(
        title="Efficient Frontier with Capital Market Line",
        title_x=0.5,
        xaxis_title="Standard Deviation (Risk)",
        yaxis_title="Return",
        autosize=True,  
        template="plotly_white",
        legend=dict(
            font=dict(size=12), 
            y=0.98, 
            yanchor="top",
            x=0.99,                            
            xanchor="right"
        ),
        margin=dict(l=50, r=120, t=80, b=50), 
        plot_bgcolor='white', 
        paper_bgcolor='white'
    )

    fig.update_xaxes(tickformat=".0%", dtick=0.05, range=[x_min, x_max])
    fig.update_yaxes(tickformat=".0%", dtick=0.05, range=[y_min, y_max])
    
    # Add config for better interaction in full screen
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="sans-serif"
        )
    )
    
    return fig

def print_portfolio_comparison(result):
    """Print comparison between pure mathematical and diversified portfolios"""
    pure = result['pure_portfolio']
    div = result['diversified_portfolio']
    
    print("\n=== PURE MATHEMATICAL PORTFOLIO (NO SECTOR CONSTRAINTS) ===")
    print(f"Sharpe Ratio: {pure['sharpe']:.4f}")
    print(f"Expected Return: {pure['return']:.2%}")
    print(f"Expected Volatility: {pure['volatility']:.2%}")
    
    print("\nTop 5 holdings:")
    for holding in pure['top_holdings']:
        print(f"{holding['ticker']}: {holding['weight']:.2%} | Sector: {holding['sector']}")
    
    print(f"\nNumber of unique sectors: {pure['unique_sectors']}")
    print(f"Number of assets with weight > 1%: {pure['assets_above_1pct']}")
    
    print("\n=== DIVERSIFIED PORTFOLIO (SECTOR-CONSTRAINED) ===")
    print(f"Sharpe Ratio: {div['sharpe']:.4f}")
    print(f"Expected Return: {div['return']:.2%}")
    print(f"Expected Volatility: {div['volatility']:.2%}")
    
    print(f"\nNumber of unique sectors: {div['unique_sectors']}")
    print(f"Maximum weight: {div['max_weight']:.2%}")
    print(f"Sharpe ratio difference vs. pure mathematical: {(pure['sharpe'] - div['sharpe'])/pure['sharpe']:.2%}")