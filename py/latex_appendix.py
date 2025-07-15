import pandas as pd
import numpy as np


def get_global_dataframes():
    """
    Automatically discover all pandas DataFrames in the global namespace
    """
    global_vars = globals()
    dataframes = {}
    
    for name, obj in global_vars.items():
        if isinstance(obj, pd.DataFrame) and not name.startswith('_'):
            dataframes[name] = obj
    
    return dataframes


def detect_column_format(series, column_name=""):
    """
    Enhanced column format detection with specific handling for statistical columns
    """
    if pd.api.types.is_numeric_dtype(series):
        column_lower = column_name.lower()
        
        # Specific handling for p-value columns
        if 'p_value' in column_lower or 'pvalue' in column_lower or column_lower.startswith('p '):
            return 'decimal_4'
        
        # Specific handling for intercept (alpha) columns - now 4 decimal places
        if 'intercept' in column_lower and 'alpha' in column_lower:
            return 'decimal_4'
        
        # Other statistical percentage columns (correlation, r_squared)
        if any(keyword in column_lower for keyword in ['correlation', 'r_squared']):
            return 'percentage'
        
        # Currency detection keywords
        currency_keywords = ['price', 'cost', 'value', 'cap', 'revenue', 'sales', 'amount', 'dollars', 'usd', 'market', 'enterprise']
        
        # If column name suggests currency or values are large (but not p-values)
        if any(keyword in column_lower for keyword in currency_keywords) or series.min() > 1000:
            return 'currency'
        
        # Check if values look like percentages (between 0 and 1)
        elif series.between(0, 1).all() and series.max() <= 1:
            return 'percentage'
        
        # Default to decimal formatting for other numeric data
        else:
            return 'decimal'
    else:
        return 'string'


def format_currency_value(value):
    """
    Format currency values with proper thousands separators and scaling
    """
    if pd.isnull(value):
        return value
    
    # Format large numbers with appropriate scale
    if abs(value) >= 1e12:
        return f"\\${value/1e12:.2f}T"
    elif abs(value) >= 1e9:
        return f"\\${value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"\\${value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"\\${value/1e3:.1f}K"
    else:
        return f"\\${value:.2f}"


def escape_latex_special_chars(s):
    """
    Escape special LaTeX characters in strings
    """
    if isinstance(s, str):
        s = s.replace('$', '\\$')
        s = s.replace('%', '\\%')
    return s


def format_dataframe_dynamic(df, max_rows=10, max_cols=10, max_char_len=15):
    """
    Enhanced formatting with 4 decimal places for p-values and intercepts
    """
    # Limit rows and columns first
    limited_df = df.iloc[:max_rows, :max_cols].copy()
    
    # Apply formatting based on detected content type
    for col in limited_df.columns:
        format_type = detect_column_format(limited_df[col], col)
        
        if format_type == 'percentage':
            # Escape the % symbol in percentage formatting
            limited_df[col] = limited_df[col].apply(lambda x: f"{x*100:.2f}\\%" if pd.notnull(x) else x)
        elif format_type == 'currency':
            # Use improved currency formatting with scaling
            limited_df[col] = limited_df[col].apply(format_currency_value)
        elif format_type == 'decimal_4':
            # Format p-values and intercepts with 4 decimal places
            limited_df[col] = limited_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else x)
        elif format_type == 'decimal':
            # Format as decimal with 2 decimal places
            limited_df[col] = limited_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
        elif format_type == 'string':
            # Truncate string columns and escape special characters
            limited_df[col] = limited_df[col].apply(
                lambda x: escape_latex_special_chars(x) if len(str(x)) <= max_char_len 
                else escape_latex_special_chars(str(x)[:max_char_len-3] + '...')
            )
        else:
            # For any other type, just escape if it's a string
            limited_df[col] = limited_df[col].apply(lambda x: escape_latex_special_chars(str(x)) if pd.notnull(x) else x)
    
    return limited_df


def df_to_latex_publication_ready(df, caption="", label="", column_format=None):
    """
    Convert DataFrame to publication-ready LaTeX table
    """
    if column_format is None:
        column_format = 'l' * len(df.columns)
    
    # Generate LaTeX table
    latex_str = "\\begin{table}[H]\n"
    latex_str += "\\centering\n"
    latex_str += "\\adjustbox{width=\\textwidth,center}\n"
    latex_str += "{\n"
    latex_str += f"\\begin{{tabular}}{{{column_format}}}\n"
    latex_str += "\\toprule\n"
    
    # Add header
    header = " & ".join(df.columns) + " \\\\\n"
    latex_str += header
    latex_str += "\\midrule\n"
    
    # Add data rows
    for _, row in df.iterrows():
        row_str = " & ".join([str(val) for val in row]) + " \\\\\n"
        latex_str += row_str
    
    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "}\n"
    
    if caption:
        latex_str += f"\\caption{{{caption}}}\n"
    if label:
        latex_str += f"\\label{{{label}}}\n"
    
    latex_str += "\\end{table}\n"
    
    return latex_str


def generate_section_name(df_name):
    """
    Generate a readable section name from DataFrame variable name
    """
    # Convert snake_case to Title Case
    words = df_name.replace('_', ' ').split()
    return ' '.join(word.capitalize() for word in words)


def generate_latex_document(skip_dataframe=None, order=None):
    """
    Generate complete LaTeX document with hierarchical section structure (no title page)
    
    Parameters:
    skip_dataframe: str, list, or None - dataframe(s) to skip
    order: list or None - optional list of dataframe names to control presentation order
    """
    
    # LaTeX document header WITHOUT title, author, date, and maketitle
    latex_header = r"""\documentclass{article}


% Page layout and margins
\usepackage[
    top=0.2in,
    bottom=0.2in,
    left=1in,
    right=1in,
    includeheadfoot
]{geometry}


% Essential packages for table formatting
\usepackage{graphicx} % Required for inserting images
\usepackage{booktabs} % For professional table formatting
\usepackage{array} % For enhanced column types
\usepackage{amsmath} % For mathematical symbols
\usepackage{multirow} % For multi-row cells
\usepackage{siunitx} % For number formatting
\usepackage{threeparttable} % For table notes
\usepackage{adjustbox} % For table scaling
\usepackage{float} % For better table positioning
\usepackage{setspace} % For line spacing control


% Enhanced siunitx setup for better number formatting
\sisetup{
    group-separator = {,},
    group-minimum-digits = 4,
    table-number-alignment = center
}


% Set line spacing
\onehalfspacing


\begin{document}


"""
    
    print(latex_header)
    
    # Main section for all tables
    print("\\section{Tables}")
    print()
    
    # Discover all global DataFrames
    all_dataframes = get_global_dataframes()
    
    # Normalize skip_dataframe to a list for easier processing
    if skip_dataframe is None:
        skip_list = []
    elif isinstance(skip_dataframe, str):
        skip_list = [skip_dataframe]
    elif isinstance(skip_dataframe, list):
        skip_list = skip_dataframe
    else:
        skip_list = [skip_dataframe]  # fallback for other types

    # Determine the order of dataframes to print
    if order is not None:
        # Filter order list to only include dataframes that exist and are not skipped
        ordered_names = [name for name in order if name in all_dataframes and name not in skip_list]
        # Add remaining dataframes not in order and not skipped
        remaining_names = [name for name in all_dataframes if name not in ordered_names and name not in skip_list]
        dataframe_names_to_print = ordered_names + remaining_names
    else:
        # Default order: all dataframes except skipped
        dataframe_names_to_print = [name for name in all_dataframes if name not in skip_list]
    
    # Process each DataFrame in the determined order
    for df_name in dataframe_names_to_print:
        df = all_dataframes[df_name]
        
        # Use subsection instead of section for hierarchical structure
        print(f"\\subsection{{{generate_section_name(df_name)}}}")
        print()
        
        # Format the dataframe dynamically with enhanced column detection
        formatted_df = format_dataframe_dynamic(df, max_rows=10, max_cols=10, max_char_len=15)
        
        # Generate appropriate column format (left-aligned for strings, right-aligned for numbers)
        column_format = ""
        for col in formatted_df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                column_format += "r"
            else:
                column_format += "l"
        
        # Generate LaTeX table
        latex_table = df_to_latex_publication_ready(
            formatted_df,
            caption=f"{generate_section_name(df_name)} Overview",
            label=f"tab:{df_name}",
            column_format=column_format
        )
        
        print(latex_table)
        print()
    
    # LaTeX document footer
    print(r"\end{document}")


def generate_latex_document_with_manual_control(skip_dataframe=None, column_format_dict=None, order=None):
    """
    Version with manual column formatting control for maximum precision and hierarchical sections (no title page)
    
    Parameters:
    skip_dataframe: str, list, or None - dataframe(s) to skip
    column_format_dict: dict mapping DataFrame names to column format dictionaries
    order: list or None - optional list of dataframe names to control presentation order
    Example: {
        'benchmark_long_df': {
            'p_value_slope': 'decimal_4',
            'p_value_intercept': 'decimal_4',
            'intercept (alpha)': 'decimal_4'
        }
    }
    """
    if column_format_dict is None:
        column_format_dict = {}
    
    # LaTeX document header WITHOUT title, author, date, and maketitle
    latex_header = r"""\documentclass{article}


% Page layout and margins
\usepackage[
    top=0.2in,
    bottom=0.2in,
    left=1in,
    right=1in,
    includeheadfoot
]{geometry}


% Essential packages for table formatting
\usepackage{graphicx} % Required for inserting images
\usepackage{booktabs} % For professional table formatting
\usepackage{array} % For enhanced column types
\usepackage{amsmath} % For mathematical symbols
\usepackage{multirow} % For multi-row cells
\usepackage{siunitx} % For number formatting
\usepackage{threeparttable} % For table notes
\usepackage{adjustbox} % For table scaling
\usepackage{float} % For better table positioning
\usepackage{setspace} % For line spacing control


% Enhanced siunitx setup for better number formatting
\sisetup{
    group-separator = {,},
    group-minimum-digits = 4,
    table-number-alignment = center
}


% Set line spacing
\onehalfspacing


\begin{document}


"""
    
    print(latex_header)
    
    # Main section for all tables
    print("\\section{Tables}")
    print()
    
    # Discover all global DataFrames
    all_dataframes = get_global_dataframes()
    
    # Normalize skip_dataframe to a list
    if skip_dataframe is None:
        skip_list = []
    elif isinstance(skip_dataframe, str):
        skip_list = [skip_dataframe]
    elif isinstance(skip_dataframe, list):
        skip_list = skip_dataframe
    else:
        skip_list = [skip_dataframe]

    # Determine the order of dataframes to print
    if order is not None:
        # Filter order list to only include dataframes that exist and are not skipped
        ordered_names = [name for name in order if name in all_dataframes and name not in skip_list]
        # Add remaining dataframes not in order and not skipped
        remaining_names = [name for name in all_dataframes if name not in ordered_names and name not in skip_list]
        dataframe_names_to_print = ordered_names + remaining_names
    else:
        # Default order: all dataframes except skipped
        dataframe_names_to_print = [name for name in all_dataframes if name not in skip_list]
    
    # Process each DataFrame in the determined order
    for df_name in dataframe_names_to_print:
        df = all_dataframes[df_name]
        
        # Use subsection instead of section for hierarchical structure
        print(f"\\subsection{{{generate_section_name(df_name)}}}")
        print()
        
        # Get manual column formatting for this DataFrame
        manual_formats = column_format_dict.get(df_name, {})
        
        # Enhanced formatting with manual overrides
        def format_with_manual_control(df, manual_formats, max_rows=10, max_cols=10, max_char_len=15):
            limited_df = df.iloc[:max_rows, :max_cols].copy()
            
            for col in limited_df.columns:
                # Check if there's a manual format override
                if col in manual_formats:
                    format_type = manual_formats[col]
                else:
                    # Use automatic detection
                    format_type = detect_column_format(limited_df[col], col)
                
                if format_type == 'percentage':
                    limited_df[col] = limited_df[col].apply(lambda x: f"{x*100:.2f}\\%" if pd.notnull(x) else x)
                elif format_type == 'currency':
                    limited_df[col] = limited_df[col].apply(format_currency_value)
                elif format_type == 'decimal_4':
                    limited_df[col] = limited_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else x)
                elif format_type == 'decimal':
                    limited_df[col] = limited_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
                elif format_type == 'string':
                    limited_df[col] = limited_df[col].apply(
                        lambda x: escape_latex_special_chars(x) if len(str(x)) <= max_char_len 
                        else escape_latex_special_chars(str(x)[:max_char_len-3] + '...')
                    )
                else:
                    limited_df[col] = limited_df[col].apply(lambda x: escape_latex_special_chars(str(x)) if pd.notnull(x) else x)
            
            return limited_df
        
        # Format the dataframe with manual control
        formatted_df = format_with_manual_control(df, manual_formats, max_rows=10, max_cols=10, max_char_len=15)
        
        # Generate column format
        column_format = ""
        for col in formatted_df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                column_format += "r"
            else:
                column_format += "l"
        
        # Generate LaTeX table
        latex_table = df_to_latex_publication_ready(
            formatted_df,
            caption=f"{generate_section_name(df_name)} Overview",
            label=f"tab:{df_name}",
            column_format=column_format
        )
        
        print(latex_table)
        print()
    
    # LaTeX document footer
    print(r"\end{document}")

# Usage Examples:

# Example 1: Use default order (no change from current behavior)
# generate_latex_document(skip_dataframe=['daily_quotes', 'df', 'portfolio_long_df'])

# Example 2: Specify exact order for specific dataframes
# generate_latex_document(
#     skip_dataframe=['daily_quotes', 'df', 'portfolio_long_df'],
#     order=['benchmark_results', 'summary_stats', 'analysis_df']
# )

# Example 3: Put important dataframes first, others will follow in discovery order
# generate_latex_document(
#     skip_dataframe=['daily_quotes', 'df'],
#     order=['executive_summary', 'key_metrics']
# )

# Example 4: With manual control function
# generate_latex_document_with_manual_control(
#     skip_dataframe=['daily_quotes', 'df'],
#     order=['results_df', 'benchmark_long_df'],
#     column_format_dict={
#         'benchmark_long_df': {
#             'p_value_slope': 'decimal_4',
#             'p_value_intercept': 'decimal_4'
#         }
#     }
# )

# Your current usage with the new order parameter:
generate_latex_document(
    skip_dataframe=['daily_quotes', 'df', 'portfolio_long_df', 'portfolio_short_df', 'strategy_df', 'stock_data', 'monthly_quotes'],
    order=['portfolio_stocks', 'strategy_summary_df', 'scenario_results'] 
)