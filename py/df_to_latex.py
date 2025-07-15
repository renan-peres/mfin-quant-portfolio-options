def df_to_latex_fixed(df, index=False, column_format=None, escape_chars=None, **kwargs):
    """
    Convert DataFrame to LaTeX with proper escaping of special characters
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to convert to LaTeX
    index : bool, default False
        Whether to include row index in output
    column_format : str, optional
        LaTeX column format (e.g., 'lrrrr'). If None, auto-detects based on data types
    escape_chars : dict, optional
        Custom escape characters. Default: {'$': '\\$', '%': '\\%'}
    **kwargs : dict
        Additional arguments passed to pd.DataFrame.to_latex()
    
    Returns:
    --------
    str
        LaTeX table string with properly escaped characters
    """
    
    # Default escape characters
    if escape_chars is None:
        escape_chars = {'$': '\\$', '%': '\\%'}
    
    # Auto-detect column format if not provided
    if column_format is None:
        format_chars = []
        for col in df.columns:
            # Check if column contains numeric data (excluding strings with $ or %)
            try:
                # Try to convert to numeric, excluding string representations
                sample_val = str(df[col].iloc[0]) if len(df) > 0 else ''
                if any(char in sample_val for char in ['$', '%']) or df[col].dtype == 'object':
                    format_chars.append('l')  # Left align for text/currency/percentages
                else:
                    format_chars.append('r')  # Right align for pure numbers
            except:
                format_chars.append('l')  # Default to left align
        
        column_format = ''.join(format_chars)
    
    # Convert to LaTeX
    latex_output = df.to_latex(
        index=index,
        escape=False,  # We handle escaping manually
        column_format=column_format,
        **kwargs
    )
    
    # Apply character escaping
    for char, escaped_char in escape_chars.items():
        latex_output = latex_output.replace(char, escaped_char)
    
    return latex_output

# Advanced function with additional LaTeX formatting options
def df_to_latex_publication_ready(df, caption=None, label=None, position='htbp', **kwargs):
    """
    Convert DataFrame to publication-ready LaTeX table with full table environment
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to convert
    caption : str, optional
        Table caption
    label : str, optional
        Table label for referencing (e.g., 'tab:strategy_comparison')
    position : str, default 'htbp'
        Table position specifier
    **kwargs : dict
        Arguments passed to df_to_latex_fixed()
    
    Returns:
    --------
    str
        Complete LaTeX table environment
    """
    
    # Get the basic table
    table_content = df_to_latex_fixed(df, **kwargs)
    
    # Wrap in table environment
    full_table = f"\\begin{{table}}[{position}]\n"
    full_table += "\\centering\n"
    
    if caption:
        full_table += f"\\caption{{{caption}}}\n"
    
    full_table += table_content
    
    if label:
        full_table += f"\\label{{{label}}}\n"
    
    full_table += "\\end{table}"
    
    return full_table