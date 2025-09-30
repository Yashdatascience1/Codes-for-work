import pandas as pd
from typing import Dict, List

def compare_dataframe_columns(*dataframes, names=None, print_results=True):
    """
    Compare columns across multiple dataframes and identify unique columns in each.
    
    Parameters:
    -----------
    *dataframes : pandas.DataFrame
        Variable number of dataframes to compare
    names : list, optional
        Names for each dataframe. If not provided, will use df1, df2, df3, etc.
    print_results : bool, default=True
        Whether to print the comparison results. Set to False to only return the dictionary.
    
    Returns:
    --------
    dict : Dictionary with dataframe names as keys and lists of unique columns as values
    
    Example:
    --------
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    df2 = pd.DataFrame({'A': [1, 2], 'D': [7, 8]})
    df3 = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'E': [9, 10]})
    
    # With printed output
    result = compare_dataframe_columns(df1, df2, df3, names=['Sales', 'Marketing', 'HR'])
    
    # Without printed output
    result = compare_dataframe_columns(df1, df2, df3, print_results=False)
    """
    
    if len(dataframes) < 2:
        raise ValueError("Please provide at least 2 dataframes to compare")
    
    # Generate default names if not provided
    if names is None:
        names = [f"df{i+1}" for i in range(len(dataframes))]
    elif len(names) != len(dataframes):
        raise ValueError(f"Number of names ({len(names)}) must match number of dataframes ({len(dataframes)})")
    
    # Store column sets for each dataframe
    column_sets = {name: set(df.columns) for name, df in zip(names, dataframes)}
    
    # Find unique columns for each dataframe
    unique_columns = {}
    
    for i, (name, df) in enumerate(zip(names, dataframes)):
        # Get columns from current dataframe
        current_cols = column_sets[name]
        
        # Get union of all other dataframes' columns
        other_cols = set()
        for j, other_name in enumerate(names):
            if i != j:
                other_cols.update(column_sets[other_name])
        
        # Find columns unique to current dataframe
        unique = sorted(list(current_cols - other_cols))
        unique_columns[name] = unique
    
    # Print results if requested
    if print_results:
        print("=" * 60)
        print("DATAFRAME COLUMN COMPARISON RESULTS")
        print("=" * 60)
        print()
        
        for df_name, unique_cols in unique_columns.items():
            if unique_cols:
                print(f"{df_name} has {unique_cols} which are not present in other dataframes")
            else:
                print(f"{df_name} has no unique columns")
        
        print("=" * 60)
    
    return unique_columns