import pandas as pd

def compute_behavioral_indices(ts_df, window=4):
    """Calculates engagement volatility and behavioral trend drift."""
    ts_df_copy = ts_df.copy()
    
    # Volatility: Rolling standard deviation
    ts_df_copy['volatility_idx'] = ts_df_copy.groupby('id_student')['sum_click'].transform(
        lambda x: x.rolling(window, min_periods=2).std()
    ).fillna(0)
    
    # Drift: Rate of change (Current week - Week X days ago) / X
    ts_df_copy['drift_idx'] = ts_df_copy.groupby('id_student')['sum_click'].transform(
        lambda x: x.diff(periods=window-1) / (window-1)
    ).fillna(0)
    
    return ts_df_copy
