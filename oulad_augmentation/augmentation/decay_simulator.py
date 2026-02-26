import pandas as pd
import numpy as np

def simulate_decay_pattern(clicks_array, pattern='gradual'):
    """Applies decay masks to an array of weekly clicks."""
    n_weeks = len(clicks_array)
    decay_mask = np.ones(n_weeks)
    
    if pattern == 'gradual':
        # Linear degradation to 10%
        decay_mask = np.linspace(1.0, 0.1, n_weeks)
        
    elif pattern == 'sudden_crash':
        # Normal engagement, then drops functionally to zero in one week
        crash_week = np.random.randint(n_weeks // 3, int(n_weeks * 0.8))
        decay_mask[crash_week:] = np.random.uniform(0.01, 0.05, n_weeks - crash_week)
        
    elif pattern == 'burnout':
        # Exponential decay representing lost motivation over time
        decay_mask = np.exp(-np.arange(n_weeks) / (n_weeks / 4.0))
        
    return clicks_array * decay_mask

def inject_dropout_timing(ts_df, collapse_threshold=5, window=3):
    """
    Labels a student as 'Dropped Out' if clicks average below threshold
    for `window` consecutive weeks.
    """
    # Calculate rolling average of clicks
    rolling_avg = ts_df.groupby('id_student')['sum_click'].rolling(window=window).mean().reset_index(0, drop=True)
    
    # Find the first week where rolling avg falls below threshold (collapse)
    collapse_mask = rolling_avg < collapse_threshold
    
    # Identify dropout events and the week they occur
    ts_df_copy = ts_df.copy()
    ts_df_copy['is_collapsed'] = collapse_mask
    dropout_weeks = ts_df_copy[ts_df_copy['is_collapsed']].groupby('id_student')['week'].min().reset_index()
    dropout_weeks.columns = ['id_student', 'dropout_week']
    
    return pd.merge(ts_df_copy, dropout_weeks, on='id_student', how='left')
