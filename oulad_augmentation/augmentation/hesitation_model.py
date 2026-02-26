import numpy as np

def generate_hesitation_time(ts_df, difficulty_index, base_hesitation=30.0):
    """
    Hesitation (seconds) = Base + (Difficulty / Engagement_Factor) + Noise
    """
    ts_df_copy = ts_df.copy()
    
    # Smoothing out zero clicks to prevent division by zero
    engagement_factor = np.log1p(ts_df_copy['sum_click']) + 1.0 
    
    # Gaussian noise for realism
    noise = np.random.normal(loc=0.0, scale=5.0, size=len(ts_df_copy))
    
    ts_df_copy['synthesized_hesitation_sec'] = (
        base_hesitation + 
        (difficulty_index / engagement_factor * 10.0) + 
        noise
    )
    
    # Ensure hesitation time cannot be negative
    ts_df_copy['synthesized_hesitation_sec'] = ts_df_copy['synthesized_hesitation_sec'].clip(lower=2.0)
    return ts_df_copy
