import pandas as pd
from lifelines.utils import add_covariate_to_timeline

def format_for_survival_analysis(df, duration_col='week', event_col='is_collapsed', static_cols=None):
    """
    Formats the time-series cross-sectional data into a structure suitable for Cox-Time Varying
    survival analysis using the `lifelines` library.
    
    The format requires:
        - id: student identifier
        - start: start time of observation period
        - stop: end time of observation period
        - event: boolean whether dropout occurred at `stop` time
    """
    print("Formatting dataset for Time-Varying Survival Analysis...")
    
    # We will use the raw timeline dataframe
    surv_df = df.copy()
    
    # Ensure start and stop times
    surv_df['start_time'] = surv_df['week']
    surv_df['stop_time'] = surv_df['week'] + 1
    
    # Rename for lifelines conventions
    surv_df.rename(columns={'is_collapsed': 'event_occurred'}, inplace=True)
    
    # Ensure boolean event flag
    surv_df['event_occurred'] = surv_df['event_occurred'].astype(bool)
    
    # Subselect required columns for the survival model
    # We strip out metadata like code_module/presentation unless they are OHE encoded
    features_to_keep = [
        'id_student', 'start_time', 'stop_time', 'event_occurred',
        'sum_click', 'drift_idx', 'volatility_idx', 'synthesized_hesitation_sec'
    ]
    
    # Add any static demographics that were one-hot encoded
    if static_cols:
        features_to_keep.extend(static_cols)
        
    final_surv_df = surv_df[features_to_keep].copy()
    
    return final_surv_df
