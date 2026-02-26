import pandas as pd
import numpy as np

def convert_to_weekly(student_vle_df, start_week=0, end_week=40):
    """Aggregates raw daily logs into weekly time-series per student."""
    # Convert 'date' (days) to weeks
    student_vle_df['week'] = np.floor(student_vle_df['date'] / 7).astype(int)
    
    # Filter valid weeks
    df = student_vle_df[(student_vle_df['week'] >= start_week) & (student_vle_df['week'] <= end_week)]
    
    # Aggregate clicks per student and week
    weekly_logs = df.groupby(
        ['id_student', 'week']
    )['sum_click'].sum().reset_index()
    
    # Create complete sequence (fill missing weeks with 0 clicks)
    idx = pd.MultiIndex.from_product(
        [weekly_logs['id_student'].unique(), range(start_week, end_week + 1)],
        names=['id_student', 'week']
    )
    
    weekly_ts = weekly_logs.set_index(['id_student', 'week']).reindex(idx, fill_value=0).reset_index()
    return weekly_ts
