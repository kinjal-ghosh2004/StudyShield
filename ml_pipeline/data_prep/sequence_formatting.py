import numpy as np
import pandas as pd

def construct_lstm_tensors(df, feature_cols, sequence_length=10):
    """
    Transforms flat weekly time-series data into 3D tensors: (Students, Time_Steps, Features)
    Expected input DataFrame should be sorted by id_student and then week.
    """
    print(f"Constructing LSTM tensors with sequence length {sequence_length}...")
    
    # Ensure chronological sort
    df_sorted = df.sort_values(['id_student', 'week']).copy()
    
    student_ids = df_sorted['id_student'].unique()
    num_students = len(student_ids)
    num_features = len(feature_cols)
    
    # Pre-allocate tensor: [Samples, Sequence_Length, Features]
    # If a student has fewer than sequence_length weeks, padded with zeros
    X_tensor = np.zeros((num_students, sequence_length, num_features), dtype=np.float32)
    y_target = np.zeros(num_students, dtype=np.int32)
    
    for idx, (student, group) in enumerate(df_sorted.groupby('id_student')):
        group_len = len(group)
        
        # Take up to the last `sequence_length` weeks
        if group_len >= sequence_length:
            window = group.iloc[-sequence_length:][feature_cols].values
        else:
            # Pad sequences that are too short (padding at the start)
            window = np.zeros((sequence_length, num_features))
            window[-group_len:] = group[feature_cols].values
            
        X_tensor[idx] = window
        
        # Determine the target for the student: did they drop out this semester?
        # Target based on if they have a recorded dropout week
        if 'dropout_week' in group.columns and not pd.isna(group['dropout_week'].iloc[0]):
            y_target[idx] = 1
        else:
            y_target[idx] = 0
            
    # Also collect static features for the embedding layer alongside LSTM
    # (Assuming static features don't change over the sequence)
    static_cols = [c for c in df.columns if c not in feature_cols and c not in ['id_student', 'week', 'dropout_week', 'is_collapsed']]
    
    # Get one row per student to extract static features
    static_features_df = df_sorted.drop_duplicates(subset=['id_student'], keep='last')[static_cols]
    static_tensor = static_features_df.values.astype(np.float32)
            
    return X_tensor, static_tensor, y_target, student_ids
