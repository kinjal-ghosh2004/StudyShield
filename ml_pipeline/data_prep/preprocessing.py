import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_merge_data(ts_path, info_path):
    """
    Loads time-series behavioral data and merges it with static demographic student information.
    """
    ts_df = pd.read_csv(ts_path)
    info_df = pd.read_csv(info_path)
    
    # Merge on id_student
    # Note: info_df might have multiple entries for retaking students. 
    # For simplicity, we keep the first occurrence of demographics per student.
    info_df_unique = info_df.drop_duplicates(subset=['id_student'], keep='first')
    
    merged_df = pd.merge(ts_df, info_df_unique, on='id_student', how='left')
    return merged_df

def impute_missing_values(df):
    """
    Handles missing values in both static and temporal data.
    """
    df_clean = df.copy()
    
    # Impute categorical demographics
    if 'imd_band' in df_clean.columns:
        df_clean['imd_band'] = df_clean['imd_band'].fillna('Unknown')
    
    # Forward fill temporal data if needed (though our previous pipeline handles sparse weeks natively)
    df_clean.fillna(method='ffill', inplace=True)
    df_clean.fillna(0, inplace=True) # Final fallback for starting NAs
    
    return df_clean

def encode_categorical(df, cat_cols):
    """
    One-hot encodes specified categorical variables.
    """
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df_encoded

from .behavioral_drift import BehavioralDriftDetector

def construct_tabular_features(df, n_lags=3):
    """
    Transforms sequence data into flattened tabular format suitable for XGBoost and Survival models.
    Also injects Advanced Behavioral Drift metrics (JSD/DTW).
    """
    print("Constructing tabular cross-features and lag variables...")
    
    # Create temporal cross-features
    global_avg_clicks = df.groupby('week')['sum_click'].transform('mean')
    df['cumulative_engagement_rate'] = df.groupby('id_student')['sum_click'].cumsum() / df.groupby(['id_student', 'week'])['week'].transform(lambda w: (w+1) * global_avg_clicks.mean() + 1e-5)
    
    df['volatilty_hesitation_ratio'] = df['volatility_idx'] / (df['synthesized_hesitation_sec'] + 1e-5)
    
    # === INTEGRATE BEHAVIORAL DRIFT FRAMEWORK ===
    drift_detector = BehavioralDriftDetector(historical_window=4, current_window=2)
    # 1. Intra-student
    df = drift_detector.calculate_intra_student_drift(df, feature_col='sum_click')
    
    # 2. Inter-student (Assuming students who didn't collapse are 'successful' prototypes for this simplified run)
    df_successful = df[df['is_collapsed'] == False].copy()
    drift_detector.build_successful_prototypes(df_successful)
    df = drift_detector.calculate_inter_student_drift(df, feature_cols=['sum_click'])
    
    # 3. Unified Index
    df = drift_detector.calculate_unified_drift_index(df)
    
    # Flatten the last N weeks of behavioral metrics for Tabular models
    tabular_df = df.copy()
    
    # Adding the new drift features to the flattening list
    features_to_lag = [
        'sum_click', 'drift_idx', 'volatility_idx', 'synthesized_hesitation_sec',
        'drift_jsd', 'drift_dtw_zscore', 'udi_final'
    ]
    
    for feature in features_to_lag:
        for lag in range(1, n_lags + 1):
            tabular_df[f'{feature}_lag_{lag}'] = tabular_df.groupby('id_student')[feature].shift(lag).fillna(0)
    
    return tabular_df
    
    for feature in features_to_lag:
        for lag in range(1, n_lags + 1):
            tabular_df[f'{feature}_lag_{lag}'] = tabular_df.groupby('id_student')[feature].shift(lag).fillna(0)
    
    return tabular_df
