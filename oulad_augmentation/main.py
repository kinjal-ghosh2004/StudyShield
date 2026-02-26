import pandas as pd
import os
import argparse

from preprocessing.time_series import convert_to_weekly
from augmentation.decay_simulator import simulate_decay_pattern, inject_dropout_timing
from augmentation.hesitation_model import generate_hesitation_time
from feature_engineering.behavioral_metrics import compute_behavioral_indices
from rl_env.intervention_sim import simulate_rl_transitions

def run_augmentation_pipeline(raw_vle_path):
    print("1. Loading and converting to weekly time series...")
    if not os.path.exists(raw_vle_path):
        raise FileNotFoundError(f"Data file not found: {raw_vle_path}")
        
    vle_df = pd.read_csv(raw_vle_path)
    weekly_ts = convert_to_weekly(vle_df)
    
    print("2. Augmenting trajectories with decay patterns (Data Multiplication)...")
    # Duplicate dataframe for augmentation
    augmented_ts = weekly_ts.copy()
    augmented_ts['id_student'] = augmented_ts['id_student'].astype(str) + "_aug_crash"
    
    # Apply Sudden Crash pattern mapping
    augmented_ts['sum_click'] = augmented_ts.groupby('id_student')['sum_click'].transform(
        lambda x: simulate_decay_pattern(x.values, pattern='sudden_crash')
    )
    
    # Combine real data with augmented synthetic data
    full_ts = pd.concat([weekly_ts, augmented_ts]).reset_index(drop=True)
    
    print("3. Injecting Target Dropouts & Generating Hesitation...")
    full_ts = inject_dropout_timing(full_ts, collapse_threshold=5, window=3)
    
    # Assume global difficulty index of 1.5 for this specific module
    full_ts = generate_hesitation_time(full_ts, difficulty_index=1.5)
    
    print("4. Computing Behavioral Drift and Volatility...")
    full_ts = compute_behavioral_indices(full_ts)
    
    print("5. Generating Offline RL Transitions...")
    rl_dataset = simulate_rl_transitions(full_ts)
    
    print("Pipeline Complete. RL Dataset Ready.")
    return full_ts, rl_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OULAD Data Augmentation Pipeline')
    parser.add_argument('--input', type=str, default='studentVle.csv', help='Path to studentVle.csv')
    parser.add_argument('--output_ts', type=str, default='augmented_ts.csv', help='Path to output time series')
    parser.add_argument('--output_rl', type=str, default='rl_dataset.csv', help='Path to output RL tuples')
    args = parser.parse_args()
    
    try:
        ts_features, rl_data = run_augmentation_pipeline(args.input)
        
        # Save results
        ts_features.to_csv(args.output_ts, index=False)
        rl_data.to_csv(args.output_rl, index=False)
        print(f"Saved time-series features to {args.output_ts}")
        print(f"Saved RL transitions to {args.output_rl}")
    except FileNotFoundError as e:
        print(e)
        print("Please ensure the dataset file exists before running the pipeline.")
