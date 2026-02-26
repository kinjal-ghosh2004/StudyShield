import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class BehavioralDriftDetector:
    def __init__(self, historical_window=4, current_window=2):
        self.hist_win = historical_window
        self.cur_win = current_window
        self.prototype_trajectories = {} # Will store successful peer prototypes

    def compute_jsd(self, P, Q):
        """Computes Jensen-Shannon Divergence between two distributions."""
        # Add small epsilon to avoid division by zero
        P = np.array(P) + 1e-10
        Q = np.array(Q) + 1e-10
        
        # Normalize to probability distributions
        P = P / np.sum(P)
        Q = Q / np.sum(Q)
        
        return jensenshannon(P, Q)

    def calculate_intra_student_drift(self, df, feature_col='sum_click'):
        """
        Calculates JSD between historical baseline and current behavior window.
        """
        print(f"Calculating Intra-Student Drift (JSD) for {feature_col}...")
        df_sorted = df.sort_values(['id_student', 'week']).copy()
        
        jsd_scores = []
        
        for student, group in df_sorted.groupby('id_student'):
            vals = group[feature_col].values
            n = len(vals)
            
            student_jsd = np.zeros(n)
            
            # We need at least (hist_win + cur_win) weeks of data to calculate drift
            for i in range(self.hist_win + self.cur_win - 1, n):
                # Historical distribution (e.g. weeks 1-4)
                hist_dist = vals[i - self.cur_win - self.hist_win + 1 : i - self.cur_win + 1]
                # Current distribution (e.g. weeks 5-6)
                cur_dist = vals[i - self.cur_win + 1 : i + 1]
                
                jsd = self.compute_jsd(hist_dist, cur_dist)
                student_jsd[i] = jsd
                
            jsd_scores.extend(student_jsd)
            
        df_sorted['drift_jsd'] = jsd_scores
        return df_sorted

    def build_successful_prototypes(self, df_successful, feature_cols=['sum_click']):
        """
        Builds the Y_proto reference trajectories using DTW averaging or median 
        paths of students who successfully completed the course.
        Simplified here to take the median weekly trajectory.
        """
        print("Building successful prototype trajectories...")
        self.prototype_trajectories = df_successful.groupby('week')[feature_cols].median().reset_index()

    def calculate_inter_student_drift(self, df, feature_cols=['sum_click']):
        """
        Calculates DTW distance between a student's current sequence and the prototype.
        """
        if self.prototype_trajectories is None or len(self.prototype_trajectories) == 0:
            raise ValueError("Prototypes must be built before calculating inter-student drift.")
            
        print("Calculating Inter-Student Drift (DTW Z-Score)...")
        df_sorted = df.sort_values(['id_student', 'week']).copy()
        
        dtw_distances = []
        
        for student, group in df_sorted.groupby('id_student'):
            dtw_student = np.zeros(len(group))
            
            for i in range(1, len(group) + 1):
                student_seq = group.iloc[:i][feature_cols].values
                proto_seq = self.prototype_trajectories[self.prototype_trajectories['week'] <= group.iloc[i-1]['week']][feature_cols].values
                
                # If prototype is shorter or missing, skip DTW
                if len(proto_seq) == 0 or len(student_seq) == 0:
                    dtw_student[i-1] = 0
                    continue
                    
                distance, _ = fastdtw(student_seq, proto_seq, dist=euclidean)
                dtw_student[i-1] = distance
                
            dtw_distances.extend(dtw_student)
            
        df_sorted['dtw_distance'] = dtw_distances
        
        # Convert absolute DTW to Z-scores per week to represent 'anomaly' severity
        df_sorted['drift_dtw_zscore'] = df_sorted.groupby('week')['dtw_distance'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-5)
        ).fillna(0)
        
        return df_sorted

    def calculate_unified_drift_index(self, df, alpha=0.6, beta=0.4, gamma=0.2):
        """
        Combines JSD and DTW Z-scores into a single Unified Drift Index (UDI)
        """
        # Ensure DTW anomaly pushes upwards via Sigmoid scaling
        # Sigmoid(Z) centers at 0.5. We shift it so a Z score of 0 contributes 0, and high Z pushes near 1.
        def sigmoid_shift(x):
            return 1 / (1 + np.exp(-x)) - 0.5
            
        dtw_component = df['drift_dtw_zscore'].apply(sigmoid_shift).clip(lower=0)
        
        df['udi'] = (alpha * df['drift_jsd']) + (beta * dtw_component)
        
        # Derivative/Acceleration Term (difference from last week)
        df['udi_derivative'] = df.groupby('id_student')['udi'].diff().fillna(0)
        
        df['udi_final'] = df['udi'] + (gamma * df['udi_derivative'].clip(lower=0))
        
        return df
