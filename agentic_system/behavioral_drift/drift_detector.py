import numpy as np
import torch
import torch.nn as nn
from collections import deque

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder to establish a baseline behavioral profile for a student.
    Expected input shape: (batch_size, sequence_length, num_features)
    """
    def __init__(self, num_features=4, hidden_dim=16, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=num_features,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        # x is (batch, seq_len, features)
        _, (hidden, _) = self.encoder(x)
        # hidden is (num_layers, batch, hidden_dim)
        
        # We need to broadcast the hidden state across the sequence length
        # for the decoder to reconstruct.
        seq_len = x.shape[1]
        
        # Take the last layer's hidden state and repeat it for each time step
        hidden_repeated = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        
        decoded, _ = self.decoder(hidden_repeated)
        return decoded

class BehavioralDriftDetector:
    """
    Monitors student engagement and compares current patterns to historical baselines
    using an LSTM Autoencoder and EWMA smoothing for drift detection.
    """
    def __init__(self, alpha=0.3, baseline_window=14, features=4):
        self.alpha = alpha  # Decay factor for EWMA
        self.baseline_window = baseline_window
        self.autoencoder = LSTMAutoencoder(num_features=features)
        
        # Placeholders for student baseline metrics (dict mapping student_id -> metrics)
        self.student_baselines = {}
        # Placeholders for smoothed drift scores D(t)
        self.current_drift_scores = {}

    def calculate_hesitation_index(self, session_telemetry):
        """
        Proprietary calculation of the Hesitation Index (H_t).
        Formula: (T_on_screen - T_active_scrolling) * (1 + DOM_Click_Variance)
        """
        # In a real system, these would be extracted from the session_telemetry object.
        # Here we mock the extraction of these specific metrics.
        t_on_screen = np.random.uniform(500, 1500) # Total seconds on page
        t_active_scrolling = np.random.uniform(100, 400) # Seconds actively moving mouse/scrolling
        
        # DOM_Click_Variance measures erratic, non-productive clicks (e.g. highlighting text repeatedly without action)
        dom_click_variance = np.random.uniform(0.1, 1.5) 
        
        # Calculate raw hesitation time
        raw_hesitation = max(0, t_on_screen - t_active_scrolling)
        
        # Scale by erratic behavior
        h_t = raw_hesitation * (1 + dom_click_variance)
        return h_t

    def extract_features(self, activity_logs):
        """
        Dynamically extracts features for a time window delta t.
        activity_logs: raw events
        Returns X_t = [f_pace, f_lag, f_hesitation, f_volatility]
        """
        # Placeholder for complex log parsing
        f_pace = np.random.uniform(0.5, 1.5)       # (Modules / Expected)
        f_lag = np.random.uniform(0.1, 5.0)        # Days
        
        # Use the proprietary formula instead of pure random
        f_hesitation = self.calculate_hesitation_index(activity_logs)  
        
        f_volatility = np.random.uniform(0.1, 2.0) # Variance in login time
        
        return np.array([f_pace, f_lag, f_hesitation, f_volatility])

    def train_baseline(self, student_id, historical_data):
        """
        Trains the autoencoder on the student's normal peak period.
        historical_data: shape (baseline_window, num_features)
        """
        # In a real scenario, we'd train the NN weights per student,
        # or use a global NN and just compute mu_error specific to the student.
        # We assume a global NN here and calculate student specific mu_error, sigma_error.
        
        self.autoencoder.eval()
        x_tensor = torch.tensor(historical_data, dtype=torch.float32).unsqueeze(0) # (1, seq, features)
        
        with torch.no_grad():
            reconstructed = self.autoencoder(x_tensor)
            
        # Error array for each timestep
        errors = torch.norm(x_tensor - reconstructed, dim=2).squeeze().numpy()
        
        mu_error = np.mean(errors)
        sigma_error = np.std(errors) + 1e-6 # prevent div/0
        
        self.student_baselines[student_id] = {
            'mu_error': mu_error,
            'sigma_error': sigma_error,
            'historical_seq': deque(historical_data.tolist()[-self.baseline_window:], maxlen=self.baseline_window)
        }
        self.current_drift_scores[student_id] = 0.0 # Initial D(t)

    def calculate_instantaneous_deviation(self, student_id, X_t):
        """
        Inject new daily vector and get reconstruction error d_t.
        """
        baseline = self.student_baselines[student_id]
        
        # Append new vector to sequence history to run through LSTM
        baseline['historical_seq'].append(X_t.tolist())
        seq = np.array(baseline['historical_seq'])
        x_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        
        self.autoencoder.eval()
        with torch.no_grad():
            reconstructed = self.autoencoder(x_tensor)
            
        # Get error for the most recent timestep ONLY (the last one)
        t_minus_1_target = x_tensor[:, -1, :]
        t_minus_1_pred = reconstructed[:, -1, :]
        d_t = torch.norm(t_minus_1_target - t_minus_1_pred).item()
        
        return d_t

    def update_drift_score(self, student_id, X_t):
        """
        Updates the EWMA drift score D(t) for a given student.
        """
        if student_id not in self.student_baselines:
            raise ValueError("Student baseline not set. Call train_baseline first.")
            
        baseline = self.student_baselines[student_id]
        d_t = self.calculate_instantaneous_deviation(student_id, X_t)
        
        # d_t normalized Z-score
        z_score = (d_t - baseline['mu_error']) / baseline['sigma_error']
        
        # EWMA
        prev_D = self.current_drift_scores[student_id]
        D_t = self.alpha * z_score + (1 - self.alpha) * prev_D
        self.current_drift_scores[student_id] = D_t
        
        return D_t

    def evaluate_threshold(self, D_t):
        """
        Multi-Tiered Z-Score Thresholding Mechanism.
        """
        if D_t <= 1.5:
            return "Zone 0: Nominal", "Passive monitoring"
        elif 1.5 < D_t <= 2.5:
            return "Zone 1: Micro-Drift", "Trigger Subtle Generative Nudge"
        elif 2.5 < D_t <= 3.5:
            return "Zone 2: Structural Drift", "Trigger ReAct Reasoning Loop"
        else:
            return "Zone 3: Critical Rupture", "Trigger Emergency Escalation"

    def generate_micro_warning(self, student_id: str, D_t: float, X_t: np.ndarray) -> dict:
        """
        Generates a lightweight, template-based nudge for Zone 1 drift
        without requiring the full ReAct / GenAI pipeline.
        X_t = [pace, lag, hesitation, volatility]
        """
        pace, lag, hesitation, volatility = X_t
        
        warning_type = ""
        action = ""
        text = ""
        anomaly = ""
        
        if 1.0 < D_t <= 1.5:
            warning_type = "Encouragement Nudge"
            action = "passive_notification"
            if pace < 0.8:
                anomaly = "Slight Pace Drop"
                text = "Hey! You usually cover a bit more ground by this time of the week. Just 15 minutes today keeps your momentum going!"
            else:
                anomaly = "General Friction"
                text = "You're doing great. Keep up the consistent effort!"
                
        elif 1.5 < D_t <= 2.0:
            warning_type = "Reminder Message"
            action = "active_prompt"
            if lag > 2.0:
                anomaly = "Assignment Backlog Forming"
                text = "Just a heads-up: You have an assignment pending. Want to tackle the first question now?"
            else:
                anomaly = "Login Irregularity"
                text = "We missed you yesterday! The next module is unlocked and ready for you."
                
        elif 2.0 < D_t <= 2.5:
            warning_type = "Minor Schedule Adjustment"
            action = "extend_deadline_24h"
            anomaly = "High Session Volatility"
            text = "Looks like a busy week! We've automatically shifted your next deadline back by 24 hours to give you some breathing room."
            
        return {
            "warning_type": warning_type,
            "drift_score_trigger": round(D_t, 2),
            "detected_anomaly": anomaly,
            "suggested_action": action,
            "student_notification_text": text
        }

if __name__ == "__main__":
    # Test the module
    detector = BehavioralDriftDetector(alpha=0.3, baseline_window=10)
    student = "STU_1001"
    
    # Mock baseline data (10 days of normal activity)
    normal_data = np.random.normal(loc=[1.0, 0.5, 30, 0.2], scale=[0.1, 0.1, 5, 0.05], size=(10, 4))
    detector.train_baseline(student, normal_data)
    print(f"Baseline trained. Mu: {detector.student_baselines[student]['mu_error']:.3f}")
    
    # Mock normal day
    X_normal = np.array([0.9, 0.6, 28, 0.25])
    D_normal = detector.update_drift_score(student, X_normal)
    print(f"Normal day D(t): {D_normal:.3f} -> {detector.evaluate_threshold(D_normal)[0]}")
    
    # Mock highly anomalous day (Drift)
    X_drift = np.array([0.2, 4.0, 200, 1.8]) # Slow pace, high lag, high hesitation, high volatility
    for _ in range(3): # Drifting over 3 days
        D_drift = detector.update_drift_score(student, X_drift)
        zone, action = detector.evaluate_threshold(D_drift)
        print(f"Drifting day D(t): {D_drift:.3f} -> {zone}: {action}")
        
        if "Zone 1" in zone:
            warning = detector.generate_micro_warning(student, D_drift, X_drift)
            print("  [Micro-Warning Fired]:", warning["student_notification_text"])
