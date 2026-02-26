import pandas as pd
import numpy as np

def simulate_rl_transitions(ts_df, intervention_boost_effect=1.5, dropout_penalty=-10, retention_reward=1):
    """
    Creates (State, Action, Reward, Next_State) tuples for offline RL training.
    Injects random action exploration and modifies the trajectory.
    """
    records = []
    # Ensure dataframe is sorted chronologically per student
    ts_df = ts_df.sort_values(['id_student', 'week']).reset_index(drop=True)
    
    # Create shifted columns for 'Next State'
    # By shifting by -1, we pull the next week's data up to the current row
    # We must ensure we don't bleed data across different students
    ts_df['next_student'] = ts_df['id_student'].shift(-1)
    
    # Only keep rows where the current row and the next row belong to the same student
    valid_transitions = ts_df[ts_df['id_student'] == ts_df['next_student']].copy()
    
    # Get actual next state values
    valid_transitions['actual_next_clicks'] = ts_df['sum_click'].shift(-1)[valid_transitions.index]
    valid_transitions['actual_next_hesitation'] = ts_df['synthesized_hesitation_sec'].shift(-1)[valid_transitions.index]
    
    # Simulate RL Action (0 = Null, 1 = Intervene) vectorized
    valid_transitions['action'] = np.random.choice([0, 1], size=len(valid_transitions), p=[0.7, 0.3])
    
    # Calculate Next State Engagement based on action 
    # If action == 1, multiply by boost. If 0, multiply by 1
    action_mask = valid_transitions['action'] == 1
    
    valid_transitions['next_state_clicks'] = valid_transitions['actual_next_clicks']
    valid_transitions.loc[action_mask, 'next_state_clicks'] *= intervention_boost_effect
    
    valid_transitions['next_state_hesitation'] = valid_transitions['actual_next_hesitation']
    valid_transitions.loc[action_mask, 'next_state_hesitation'] *= 0.8
    
    # Compute Reward Vectorized
    # Base reward is retention
    valid_transitions['reward'] = retention_reward
    
    # Apply penalty condition: next_clicks < 5.0 AND current state drift < 0
    penalty_mask = (valid_transitions['next_state_clicks'] < 5.0) & (valid_transitions.get('drift_idx', 0) < 0)
    valid_transitions.loc[penalty_mask, 'reward'] = dropout_penalty
    
    # Build final DataFrame
    records_df = pd.DataFrame({
        'id_student': valid_transitions['id_student'],
        'week_t': valid_transitions['week'],
        'state_clicks': valid_transitions['sum_click'],
        'state_drift': valid_transitions.get('drift_idx', 0.0),
        'state_volatility': valid_transitions.get('volatility_idx', 0.0),
        'action': valid_transitions['action'],
        'reward': valid_transitions['reward'],
        'next_state_clicks': valid_transitions['next_state_clicks'],
        'next_state_hesitation': valid_transitions['next_state_hesitation']
    })
    
    return records_df
