import streamlit as st
import numpy as np
import pandas as pd
import time
import sys
import os

# Ensure root import capability
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agentic_system.behavioral_drift.drift_detector import BehavioralDriftDetector
from agentic_system.risk_prediction.predictor import RiskPredictor
from agentic_system.react_planner.agent import ReActPlanner, StudentState
from agentic_system.rl_intervention.environment import ContextualBanditRLEngine
from agentic_system.ethical_ai.monitor import EthicalMonitor

st.set_page_config(page_title="AI Dropout Prevention", layout="wide")
st.title("🎓 Agentic AI Dropout Prevention System")
st.markdown("This dashboard demonstrates the proprietary technical mechanisms outlined in our patent claims: **Cognitive Struggle Index ($CSI_t$)**, **Self-Correcting Intervention Router**, **Counterfactual Risk Analysis**, and **Autonomous Pacing Governor**.")

# Initialize Session State
if 'detector' not in st.session_state:
    st.session_state.detector = BehavioralDriftDetector(alpha=0.3, baseline_window=10)
    st.session_state.predictor = RiskPredictor()
    st.session_state.planner = ReActPlanner()
    st.session_state.rl_engine = ContextualBanditRLEngine(action_space_size=len(st.session_state.planner.strategies))
    st.session_state.ethical_monitor = EthicalMonitor()
    st.session_state.students = {}

def setup_student(student_id):
    normal_data = np.random.normal(loc=[1.0, 0.5, 30, 0.2], scale=[0.1, 0.1, 5, 0.05], size=(10, 4))
    st.session_state.detector.train_baseline(student_id, normal_data)
    st.session_state.students[student_id] = {
        "intervention_history": [],
        "last_drift_score": 0.0,
        "last_time_to_dropout": 100
    }

st.sidebar.header("Simulation Scenarios")
scenario = st.sidebar.selectbox("Select Scenario", [
    "Scenario 1: Normal Engagement", 
    "Scenario 2: Gradual Decline", 
    "Scenario 3: Sudden Performance Drop (Emergency)"
])

if st.sidebar.button("Run Simulation", type="primary"):
    st.divider()
    
    if scenario == "Scenario 1: Normal Engagement":
        student_id = "Alice_001"
        days = 3
        behaviors = [np.random.normal(loc=[1.0, 0.5, 30, 0.2], scale=[0.1, 0.1, 5, 0.05]) for _ in range(3)]
        desc = "Student maintains normal engagement near baseline."
    elif scenario == "Scenario 2: Gradual Decline":
        student_id = "Bob_002"
        days = 3
        behaviors = [
            np.array([0.8, 1.5, 40, 0.4]),
            np.array([0.4, 4.0, 100, 1.2]),
            np.array([0.3, 5.0, 120, 1.5])
        ]
        desc = "Student progressively worsens over 3 days, triggering the Agent, safety critic, and reinforcement learning."
    else:
        student_id = "Charlie_003"
        days = 2
        behaviors = [
            np.array([1.0, 0.5, 30, 0.2]),
            np.array([0.1, 8.0, 300, 4.0])
        ]
        desc = "Student drops suddenly, bypassing standard protocols."
        
    setup_student(student_id)
    
    st.subheader(f"📊 Monitoring Student: {student_id}")
    st.caption(desc)
    
    for day in range(1, days + 1):
        behavior_vector = behaviors[day-1]
        st.markdown(f"### 🗓️ Day {day}")
        
        # Display behavioral input
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pace", f"{behavior_vector[0]:.2f}")
        col2.metric("Lag (Days)", f"{behavior_vector[1]:.2f}")
        col3.metric("Hesitation (ms)", f"{behavior_vector[2]:.2f}")
        col4.metric("Volatility", f"{behavior_vector[3]:.2f}")
        
        # Drift calculation
        drift_score = st.session_state.detector.update_drift_score(student_id, behavior_vector)
        zone, message = st.session_state.detector.evaluate_threshold(drift_score)
        
        with st.container(border=True):
            st.markdown(f"**Drift Score:** `{drift_score:.2f}` ➔ {zone}")
        
        # RL Feedback Loop
        history = st.session_state.students[student_id]["intervention_history"]
        student_data = st.session_state.students[student_id]
        
        if history and "success_score" not in history[-1]:
            s_true = max(0.01, 1.0 - (drift_score / 10.0))
            s_hat = max(0.01, 1.0 - (student_data["last_drift_score"] / 10.0))
            p_fatigue = len(history)
            
            reward = st.session_state.rl_engine.calculate_proprietary_reward(s_true=s_true, s_counterfactual=s_hat, p_fatigue=p_fatigue)
            action_idx = st.session_state.planner.strategies.index(history[-1]["strategy_used"])
            st.session_state.rl_engine.update_policy(action_idx, reward)
            history[-1]["success_score"] = 1.0 if reward > 0 else 0.1
            
            st.info(f"🧠 **Proprietary RL Router Feedback Loop ($R_{{intervene}}$):** Evaluated previous intervention (`{history[-1]['strategy_used']}`). Reward Scalar calculated: **{reward:.4f}**")

        student_data["last_drift_score"] = drift_score
        
        if drift_score > 2.5: # Triggers intervention
            st.warning("⚠️ **Significant drift detected. Escalating to Risk Prediction...**")
            
            risk_info = st.session_state.predictor.predict(behavior_vector, drift_score)
            
            r_col1, r_col2, r_col3 = st.columns(3)
            r_col1.metric("Dropout Probability", f"{risk_info['risk_score']:.2f}")
            r_col2.metric("Predicted Days Left", f"{risk_info['predicted_dropout_days']}")
            r_col3.metric("Dropout Type", f"{risk_info['classification']['dropout_type']}")
            st.markdown(f"**Top Predictive Flags:** `{', '.join(risk_info['top_contributing_features'])}`")
            
            # CSI Calculation usage
            mock_csi = st.session_state.predictor.calculate_csi(rewinds=5, difficulty_weight=1.2, hesitation_time=behavior_vector[2])
            st.markdown(f"**Cognitive Struggle Index ($CSI_t$):** `{mock_csi}`")
            
            student_data["last_time_to_dropout"] = risk_info['predicted_dropout_days']
            
            state = StudentState(
                drift_score=drift_score,
                drift_vector=behavior_vector.tolist(),
                dropout_prob=risk_info['risk_score'],
                time_to_dropout=risk_info['predicted_dropout_days'],
                context={"student_id": student_id, "current_module": day},
                intervention_history=history
            )
            
            # Ethical check
            is_fatigued, fatigue_msg = st.session_state.ethical_monitor.check_fatigue(student_id, history, day)
            if is_fatigued:
                st.error(f"🛡️ **Ethical Monitor:** {fatigue_msg} Skipping further interventions to avoid student fatigue.")
                continue
                
            # ReAct planning
            intervention_result = st.session_state.planner.execute_react_loop(state, risk_info["top_contributing_features"])
            action_params = intervention_result["action_parameters"]
            
            st.markdown(f"🤖 **ReAct Agentic Planner:** Root Cause Diagnosed as *'{intervention_result['root_cause']}'* | Selected Strategy: **{action_params['strategy']}**")
            
            critic_eval = intervention_result.get("critic_evaluation", {})
            if critic_eval:
                is_safe = critic_eval.get("is_safe", True)
                critic_msg = critic_eval.get("message", "Passed")
                if is_safe:
                    st.success(f"✅ **Critic Agent Validation PASSED:** {critic_msg}")
                else:
                    st.error(f"🚨 **Critic Agent Validation REJECTED (Safety Override):** {critic_msg}")
                    
            # Counterfactual sandbox
            cf_stats = st.session_state.predictor.simulate_intervention_impact(risk_info['risk_score'], action_params['strategy'])
            
            st.markdown(f"""
            **Sandboxed Counterfactual Risk Analysis ($X'_{{strategy}}$):** 
            - Action Baseline Risk: `{cf_stats['risk_without_intervention']:.2f}`
            - Projected New Risk: `{cf_stats['risk_with_intervention']:.2f}`
            - Delta Impact: **-{cf_stats['risk_reduction_percentage']:.1f}%**
            """)
            
            st.session_state.ethical_monitor.log_transparency(
                student_id=student_id,
                day=day,
                risk_score=risk_info['risk_score'],
                cause=intervention_result['root_cause'],
                strategy=action_params['strategy'],
                features=risk_info['top_contributing_features']
            )
            
            st.session_state.students[student_id]["intervention_history"].append({
                "strategy_used": action_params["strategy"],
                "timestamp": f"Day {day}"
            })
            
            with st.expander("Show GenAI Payload to Student"):
                st.json(intervention_result["generated_payload"])
                
        st.divider()
        time.sleep(1) # Simulation delay
