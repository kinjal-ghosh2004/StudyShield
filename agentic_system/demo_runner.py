import sys
import os
import time
import numpy as np

# Ensure root import capability
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agentic_system.behavioral_drift.drift_detector import BehavioralDriftDetector
from agentic_system.risk_prediction.predictor import RiskPredictor
from agentic_system.react_planner.agent import ReActPlanner, StudentState
from agentic_system.rl_intervention.environment import ContextualBanditRLEngine
from agentic_system.ethical_ai.monitor import EthicalMonitor

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

class DemoRunner:
    def __init__(self):
        console.print(Panel.fit("[bold cyan]Initializing Agentic AI Dropout Prevention System...[/bold cyan]", border_style="cyan"))
        self.detector = BehavioralDriftDetector(alpha=0.3, baseline_window=10)
        self.predictor = RiskPredictor()
        self.planner = ReActPlanner()
        self.rl_engine = ContextualBanditRLEngine(action_space_size=len(self.planner.strategies))
        self.ethical_monitor = EthicalMonitor()
        
        # Student memory store
        self.students = {}
        
    def setup_student(self, student_id: str):
        console.print(f"\n[bold green]--- Setting up Baseline for Student {student_id} ---[/bold green]")
        # Train baseline with normal data
        # [pace, lag, hesitation, volatility]
        normal_data = np.random.normal(loc=[1.0, 0.5, 30, 0.2], scale=[0.1, 0.1, 5, 0.05], size=(10, 4))
        self.detector.train_baseline(student_id, normal_data)
        
        self.students[student_id] = {
            "intervention_history": [],
            "last_drift_score": 0.0,
            "last_time_to_dropout": 100
        }
        console.print("[dim]Baseline established. Monitoring active.[/dim]")
        
    def process_day(self, student_id: str, behavior_vector: np.ndarray, day: int):
        day_text = Text(f"Day {day} | Student {student_id}", style="bold magenta")
        console.rule(day_text, style="magenta")
        console.print(f"[dim]Processing Behavior Vector:[/dim] [cyan]{np.round(behavior_vector, 2)}[/cyan]\n")
        
        # 1. Behavioral Drift Detection Layer
        drift_score = self.detector.update_drift_score(student_id, behavior_vector)
        zone, message = self.detector.evaluate_threshold(drift_score)
        
        # Determine color for drift score
        color = "green" if drift_score <= 1.5 else "yellow" if drift_score <= 2.5 else "red"
        
        details_table = Table(show_header=False, box=box.SIMPLE)
        details_table.add_column("Layer", style="bold bright_blue", width=25)
        details_table.add_column("Details")
        
        details_table.add_row("Drift Detection Layer", f"Score: [{color}]{drift_score:.2f}[/{color}] ➔ {zone}")
        
        # Feedback Loop: Process previous day's intervention effectiveness
        history = self.students[student_id]["intervention_history"]
        student_data = self.students[student_id]
        if history and "success_score" not in history[-1]:
            # Provide RL Feedback based on current drift vs yesterday's
            reward = self.rl_engine.calculate_reward(student_data["last_drift_score"], drift_score, student_data["last_time_to_dropout"], student_data["last_time_to_dropout"]) # Simplified Td mapping
            action_idx = self.planner.strategies.index(history[-1]["strategy_used"])
            self.rl_engine.update_policy(action_idx, reward)
            # Log success score back into memory for ReAct Reflection later
            history[-1]["success_score"] = 1.0 if reward > 0 else 0.1
            success_str = f"[bold green]{history[-1]['success_score']}[/bold green]" if history[-1]['success_score'] == 1.0 else f"[bold red]{history[-1]['success_score']}[/bold red]"
            details_table.add_row("RL Feedback Loop", f"Eval prev intervention '{history[-1]['strategy_used']}'. Success: {success_str}")
            
        student_data["last_drift_score"] = drift_score
        
        if drift_score > 2.5: # Triggers intervention
            details_table.add_row("", "[bold red]Significant drift detected. Escalating to Risk Prediction...[/bold red]")
            
            # 2. Risk Prediction Layer
            risk_info = self.predictor.predict(behavior_vector, drift_score)
            
            risk_str = f"Prob: [bold red]{risk_info['risk_score']:.2f}[/bold red] | Days Left: [bold yellow]{risk_info['predicted_dropout_days']}[/bold yellow] | Type: [bold]{risk_info['classification']['dropout_type']}[/bold]"
            flags_str = f"Top Flags: {risk_info['top_contributing_features']}"
            details_table.add_row("Risk Prediction Layer", risk_str + "\n" + flags_str)
            
            student_data["last_time_to_dropout"] = risk_info['predicted_dropout_days']
            
            # 3 & 4. Agentic Intervention & GenAI Layers
            state = StudentState(
                drift_score=drift_score,
                drift_vector=behavior_vector.tolist(),
                dropout_prob=risk_info['risk_score'],
                time_to_dropout=risk_info['predicted_dropout_days'],
                context={"student_id": student_id, "current_module": day},
                intervention_history=history
            )
            
            # Ethical Guardrail: Fatigue Check
            is_fatigued, fatigue_msg = self.ethical_monitor.check_fatigue(student_id, history, day)
            if is_fatigued:
                details_table.add_row("Ethical Monitor Layer", f"[bold yellow]{fatigue_msg}[/bold yellow] Skipping ReAct Planning to avoid fatigue.")
                console.print(details_table)
                return

            intervention_result = self.planner.execute_react_loop(state, risk_info["top_contributing_features"])
            action_params = intervention_result["action_parameters"]
            
            details_table.add_row("ReAct Planner Layer", f"Root Cause: [italic]{intervention_result['root_cause']}[/italic]\nSelected Strategy: [bold cyan]{action_params['strategy']}[/bold cyan]")
            
            critic_eval = intervention_result.get("critic_evaluation", {})
            if critic_eval:
                is_safe = critic_eval.get("is_safe", True)
                critic_msg = critic_eval.get("message", "Passed")
                status_color = "bold green" if is_safe else "bold red"
                details_table.add_row("Critic Validation", f"[{status_color}]{'PASSED' if is_safe else 'REJECTED'}[/{status_color}]: {critic_msg}")
            
            # Counterfactual Risk Analysis
            cf_stats = self.predictor.simulate_intervention_impact(risk_info['risk_score'], action_params['strategy'])
            cf_str = f"Base Risk: {cf_stats['risk_without_intervention']:.2f} | Est. New: {cf_stats['risk_with_intervention']:.2f} | Impact: [bold green]-{cf_stats['risk_reduction_percentage']:.1f}%[/bold green]"
            details_table.add_row("Counterfactual Analysis", cf_str)
            
            # Ethical Transparency Logging
            self.ethical_monitor.log_transparency(
                student_id=student_id,
                day=day,
                risk_score=risk_info['risk_score'],
                cause=intervention_result['root_cause'],
                strategy=action_params['strategy'],
                features=risk_info['top_contributing_features']
            )
            
            # Save to memory
            self.students[student_id]["intervention_history"].append({
                "strategy_used": action_params["strategy"],
                "timestamp": f"Day {day}"
            })
            
            console.print(details_table)
            
            # GenAI Output Display
            genai_table = Table(title="[bold green]GenAI Payload Sent to Student[/bold green]", box=box.ROUNDED, show_header=False, width=80)
            genai_table.add_column("Field", style="bold yellow", justify="right")
            genai_table.add_column("Content", style="white")
            for key, val in intervention_result["generated_payload"].items():
                if val:
                    genai_table.add_row(key.capitalize(), str(val))
            console.print(genai_table)
        else:
            console.print(details_table)


def run_scenarios():
    system = DemoRunner()
    
    # ---------------------------------------------------------
    # Scenario 1: Normal Engagement
    # ---------------------------------------------------------
    console.print(Panel("[bold white]SCENARIO 1: Normal Engagement (No Intervention)[/bold white]", style="bold grey53", expand=False))
    student_1 = "Alice_001"
    system.setup_student(student_1)
    for day in range(1, 4):
        # Hovering around baseline
        # [pace, lag, hesitation, volatility]
        behavior = np.random.normal(loc=[1.0, 0.5, 30, 0.2], scale=[0.1, 0.1, 5, 0.05])
        system.process_day(student_1, behavior, day)
        time.sleep(1)

    # ---------------------------------------------------------
    # Scenario 2: Gradual Decline
    # ---------------------------------------------------------
    console.print(Panel("[bold white]SCENARIO 2: Gradual Decline (Reflection Triggers)[/bold white]", style="bold orange1", expand=False))
    student_2 = "Bob_002"
    system.setup_student(student_2)
    # Day 1: Slight drift (Zone 1 - Triggers Micro Warning)
    system.process_day(student_2, np.array([0.8, 1.5, 40, 0.4]), 1)
    time.sleep(1)
    
    # Day 2: Worsening drift (Triggers Intervention)
    system.process_day(student_2, np.array([0.4, 4.0, 100, 1.2]), 2)
    time.sleep(1)
    
    # Day 3: Still bad (RL marks failure, ReAct reflects and escalates)
    system.process_day(student_2, np.array([0.3, 5.0, 120, 1.5]), 3)
    time.sleep(1)

    # ---------------------------------------------------------
    # Scenario 3: Sudden Performance Drop
    # ---------------------------------------------------------
    console.print(Panel("[bold white]SCENARIO 3: Sudden Performance Drop (Emergency)[/bold white]", style="bold red", expand=False))
    student_3 = "Charlie_003"
    system.setup_student(student_3)
    
    # Day 1: Totally normal
    system.process_day(student_3, np.array([1.0, 0.5, 30, 0.2]), 1)
    time.sleep(1)
    
    # Day 2: Massive spike in volatility and lag (Triggers Emergency Escalation due to low T_d estimation)
    system.process_day(student_3, np.array([0.1, 8.0, 300, 4.0]), 2)

if __name__ == "__main__":
    console.print("[bold yellow]Starting Agentic Dropout Prevention System Demo...[/bold yellow]\n")
    run_scenarios()
