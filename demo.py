import sys
import os

# Ensure the root project directory is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_system.demo_runner import run_scenarios

if __name__ == "__main__":
    print("Starting Agentic Dropout Prevention System Demo...\n")
    run_scenarios()
