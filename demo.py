import sys
import os
import subprocess

# Ensure the root project directory is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Starting Agentic Dropout Prevention System Streamlit Dashboard...\n")
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentic_system", "dashboard.py")
    
    # Launch Streamlit dashboard
    try:
        subprocess.run(["streamlit", "run", dashboard_path], check=True)
    except FileNotFoundError:
        print("Error: 'streamlit' command not found. Please ensure Streamlit is installed (pip install streamlit).")
    except Exception as e:
        print(f"An error occurred while launching the dashboard: {e}")
