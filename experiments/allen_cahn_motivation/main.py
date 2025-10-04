import os
import subprocess
from shared_parameters import *

def run_script(script_name):
    """Helper function to run a Python script."""
    print(f"Running {script_name}...")
    subprocess.run(["python", script_name], check=True)

def main():
    # Step 1: Check if data exists and is complete in the data_path folder
    if not os.path.exists(f"{data_path}/processed_data.pkl"):
        print(f"Data in '{data_path}' is incomplete or missing, running simulation.py...")
        run_script("simulation.py")
    else:
        print(f"Data in '{data_path}' is complete, skipping simulation.py...")

    run_script("data_processing.py")

    # Step 2: run the data processing script
    if not os.path.exists(f"{data_path}/processed_data.pkl"):
        print(f"Processed data in '{data_path}' is missing, running data_processing.py...")
        run_script("data_processing.py")
    else:
        print(f"Processed data in '{data_path}' is complete, skipping data_processing.py...")

    # Step 3: run the plotting script
    run_script("plotting.py")

if __name__ == "__main__":
    main()