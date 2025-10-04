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

    # Step 2: process the data
    run_script("data_processing.py")

if __name__ == "__main__":
    main()