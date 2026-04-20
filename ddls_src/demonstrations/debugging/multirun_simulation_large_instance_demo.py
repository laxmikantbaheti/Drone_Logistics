import os
import sys
import time
from contextlib import redirect_stdout
import ddls_src.demonstrations.howto_log_simulation as simulation


def run_multiple_times(num_runs):
    print(f"Script B: Starting execution of script_a {num_runs} times.")
    print("Script B: All output from script_a will be entirely MUTED.\n")

    # Capture the absolute start time before any runs begin
    total_start_time = time.time()

    for i in range(num_runs):
        # Capture the time immediately before the individual run starts
        start_time = time.time()

        # The 'with' block temporarily catches ALL standard print statements
        # happening anywhere in Python and throws them into the void (devnull).
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            simulation.run_scenario_demo()

        # Capture the time immediately after the individual run finishes
        end_time = time.time()

        # Calculate the individual run duration
        run_duration = end_time - start_time

        # Once outside the 'with' block, printing is restored automatically
        print(f"Script B: Completed run {i + 1} of {num_runs} | Runtime: {run_duration:.4f} seconds")

    # Capture the absolute end time after all runs are complete
    total_end_time = time.time()

    # Calculate the total duration for the entire batch
    total_duration = total_end_time - total_start_time

    print("\nScript B: All runs finished successfully!")
    print(f"Script B: Total execution time for all {num_runs} runs: {total_duration:.4f} seconds")


if __name__ == "__main__":
    # Specify how many times you want the scenario to run
    run_multiple_times(500)