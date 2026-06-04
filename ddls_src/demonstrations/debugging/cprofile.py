import cProfile
import pstats
from ddls_src.demonstrations.howto_vrpd_demonstration import run_matrix_scenario_demo

pr = cProfile.Profile()
pr.enable()

# Your code here
run_matrix_scenario_demo("A-n80-k10.vrp", 0,0)

pr.disable()

# Save to a readable text file
with open('profile_results.txt', 'w') as f:
    ps = pstats.Stats(pr, stream=f)
    ps.sort_stats('tottime')  # Sort by cumulative time
    ps.print_stats()