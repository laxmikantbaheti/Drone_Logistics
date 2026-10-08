class ALNSConfig:
    ITERATIONS_SMALL = 200     # 10 - 25 nodes (e.g., A-n32-k5)[cite: 1]
    ITERATIONS_MEDIUM = 400    # 30 - 45 nodes (e.g., A-n45-k6)[cite: 1]
    ITERATIONS_LARGE = 600     # 45 - 55 nodes (e.g., A-n55-k9)[cite: 1]

    # Simulated Annealing settings
    START_TEMPERATURE = 800.0
    END_TEMPERATURE = 1.0
    STEP_DECAY = 0.985

    # Operator selection scores: [new_global_best, better, accepted, rejected]
    SCORES = [4.0, 2.0, 1.0, 0.5]
    DECAY = 0.8
    # Number of iterations before operator weights are updated from accumulated scores
    UPDATE_INTERVAL = 25  # or seg_length depending on signature