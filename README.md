# Sequential Two-Echelon Capacitated Heterogeneous Vehicle Routing with Drones, Direct Deliveries, and Satellite Replenishment Hubs (2E-CVRP-D-DD)

Official code, synthesized benchmark instances, constraint-masked simulation environment, and pretrained Maskable PPO checkpoints accompanying the manuscript:

> **Sequential Two-Echelon Capacitated Heterogeneous Vehicle Routing with Drones, Direct Deliveries and Satellite Replenishment Hubs using Feasibility-Enforced Reinforcement Learning**  
> Laxmikant Shrikant Baheti, Andreas Schwung, and Stefan Lier  
> *Department of Logistics and Supply Chain Management & Department of Automation Technology, South Westphalia University of Applied Sciences (FH Südwestfalen), Germany*  
> *(Submitted to Transportation Research Part C: Emerging Technologies)*

---

## Overview

Urban parcel delivery combining ground trucks and autonomous drones faces a structural trade-off: synchronized tandem truck-drone systems suffer from truck idling during rendezvous, whereas strictly segregated two-echelon routing prevents primary trucks from serving customers directly along their routes.

This repository provides the simulation, training, and benchmarking framework for the **Two-Echelon Capacitated Vehicle Routing Problem with Drones and Direct Deliveries (2E-CVRP-D-DD)**:
- **Dual-Role First-Echelon Ground Trucks:** Serve heavy or route-aligned customers directly along Manhattan ($L_1$) road networks while delivering consolidated replenishment freight to stationary satellite micro-hubs.
- **Asynchronous Second-Echelon Drone Sorties:** Stationed drones execute closed-loop out-and-back deliveries along Euclidean ($L_2$) corridors bounded by drone payload capacity ($Q_{\text{drone}} = 0.20 \cdot Q_{\text{truck}}$) and maximum flight range ($R_{\text{max}}$).
- **Feasibility-Enforced Dispatching via Maskable PPO:** Formulates routing as a sequential Markov Decision Process (MDP) solved in a single greedy forward pass (`1-Pass`, $\approx 0.40\text{ s}$/instance) using Maskable Proximal Policy Optimization (`sb3-contrib`) paired with a reference-counted dynamic action mask that guarantees physical feasibility by construction.

---

## Repository Structure

```text
.
├── ddls_src/                                # Core simulation engine, action masking & benchmark data
│   ├── actions/                             # Base & conditional dispatch action definitions
│   ├── alns_benchmark/                      # Mask-compliant Priority-ALNS metaheuristic & operators
│   ├── config/                              # Entity & network JSON configurations
│   ├── core/                                # LogisticsSystem, ConstraintManager, GlobalState & Network
│   ├── entities/                            # Depot, Customers, MicroHubs, Orders & 1E/2E Vehicles
│   ├── functions/                           # Event logging, reporting & trajectory plotting utilities
│   ├── managers/
│   │   ├── action_masking/                  # Reference-counted dynamic feasibility mask (ActionMasker)
│   │   └── resource_manager/                # FleetManager & MicroHubManager
│   └── scenarios/
│       └── vrp_d_instances/
│           ├── VRP-D/                       # 27 synthesized Augerat Set A 2E-CVRP-D-DD instances (.vrp, .mh, .dist)
│           ├── VRP-D-REDUCED/               # Reduced-scale Augerat Set A & B test variants
│           ├── f2evrpd.py                   # 2E-CVRP-D-DD scenario loader & environment builder
│           └── vrpd_instance_generator.py   # Constrained mini-sum micro-hub placement generator
├── rl_ext/                                  # Gymnasium & Maskable PPO reinforcement learning extension
│   ├── env.py                               # RL environment wrapper with dynamic action masking
│   ├── observations.py                      # Flattened state vector construction
│   ├── rewards.py                           # Normalized episodic cost reward formulation
│   └── training/
│       ├── maskable_ppo_f2evrpd.py          # Training script for 2E-CVRP-D-DD Maskable PPO policies
│       └── maskable_ppo_vrpd_instances.py   # Training script for 1E-CVRP calibration policies
├── demonstrations/                          # Evaluation, baseline & sensitivity analysis scripts
│   ├── evaluate_maskable_ppo.py             # Deterministic single-pass evaluation on 2E-CVRP-D-DD
│   ├── evaluation_loop_sa.py                # Fleet-sizing (k = 1..10) & topology (S1 vs. S2) sweep
│   ├── greedy_dispatch.py                   # Nearest Demand & Largest Demand constructive baselines
│   └── execution_latency.py                 # Single-thread CPU runtime benchmarking
├── models/                                  # Pretrained Maskable PPO checkpoints (.zip)
│   ├── final_MaskablePPO_A_n*_model.zip     # 27 pretrained 2E-CVRP-D-DD models (A-n32-k5 .. A-n80-k10)
│   └── truck_only_models/                   # 27 pretrained 1E-CVRP truck-only calibration models
├── results/                                 # Result visualization & plotting scripts
├── LICENSE                                  # Apache-2.0 License
└── README.md
```

---

## Benchmark Instances (`ddls_src/scenarios/vrp_d_instances/VRP-D/`)

The 27 multimodal benchmark topologies are derived from **CVRPLib Augerat Set A** (`A-n32-k5` through `A-n80-k10`), stratified into Small ($N \le 40$, 10 instances), Medium ($41 \le N \le 60$, 9 instances), and Large ($N > 60$, 8 instances) tiers. Each instance comprises three synchronized files:

- `*.vrp` — Node coordinates, customer demands, depot index, and primary truck capacity ($Q_{\text{truck}}$).
- `*.mh` — Pre-positioned stationary micro-hub coordinates ($|\mathcal{S}| \in [1, 7]$) generated via continuous mini-sum spatial clustering under the closed-loop flight range bound $R_{\text{max}}$.
- `*.dist` — Precomputed surface ($L_1$ Manhattan) and aerial ($L_2$ Euclidean) distance matrices.

---

## Pretrained Models (`models/`)

To support immediate reproducibility without retraining from scratch, deterministic checkpoints for all 27 Augerat Set A topologies are included:

- **`models/final_MaskablePPO_<INSTANCE>_model.zip`**: Pretrained Maskable PPO policies for the full multimodal **2E-CVRP-D-DD** environment.
- **`models/truck_only_models/final_MaskablePPO_<INSTANCE>_model.zip`**: Pretrained single-echelon **1E-CVRP** policies used for CVRPLib BKS calibration and truck-only baseline comparisons.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/laxmikantbaheti/drone_logistics.git](https://github.com/laxmikantbaheti/drone_logistics.git)
   cd drone_logistics
   ```

2. **Create a Python 3.10+ virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   pip install numpy scipy pandas matplotlib gymnasium torch stable-baselines3 sb3-contrib
   ```

---

## Usage

### 1. Evaluate Pretrained Maskable PPO Policies (2E-CVRP-D-DD)

Run deterministic single-pass inference across the 2E-CVRP-D-DD benchmark suite using the pretrained checkpoints in `models/`:

```bash
python demonstrations/evaluate_maskable_ppo.py
```

### 2. Run Constructive & Metaheuristic Baselines

All baselines query the identical `ActionMasker` feasibility filter for fair comparison:

```bash
# Constructive dispatch rules (Nearest Demand & Largest Demand)
python demonstrations/greedy_dispatch.py

# Mask-compliant Priority-ALNS metaheuristic
python ddls_src/alns_benchmark/run_alns.py
```

### 3. Run Fleet-Sizing & Topology Sensitivity Analysis (`k = 1..10`)

Reproduce the Micro-Hub Only (`S1`) vs. Hybrid Depot-plus-Hub (`S2`) fleet sweep across $k \in \{1, \dots, 10\}$ drones per facility:

```bash
python demonstrations/evaluation_loop_sa.py
```

### 4. Train a Maskable PPO Policy from Scratch

Train a new Maskable PPO agent on a 2E-CVRP-D-DD instance:

```bash
python rl_ext/training/maskable_ppo_f2evrpd.py
```

---

## Citation

If you use this simulation environment, synthesized benchmark instances, or pretrained models in your research, please cite:

```bibtex
@article{baheti2026_2ecvrpddd,
  title   = {Sequential Two-Echelon Capacitated Heterogeneous Vehicle Routing with Drones, Direct Deliveries and Satellite Replenishment Hubs using Feasibility-Enforced Reinforcement Learning},
  author  = {Baheti, Laxmikant Shrikant and Schwung, Andreas and Lier, Stefan},
  journal = {Transportation Research Part C: Emerging Technologies},
  year    = {2026},
  note    = {Under review}
}
```

For details on the underlying differential, reference-counted dynamic action-masking framework, please also cite:

```bibtex
@article{baheti2026_masking,
  title   = {A Modular Constraint-Aware Simulation and Deep Reinforcement Learning Environment for Two-Echelon Drone Logistics},
  author  = {Baheti, Laxmikant Shrikant and Schwung, Andreas and Lier, Stefan},
  year    = {2026}
}
```

---

## License

This project is licensed under the terms of the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details[cite: 2].
