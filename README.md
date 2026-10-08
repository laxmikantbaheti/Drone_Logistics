# Modular Constraint Based Simulation and Reinforcement Learning Environment of a Truck-Drone Co-ordinated Delivery Logistics System with Micro Hubs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Paper: MLWA](https://img.shields.io/badge/MLWA-Major--Revision-orange.svg)](https://www.editorialmanager.com/mlwa/)

This repository provides the official implementation, simulation testbed, and experimental validation scripts for the manuscript:

> **"Modular Constraint Based Simulation and Reinforcement Learning Environment of a Truck-Drone Co-ordinated Delivery Logistics System with Micro Hubs"**  
> *Machine Learning with Applications (MLWA)*, Manuscript No.: `MLWA-D-26-01308`.

---

## 📁 Repository Structure

```text
├── demonstrations/               # Demonstration and experiment reproduction scripts
│   ├── demonstration_simulation.py               # 1. Base simulation workflow with distance matrix
│   ├── evaluate_maskable_ppo_dymamic_pertubation.py # 2. Evaluation on stochastic, perturbed demands
│   ├── execution_latency.py                      # 3. Computational throughput & constraint profiling
│   ├── greedy_dispatch.py                        # 4. System-decoded greedy nearest-demand heuristic
│   ├── maskable_ppo_dynamic_pertubation.py       # 5. Maskable PPO training with dynamic perturbations
│   ├── maskable_ppo_random_instances.py          # 6. Procedural random instance single-seed training
│   ├── maskable_ppo_random_multi_seed.py         # 7. Multi-seed training (5 seeds) for statistical curves
│   ├── ppo_without_masking.py                    # 8. Unmasked PPO baseline (immediate termination)
│   ├── ppo_without_masking_no_termination.py     # 9. Unmasked PPO baseline (penalty retry up to threshold)
│   └── models/                                   # Pre-trained policy checkpoints (e.g., n=60, k=10)
├── ddls_src/                     # Core simulation & environment source code
│   ├── core/                     # Constraint manager, system clock, state representation
│   ├── entities/                 # Trucks, drones, micro-hubs, nodes, edges, orders
│   ├── managers/                 # Fleet, resource, and action-masking managers
│   └── scenarios/                # Instance loaders, procedural generators, VRP-D datasets
├── rl_ext/                       # RL extensions, Gym wrappers, reward models, training loops
├── alns_benchmark/               # Adaptive Large Neighborhood Search (ALNS) comparison suite
└── hwga/                         # Hybrid genetic algorithm baselines
```

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.9 or higher
* Recommended: Virtual environment (`venv` or `conda`)

```bash
# Clone the repository
git clone [https://github.com/laxmikantbaheti/drone_logistics.git](https://github.com/laxmikantbaheti/drone_logistics.git)
cd drone_logistics

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 🚀 Running Demonstrations & Experiments

The scripts located in the `demonstrations/` directory directly reproduce the experimental results and validations reported in Section 4 of the manuscript:

### 1. Simplified Distance Matrix Simulation Demo

```bash
python -m demonstrations.demonstration_simulation
```

---

### 2. Evaluate Trained Maskable PPO on Dynamic Perturbations
```bash
python -m demonstrations.evaluate_maskable_ppo_dymamic_pertubation --model_path models/maskable_ppo_n60_k10.zip --episodes 5 --seed 42 --numnodes 60
```
* `--model_path`: Relative or absolute path to the pre-trained `.zip` model (default: `models/maskable_ppo_n60_k10.zip`).
* `--episodes`: Number of evaluation episodes to average over (default: `1`).
* `--seed`: Initialization seed for procedural instance generation (default: `42`).
* `--numnodes`: Number of network nodes generated for the test scenario (default: `60`).

---

### 3. Execution Latency & Constraint Profiling (Section 4.4, Table 5)
```bash
python -m demonstrations.execution_latency
```

---

### 4. Baseline: System-Decoded Greedy Dispatch Heuristic
```bash
python -m demonstrations.greedy_dispatch --episodes 10 --seed 42
```
* `--episodes`: Number of benchmark episodes (default: `10`).
* `--seed`: Random seed for evaluation (default: `42`).

---

### 5. Maskable PPO Training with Dynamic Demand Perturbations
```bash
python -m demonstrations.maskable_ppo_dynamic_pertubation --timesteps 500000 --config ddls_src/config/large_instance.json
```
* `--timesteps`: Total RL training timesteps (default: `5000000`).
* `--config`: Path to the base simulation configuration file (default: `ddls_src/config/large_instance.json`).

---

### 6. Maskable PPO Training on Procedural Random Instances (Single Seed)
```bash
python -m demonstrations.maskable_ppo_random_instances --nodes 60 --timesteps 1000000 --seed 42
```
* `--nodes`: Number of network nodes (>= 50, default: `60`).
* `--timesteps`: Total RL training timesteps (default: `1000000`).
* `--seed`: Deterministic global seed (default: `42`).

---

### 7. Multi-Seed Training Pipeline (Section 4.3, Figure 8)
```bash
python -m demonstrations.maskable_ppo_random_multi_seed --nodes 60 --timesteps 1000000
```
* `--nodes`: Number of network nodes to generate (default: `60`).
* `--timesteps`: Total training timesteps allocated per seed (default: `1000000`).

---

### 8. Ablation: Standard PPO without Action Masking (Immediate Termination)
```bash
python -m demonstrations.ppo_without_masking --timesteps 1000000 --config ddls_src/config/large_instance.json
```
* `--timesteps`: Total RL training timesteps (default: `5000000`).
* `--config`: Simulation configuration file path.

---

### 9. Ablation: Standard PPO without Action Masking (Penalty Retry)
```bash
python -m demonstrations.ppo_without_masking_no_termination --timesteps 1000000 --config ddls_src/config/large_instance.json
```
* `--timesteps`: Total RL training timesteps (default: `5000000`).
* `--config`: Simulation configuration file path.


## 📜 Citation & License

This codebase is licensed under the [MIT License](LICENSE)[cite: 1].

If you find this environment or experimental baseline implementations helpful in your research, please cite:

```bibtex
@article{baheti2026modular,
  title={Modular Constraint Based Simulation and Reinforcement Learning Environment of a Truck-Drone Co-ordinated Delivery Logistics System with Micro Hubs},
  author={Baheti, Laxmikant and Schwung, Andreas and Dircksen, Michael and Lier, Stefan},
  journal={Machine Learning with Applications},
  year={2026},
  note={Under Revision}
}
```