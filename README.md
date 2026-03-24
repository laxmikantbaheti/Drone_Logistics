# Drone Logistics Simulation Framework

## Overview

This project is a flexible, hierarchical simulation framework for modeling complex logistics scenarios involving trucks, drones, and micro-hubs. Built upon the MLPro library, it provides a powerful platform for research in reinforcement learning, allowing for a clear separation between an agent's decision-making and the environment's internal logic. The framework is event-driven, supports dynamic scenario generation, and features a robust action masking system to ensure agents only select valid actions.

## Key Features

  * **Hierarchical Architecture**: The simulation is structured with a top-level `LogisticsSystem` that orchestrates various managers (Supply Chain, Resource, Network), which in turn control the simulation's entities (Vehicles, Orders, Nodes).
  * **Configurable Scenarios**: Scenarios can be defined using JSON files or generated procedurally with a `RandomDataGenerator`, allowing for a wide range of testing environments.
  * **Dynamic Action Masking**: A powerful `ActionMasker` uses a system of pluggable constraints to generate a boolean mask of all possible actions at any given moment, ensuring agents operate within the simulation's rules.
  * **Automatic Logic Engine**: The framework can automate parts of the simulation logic (e.g., automatically routing a truck after it's assigned an order) through a simple configuration file (`AUTOMATIC_LOGIC_CONFIG`), allowing researchers to focus the agent's task on specific decisions.
  * **Flexible Research Designs**: The separation of agent and environment logic allows for different research setups, such as an "assignment-only" agent that only decides which vehicle gets which order, while the environment handles the rest.
  * **Live Visualization**: The simulation can render the network and the real-time movement of vehicles for better insights and debugging.
  * **Matrix-Based Movement**: In addition to graph-based routing, the simulation supports a distance-matrix mode for scenarios where precise edge/path modeling is not required.

## Project Structure

```
/
├── agents/             # Contains agent implementations (e.g., dummy agent, assignment-only agent)
├── appendices/         # Additional documentation on project logic and actions
├── ddls_src/
│   ├── actions/        # Defines the action space, constraints, and masking logic
│   ├── config/         # Default configurations and scenario data files
│   ├── core/           # Core simulation engine (LogisticsSystem, GlobalState, Network)
│   ├── entities/       # The building blocks of the simulation (vehicles, orders, nodes)
│   ├── functions/      # Utility functions like plotting
│   ├── howtos/         # Example scripts demonstrating various features
│   ├── managers/       # High-level controllers for different aspects of the simulation
│   └── scenarios/      # Scripts for setting up and running simulation scenarios
└── main.py             # Main entry point to run a simulation
```

## Getting Started

The easiest way to get started is to run one of the provided "howto" scripts, which demonstrate different functionalities of the framework.

### Running a Distance Matrix-Based Scenario

To demonstrate the simulation, run the `howto_log_simulation.py` script:

```bash
python -m ddls_src.demonstrations.howto_log_simulation
```

To demonstrate the RL extensibility and RL training demonstration, run the `rl_training.py` script:

```bash
python -m ddls_src.demonstrations.rl_training
```

This will load a specific scenario configuration that defines a `distance_matrix` and sets the `movement_mode` to `"matrix"`. You can observe from the console output how vehicles become idle after their travel timer expires, rather than traversing a graph path.

## Core Concepts

The simulation operates in a two-phase cycle: a **Decision Phase** and a **Progression Phase**.

1.  **Decision Phase**:

      * The `ActionMasker` generates a mask of all valid low-level actions based on the current world state.
      * This system mask is translated into a simplified `agent_mask` that corresponds to the agent's defined action space.
      * The agent receives this mask and chooses a valid action.

2.  **Progression Phase**:

      * The agent's chosen action is translated and dispatched to the appropriate manager (e.g., `SupplyChainManager`) for execution, which updates the state.
      * The system then enters an "automatic action loop," where it repeatedly checks for any actions that are both possible (based on the new mask) and marked as `True` in the `AUTOMATIC_LOGIC_CONFIG`. It executes these until no more automatic actions are available.
      * Finally, the simulation clock is advanced, and all entities with time-based dynamics (like moving vehicles) update their state.


## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
