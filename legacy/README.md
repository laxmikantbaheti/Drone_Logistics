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

### Running a Full Simulation

To run a complete simulation with a simple dummy agent that chooses from all available actions, execute the `howto_run_framework.py` script:

```bash
python -m ddls_src.howto_v02.howto_run_framework
```

This script will:

1.  Load a scenario configuration from a JSON file.
2.  Instantiate the `LogisticsScenario`.
3.  Run the simulation for a predefined number of cycles.

### Running a Distance Matrix-Based Scenario

To see the matrix-based movement in action, run the `howto_distance_matrix.py` script:

```bash
python -m ddls_src.howto_v02.howto_distance_matrix
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

## "How-To" Examples

This project includes a rich set of `howto_*.py` scripts in the `ddls_src/howtos` and `ddls_src/entities/howtos` directories. These standalone scripts demonstrate the functionality of individual components and core concepts:

  * **`howto_run_framework.py`**: Runs the entire simulation from end-to-end.
  * **`howto_distance_matrix.py`**: Demonstrates timer-based movement using a distance matrix.
  * **`howto_action_masking.py`**: Shows how the action masker filters valid and invalid actions based on system state.
  * **`howto_assignment_only.py`**: An example of a specific research design where the agent only assigns orders.
  * **Entity `howtos`**: Scripts like `howto_truck.py`, `howto_drone.py`, and `howto_order.py` demonstrate the methods and properties of each simulation entity.
  * **Manager `howtos`**: Scripts like `howto_fleet_manager.py` and `howto_network_manager.py` show how to interact with the simulation's managers.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.