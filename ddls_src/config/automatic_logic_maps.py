# Local Imports
from ..actions.action_enums import SimulationAction

# -------------------------------------------------------------------------
# -- Automatic Logic Configuration
# -------------------------------------------------------------------------
# This dictionary defines the "rulebook" for the simulation's automatic
# behavior. By changing the boolean values here, a researcher can precisely
# define the boundary between agent-controlled actions and environment-automated
# actions for any given experiment.
# -------------------------------------------------------------------------

AUTOMATIC_LOGIC_CONFIG = {

    # --- Strategic / Order Lifecycle Actions ---
    # These are typically set to False, as they represent the primary
    # strategic decisions made by the agent. Setting to True would create
    # a fully autonomous simulation based on internal rules.
    SimulationAction.ACCEPT_ORDER: False,
    SimulationAction.PRIORITIZE_ORDER: False,
    SimulationAction.CANCEL_ORDER: False,
    SimulationAction.FLAG_FOR_RE_DELIVERY: False,
    SimulationAction.ASSIGN_ORDER_TO_TRUCK: False,
    SimulationAction.ASSIGN_ORDER_TO_DRONE: False,
    SimulationAction.ASSIGN_ORDER_TO_MICRO_HUB: False,
    SimulationAction.REASSIGN_ORDER: False,

    # --- Consolidation Actions ---
    # If True, a vehicle at a node with packages could automatically consolidate them.
    SimulationAction.CONSOLIDATE_FOR_TRUCK: False,
    SimulationAction.CONSOLIDATE_FOR_DRONE: False,

    # --- Routing Actions ---
    # If True, the NetworkManager will automatically route a vehicle
    # after it has been assigned an order.
    SimulationAction.TRUCK_TO_NODE: True,
    SimulationAction.RE_ROUTE_TRUCK_TO_NODE: True,
    SimulationAction.DRONE_TO_NODE: True,
    SimulationAction.RE_ROUTE_DRONE_TO_NODE: True,
    SimulationAction.LAUNCH_DRONE: True,
    SimulationAction.DRONE_TO_CHARGING_STATION: True,

    # --- Loading/Unloading Actions ---
    # If True, a vehicle will automatically load/unload its cargo
    # upon arriving at the correct node.
    SimulationAction.LOAD_TRUCK_ACTION: True,
    SimulationAction.UNLOAD_TRUCK_ACTION: True,
    SimulationAction.DRONE_LOAD_ACTION: True,
    SimulationAction.DRONE_UNLOAD_ACTION: True,
    SimulationAction.DRONE_LANDING_ACTION: True,

    # --- Micro-Hub & Maintenance Actions ---
    # If True, the system could, for example, automatically activate a hub
    # when a nearby order is received or flag a vehicle for maintenance.
    SimulationAction.ACTIVATE_MICRO_HUB: False,
    SimulationAction.DEACTIVATE_MICRO_HUB: False,
    SimulationAction.ADD_TO_CHARGING_QUEUE: True,
    SimulationAction.DRONE_CHARGE_ACTION: True,
    SimulationAction.FLAG_VEHICLE_FOR_MAINTENANCE: False,
    SimulationAction.FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB: False,

    # --- Special Actions ---
    # NO_OPERATION is a specific agent command and should never be automated.
    SimulationAction.NO_OPERATION: False,
}
