from enum import Enum, auto

class SimulationAction(Enum):
    """
    Defines all possible actions and specific sub-types/parameters within a single Enum.
    This consolidates all action-related enumerations into one class.
    """
    # Main Action Types
    ACCEPT_ORDER = auto()
    PRIORITIZE_ORDER = auto()
    CANCEL_ORDER = auto()
    FLAG_FOR_RE_DELIVERY = auto()

    ASSIGN_ORDER_TO_TRUCK = auto()
    ASSIGN_ORDER_TO_DRONE = auto()
    ASSIGN_ORDER_TO_MICRO_HUB = auto()

    CONSOLIDATE_FOR_TRUCK = auto()
    CONSOLIDATE_FOR_DRONE = auto()

    REASSIGN_ORDER = auto()

    TRUCK_TO_NODE = auto()
    RE_ROUTE_TRUCK_TO_NODE = auto()

    LOAD_TRUCK_ACTION = auto()
    UNLOAD_TRUCK_ACTION = auto()

    LAUNCH_DRONE = auto()
    DRONE_LANDING_ACTION = auto()
    DRONE_TO_CHARGING_STATION = auto()

    DRONE_LOAD_ACTION = auto()
    DRONE_UNLOAD_ACTION = auto()
    DRONE_CHARGE_ACTION = auto()

    ACTIVATE_MICRO_HUB = auto()
    DEACTIVATE_MICRO_HUB = auto()
    ADD_TO_CHARGING_QUEUE = auto()

    FLAG_VEHICLE_FOR_MAINTENANCE = auto()
    FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB = auto()

    NO_OPERATION = auto()

    # Truck Specific Sub-Actions (used as parameters where applicable)
    TRUCK_LOAD_PARAM = auto() # Represents the 'LOAD' parameter for truck actions
    TRUCK_UNLOAD_PARAM = auto() # Represents the 'UNLOAD' parameter for truck actions

    # Drone Specific Sub-Actions (used as parameters where applicable)
    DRONE_LOAD_PARAM = auto() # Represents the 'LOAD' parameter for drone actions
    DRONE_UNLOAD_PARAM = auto() # Represents the 'UNLOAD' parameter for drone actions
    DRONE_CHARGE_PARAM = auto() # Represents the 'CHARGE' parameter for drone actions
    DRONE_LANDING_PARAM = auto() # Represents the 'LANDING' parameter for drone actions

    # Resource Specific Types (used as parameters where applicable)
    RESOURCE_CHARGING_SLOT = auto()
    RESOURCE_PACKAGE_SORTING_SERVICE = auto()
    RESOURCE_LAUNCHES = auto()
    RESOURCE_RECOVERIES = auto()

    def __str__(self):
        return self.name

