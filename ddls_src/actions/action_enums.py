from enum import Enum, auto

class SimulationAction(Enum):
    """
    Defines all possible actions and specific sub-types/parameters within a single Enum.
    This consolidates all action-related enumerations into one class.
    """
    # -------------------------------------------------------------------------
    # -- Supply Chain Manager Actions
    # -------------------------------------------------------------------------
    ACCEPT_ORDER = auto()
    PRIORITIZE_ORDER = auto()
    CANCEL_ORDER = auto()
    FLAG_FOR_RE_DELIVERY = auto()
    ASSIGN_ORDER_TO_TRUCK = auto()
    ASSIGN_ORDER_TO_DRONE = auto()
    ASSIGN_ORDER_TO_MICRO_HUB = auto()
    REASSIGN_ORDER = auto()
    CONSOLIDATE_FOR_TRUCK = auto()
    CONSOLIDATE_FOR_DRONE = auto()

    # -------------------------------------------------------------------------
    # -- Resource Manager Actions (Fleet & Hubs)
    # -------------------------------------------------------------------------
    LOAD_TRUCK_ACTION = auto()
    UNLOAD_TRUCK_ACTION = auto()
    DRONE_LOAD_ACTION = auto()
    DRONE_UNLOAD_ACTION = auto()
    DRONE_CHARGE_ACTION = auto()
    ACTIVATE_MICRO_HUB = auto()
    DEACTIVATE_MICRO_HUB = auto()
    ADD_TO_CHARGING_QUEUE = auto()
    FLAG_VEHICLE_FOR_MAINTENANCE = auto()
    FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB = auto()

    # -------------------------------------------------------------------------
    # -- Network Manager Actions
    # -------------------------------------------------------------------------
    TRUCK_TO_NODE = auto()
    RE_ROUTE_TRUCK_TO_NODE = auto()
    LAUNCH_DRONE = auto()
    DRONE_LANDING_ACTION = auto()
    DRONE_TO_CHARGING_STATION = auto()

    # -------------------------------------------------------------------------
    # -- Special Actions
    # -------------------------------------------------------------------------
    NO_OPERATION = auto()

    # -------------------------------------------------------------------------
    # -- Parameter Enums (used in action tuples)
    # -------------------------------------------------------------------------
    RESOURCE_CHARGING_SLOT = auto()
    RESOURCE_PACKAGE_SORTING_SERVICE = auto()
    RESOURCE_LAUNCHES = auto()
    RESOURCE_RECOVERIES = auto()

    def __str__(self):
        return self.name


if __name__ == "__main__":
    for action in SimulationAction:
        print(action.value, "->", action.name)
