from mlpro.bf.math import MSpace, Dimension


def create_agent_action_space(num_orders: int, num_vehicles: int) -> MSpace:
    """
    Creates the simplified, agent-facing action space for the assignment-only research design.

    The action space is a single discrete integer, where each integer represents a unique
    (order_id, vehicle_id) assignment pair.

    Parameters:
        num_orders (int): The total number of orders the agent can assign.
        num_vehicles (int): The total number of vehicles the agent can assign to.

    Returns:
        MSpace: The MLPro action space for the agent.
    """
    action_space_size = num_orders * num_vehicles

    action_space = MSpace()
    action_space.add_dim(Dimension(p_name_short='agent_action',
                                   p_base_set='Z',
                                   p_name_long='Agent Assignment Action',
                                   p_boundaries=[0, action_space_size - 1]))
    return action_space

