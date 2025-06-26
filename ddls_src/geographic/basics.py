## -------------------------------------------------------------------------------------------------
## -- Project : --MLPro - A Synoptic Framework for Standardized Machine Learning Tasks--
## -- Package : mlpro-logistics.geographic
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-05-22  0.0.0     LSB       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-06-08)

This module provides the base implementations for the vechilc objects in the MLPro Logistics Framework.
"""


from mlpro.bf.systems import *
from typing import Tuple, List, Iterable





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Node(ScientificObject, Log, Id):


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id,
                 p_name = None,
                 p_location:Tuple[float] = None):

        """

        :param p_id:
        :param p_name:
        :param p_location:
        """

        Id.__init__(self,
                    p_id = p_id)

        self.location = p_location

        if p_name is None:
            self.name = f"Node {p_id}"
        else:
            self.name = p_name

        ScientificObject.__init__(self)
        Log.__init__(self)








## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Edge(ScientificObject, Log, Id):


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id,
                 p_name,
                 p_geometry,
                 p_bounding_nodes):

        Id.__init__(self,
                    p_id = p_id)

        self.name = p_name
        self._geometry = p_geometry
        self.bounding_nodes = p_bounding_nodes

        self.network = None


## -------------------------------------------------------------------------------------------------
    def get_bounding_nodes(self):
        """

        :return:
        """

        return self.bounding_nodes


## -------------------------------------------------------------------------------------------------
    def add_network(self,
                    p_network):

        """

        :param p_network:
        :return:
        """
        if self.network is not None:
            raise ParamError("The node is already assigned to a network.")
        self.network = Network


## -------------------------------------------------------------------------------------------------
    def get_geometry(self):
        """

        :return:
        """

        return self._geometry



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NetworkConfig(ScientificObject):

    C_IP_BUILD_DTYPE = "Input Build Type"
    C_DTYPE_COORDS = "Co-ordinates"
    C_DTYPE_MATRIX = "Matrix"


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_build_file,
                 p_build_file_dir,
                 p_build_dtype,
                 p_nodes,
                 p_edges):

        if (p_build_file is not None) and (p_build_file_dir is not None) and (p_build_dtype is not None):

            nodes, edges = self.setup_network_metadata(p_build_file, p_build_file_dir, p_build_dtype)

        elif (p_nodes is not None) and (p_edges is not None):

            nodes, edges = p_nodes, p_edges

        # TODO Setup also a logic to store the node and edge data-structure, so that you can find a particular edge given
        #  its nodes, and vice-versa

        else:
            raise Error


## -------------------------------------------------------------------------------------------------
    def setup_network_metadata(self, p_build_file, p_build_file_dir, p_build_dtype) -> Iterable[Iterable[Node],Iterable[Edge]]:
        """

        :param p_build_file:
        :param p_build_file_dir:
        :param p_build_dtype:
        :return:
        """
        nodes = [None]
        edges = [None]

        # TODO setup the logic for building the network metadata

        # Logic Here
        # ...
        # ...
        # ...

        # TODO add the metadata as a standard datatype to this config class

        return nodes, edges






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Network(System, Node):

    C_NET_TYP = "Network Type"

    C_NET_TYP_MATRIX = 0
    C_NET_TYP_GEOGRAPHIC = 1


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                  p_id = None,
                  p_name : str =None,
                  p_range_max : int = Async.C_RANGE_NONE,
                  p_autorun = Task.C_AUTORUN_NONE,
                  p_class_shared = None,
                  p_mode = Mode.C_MODE_SIM,
                  p_network_config = None,
                  p_latency : timedelta = None,
                  p_t_step : timedelta = None,
                  p_fct_strans : FctSTrans = None,
                  p_fct_success : FctSuccess = None,
                  p_fct_broken : FctBroken = None,
                  p_mujoco_file = None,
                  p_frame_skip : int = 1,
                  p_state_mapping = None,
                  p_action_mapping = None,
                  p_camera_conf : tuple = (None, None, None),
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs ):

        System.__init__(self,
                  p_id = p_id,
                  p_name =p_name,
                  p_range_max = p_range_max,
                  p_autorun = p_autorun,
                  p_class_shared = p_class_shared,
                  p_mode = p_mode,
                  p_latency = p_latency,
                  p_t_step = p_t_step,
                  p_fct_strans = p_fct_strans,
                  p_fct_success = p_fct_success,
                  p_fct_broken = p_fct_broken,
                  p_mujoco_file = p_mujoco_file,
                  p_frame_skip = p_frame_skip,
                  p_state_mapping = p_state_mapping,
                  p_action_mapping = p_action_mapping,
                  p_camera_conf = p_camera_conf,
                  p_visualize = p_visualize,
                  p_logging = p_logging,
                  **p_kwargs)


        self.vehicles = {}
        self.nodes = {}
        self.node_ids = []
        self.edges = {}
        self.edge_ids = []
        self.routes = {}
        self.network_config = p_network_config


## -------------------------------------------------------------------------------------------------
    def setup_network(self, p_network_config:NetworkConfig):
        """

        :param p_network_config:
        :return:
        """
        # Logic Here
        # ...
        # ...


## -------------------------------------------------------------------------------------------------
    def add_node(self, p_node:Node, p_node_id):
        """

        :param p_node:
        :param p_node_id:
        :return:
        """

        if p_node_id in self.nodes.keys():
            raise ValueError("The node already exists in the Network.")

        self.nodes[p_node_id] = p_node

        if len(self.node_ids) != len(self.nodes):
            raise ValueError("The current nodes and node ids lengths are not matching. Please check the "
                             "network configuration.")


        self.node_ids.append(p_node_id)

## -------------------------------------------------------------------------------------------------
    def get_node(self, p_node_id):
        """

        :param p_node_id:
        :return:
        """

        node = self.nodes[p_node_id]
        return node


## -------------------------------------------------------------------------------------------------
    def add_edge(self, p_node1:Node, p_node2:Node, bidirectional, p_edge):
        """

        :param p_node1:
        :param p_node2:
        :param bidirectional:
        :param p_edge:
        :return:
        """

        if (p_node1, p_node2) in self.edges.keys():
            raise ValueError("The edge already exists in the Network.")

        if bidirectional:

            if (p_node2, p_node1) in self.edges.keys():
                raise ValueError("The edge already exists in the Network.")

            self.edges[(p_node2.get_id(), p_node2.get_id())] = [p_edge, 1]


        self.edges[(p_node1.get_id(), p_node2.get_id())] = [p_edge, 0]


## -------------------------------------------------------------------------------------------------
    def get_edge(self, p_edge_id):
        """

        :param p_edge_id:
        :return:
        """

        idx = self.edge_ids.index(p_edge_id)
        edge = self.edges[idx]

        return edge


## -------------------------------------------------------------------------------------------------
    def get_edge_nodes(self,
                       p_edge_id,
                       p_edge = None):
        """

        :param p_edge_id:
        :param p_edge:
        :return:
        """
        if p_edge is not None:
            p_edge = self.get_edge(p_edge_id)
        node1, node2 = p_edge.get_bounding_nodes()

        return node1, node2

## -------------------------------------------------------------------------------------------------
    def get_connecting_edge(self, p_node1, p_node2):
        """

        :param p_node1:
        :param p_node2:
        :return:
        """

        edge = self.edges[(p_node1, p_node2)]
        return edge


## -------------------------------------------------------------------------------------------------
    def get_edge_geometry(self, p_edge_id):
        """

        :param p_edge_id:
        :return:
        """

        return self.edges[p_edge_id].get_geometry()


## -------------------------------------------------------------------------------------------------



## -------------------------------------------------------------------------------------------------
    def add_vehicles(self, p_vehicles, p_locations):
        """

        :param p_vehicles:
        :param p_locations:
        :return:
        """
        if len(p_vehicles) == len(p_locations):
            for loc, vehicle in enumerate(p_vehicles):
                if vehicle in self.vehicles.keys():
                    raise ValueError("Vehicle is already added in the Network.")
                else:
                    self.vehicles[vehicle] = [p_locations[loc]]
        else:
            raise ValueError("The number of vehicles and the number of locations of the vehicles to be added "
                             "do not match. Please provided equal length lists to add multiple vehicles.")


## -------------------------------------------------------------------------------------------------
    def get_vehicles_info(self):
        """

        :return:
        """
        return self.vehicles


## -------------------------------------------------------------------------------------------------
    def check_node_arrivals(self):
        """

        :return:
        """
        # TODO Need to add a logic to check if any of the vehicle has reached the end of an edge and then
        #  subsequently raise a corresponding event
        # Logic Here
        # ...
        # ...
        # ...
        # Also an idea of cascaded events. An event can cause a chain reaction of the subsequent events.

        pass


## -------------------------------------------------------------------------------------------------
    def update_action_masks(self):
        """

        :return:
        """



## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State = None, p_action: Action = None, p_t_step:timedelta = None) -> State:
        """

        :param p_state:
        :param p_action:
        :param p_t_step:
        :return:
        """

        actions = p_action.get_sorted_values()[0]
        vehicle = actions[0]
        next_node = actions[1]

        # TODO Implement the simulation logic for a system

        # 1. Simulate Traffic


        # 1.1 Check Node Arrivals

        # 2. Simulate Network Properties

        # 3. Simulate SC Nodes

