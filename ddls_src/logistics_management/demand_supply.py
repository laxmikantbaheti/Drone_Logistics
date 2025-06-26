## -------------------------------------------------------------------------------------------------
## -- Project : --MLPro - A Synoptic Framework for Standardized Machine Learning Tasks--
## -- Package : mlpro-logistics.logistics_management
## -- Module  : demand_supply.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-05-24  0.0.0     LSB       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-06-08)

This module provides the base implementations for the logistical elements like customer nodes and
supplier nodes for the logistics simulator.
"""


from mlpro.bf.systems import *
from typing import Tuple, List, Iterable
from ddls_src.geographic.basics import Node
from ddls_src.logistics_management.basics import Order
from ddls_src.logistics_management.events import OrderPlaced




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Supplier(Task, Node):
    """

    """


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
    def __init__(self,
                  p_id = None,
                  p_name : str = None,
                  p_range_max : int = Async.C_RANGE_THREAD,
                  p_autorun = Task.C_AUTORUN_NONE,
                  p_class_shared = None,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs):

        """

        :param p_id:
        :param p_name:
        :param p_range_max:
        :param p_autorun:
        :param p_class_shared:
        :param p_visualize:
        :param p_logging:
        :param p_kwargs:
        """

        Task.__init__(self,
                      p_id = p_id,
                      p_name = p_name,
                      p_range_max = p_range_max,
                      p_autorun = p_autorun,
                      p_class_shared = p_class_shared,
                      p_visualize = p_visualize,
                      p_logging = p_logging,
                      **p_kwargs)

        Node.__init__(self)
        self.orders = {}
        self.processed_orders = {}
        self.in_process_orders = {}
        self.pending_orders = {}


## -------------------------------------------------------------------------------------------------
    def process_order(self, p_order):
        """

        :param p_order:
        :return:
        """
        # TODO event handler of dynamic orders raised by an event through a Cliet
        # Logic Here
        # ...
        # ...


## -------------------------------------------------------------------------------------------------
    def process_static_orders(self, p_orders: Iterable[Order]):
        """

        :param p_orders:
        :return:
        """

        for order in p_orders:
            if order.get_id() not in self.orders.keys():
                self.orders[order.get_id()] = order

            else:
                raise ValueError("Order with similar order ID already processed by the client.")





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Customer(Task, Node):


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                  p_id = None,
                  p_name : str = None,
                  p_location:Tuple[int] = None,
                  p_range_max : int = Async.C_RANGE_THREAD,
                  p_autorun = Task.C_AUTORUN_NONE,
                  p_class_shared = None,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs):

        """

        :param p_id:
        :param p_name:
        :param p_range_max:
        :param p_autorun:
        :param p_class_shared:
        :param p_visualize:
        :param p_logging:
        :param p_kwargs:
        """

        Task.__init__(self,
                      p_id = p_id,
                      p_name = p_name,
                      p_range_max = p_range_max,
                      p_autorun = p_autorun,
                      p_class_shared = p_class_shared,
                      p_visualize = p_visualize,
                      p_logging = p_logging,
                      **p_kwargs)

        Node.__init__(self, p_location = p_location)
        self.static_orders = {}
        self.requested_orders = {}
        self.delivered_orders = {}


## -------------------------------------------------------------------------------------------------
    def request_orders(self,
                       p_orders: List[Order],
                       p_suppliers: List[Supplier]):
        """

        :param p_orders:
        :param p_suppliers:
        :return:
        """

        if not len(p_orders) == len(p_suppliers):
            raise ValueError("Please provide equal lengths of orders and suppliers. Each order shall be given "
                             "a value of the Supplier from which the order is to be placed.")

        for idx, order in enumerate(p_orders):
            event = OrderPlaced(p_raising_object=self, p_order = order, p_customer=self)
            if order.get_id() in self.requested_orders.keys():
                del event
                raise ValueError(f"The order {order.get_id()} is already in the list of requested orders by the customer")
            else:
                self._raise_event(p_event_id=OrderPlaced.C_NAME, p_event_object=event)
                self.requested_orders[order.get_id()] = [p_suppliers[idx], order]


## -------------------------------------------------------------------------------------------------
    def request_static_orders(self,
                              p_orders:List[Order],
                              p_suppliers:List[Supplier]):

        """

        :param p_orders:
        :param p_suppliers:
        :return:
        """

        if not len(p_orders) == len(p_suppliers):
            raise ValueError("Please provide equal lengths of orders and suppliers. Each order shall be given "
                             "a value of the Supplier from which the order is to be placed.")

        for idx, order in enumerate(p_orders):
            event = OrderPlaced(p_raising_object=self, p_order=order, p_customer=self)
            if order.get_id() in self.static_orders.keys():
                del event
                raise ValueError(f"The order {order.get_id()} is already in the list of static orders by the customer")
            else:
                self._raise_event(p_event_id=OrderPlaced.C_NAME, p_event_object=event)
                self.static_orders[order.get_id()] = [p_suppliers[idx], order]
