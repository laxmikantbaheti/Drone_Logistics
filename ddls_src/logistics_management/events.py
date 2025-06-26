## -------------------------------------------------------------------------------------------------
## -- Project : --MLPro - A Synoptic Framework for Standardized Machine Learning Tasks--
## -- Package : mlpro-logistics.logistics_management
## -- Module  : events.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-05-23  0.0.0     LSB       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-06-08)

This module provides the base implementations for the logistical and delivery related events
that can happen in the DDLS.

"""


from mlpro.bf.systems import *
from typing import Tuple, List, Iterable

from scipy.cluster.vq import py_vq

from ddls_src.logistics_management.basics import Order





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OrderDelivered(Event, Log):

    """

    """


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_order:Order,
                 p_raising_object,
                 p_delivery_time):
        """

        :param p_order:
        :param p_raising_object:
        :param p_delivery_time:
        """


        Event.__init__(self, p_raising_object = p_raising_object)
        Log.__init__(self)

        self.order = p_order
        self.delivery_time = p_delivery_time




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VehicleAtTransferNode(Event, Log):
    """

    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_vehicle, p_transfer_node, p_raising_object):
        """

        :param p_vehicle:
        :param p_transfer_node:
        :param p_raising_object:
        """
        Event.__init__(self, p_raising_object = p_raising_object)
        Log.__init__(self)

        self.vehicle = p_vehicle
        self.transfer_node = p_transfer_node





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VehicleAtDeliveryNode(Event, Log):
    """

    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_vehicle, p_delivery_node, p_order, p_raising_object):
        """

        :param p_vehicle:
        :param p_delivery_node:
        :param p_order:
        """

        Event.__init__(self, p_raising_object = p_raising_object)
        Log.__init__(self)

        self.vehicle = p_vehicle
        self.delivery_node = p_delivery_node
        self.order = p_order





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OrderTransfer(Event, Log):
    """

    """


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_order,
                 p_drop_element,
                 p_pickup_element,
                 p_raising_object,
                 p_transfer_time):
        """

        :param p_order:
        :param p_drop_element:
        :param p_pickup_element:
        :param p_raising_object:
        :param p_transfer_time:
        """


        Event.__init__(self, p_raising_object = p_raising_object)
        Log.__init__(self)

        self.order = p_order
        self.drop_element = p_drop_element
        self.pickup_element = p_pickup_element
        self.transfer_time = p_transfer_time





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OrderPlaced(Event, Log):
    """

    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_order, p_customer, p_raising_object, p_order_time = None):
        """

        :param p_order:
        :param p_customer:
        :param p_raising_object:
        """


        Event.__init__(self, p_raising_object=p_raising_object)
        Log.__init__(self)

        self.order = p_order
        self.customer = p_customer
        self.order_time = p_order_time





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ChargingStart(Event, Log):
    """

    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_vehicle, p_charging_start_time, p_raising_object):
        """

        :param p_vehicle:
        :param p_raising_object:
        """

        Event.__init__(self, p_raising_object = p_raising_object)
        Log.__init__(self)

        self.vehicle = p_vehicle
        self.charging_start_time = p_charging_start_time





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ChargingEnd(Event, Log):
    """

    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_vehicle, p_charging_finish_time, p_final_charge_state, p_raising_object):
        """

        :param p_vehicle:
        :param p_charging_finish_time:
        :param p_raising_object:
        """


        Event.__init__(self, p_raising_object = p_raising_object)
        Log.__init__(self)

        self.vehicle = p_vehicle
        self.charging_finish_time = p_charging_finish_time
        self.final_charge_state = p_final_charge_state





