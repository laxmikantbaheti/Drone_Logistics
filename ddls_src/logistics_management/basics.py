## -------------------------------------------------------------------------------------------------
## -- Project : --MLPro - A Synoptic Framework for Standardized Machine Learning Tasks--
## -- Package : mlpro-logistics.logistics_management
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-05-23  0.0.0     LSB       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-06-08)

This module provides the base implementations for the logistical and delivery related objects for the DDLS.
"""


from mlpro.bf.systems import *
from typing import Tuple, List, Iterable





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Order(Task):

    C_LSTAGE = "Life-cycle Stage"

    C_LSTAGE_PLACED = 'Order Placement'
    C_LSTAGE_LOCATION_HISTORY = "Location History"
    C_LSTAGE_DELIVERED = "Order Delivery"

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 product_id,
                 name,
                 p_weight,
                 p_volume,
                 p_value):

        Task.__init__(self,
                      p_id= product_id,
                      p_name=name)

        self.weight = p_weight
        self.volume = p_volume
        self.value = p_value
        self.lifecycle_record = {self.C_LSTAGE_PLACED : None,
                                 self.C_LSTAGE_LOCATION_HISTORY : [],
                                 self.C_LSTAGE_DELIVERED : None}


## -------------------------------------------------------------------------------------------------
    def update_lifecycle_data(self, p_location = None, p_delivery=None):

        if p_location is not None:
            self.lifecycle_record[self.C_LSTAGE_LOCATION_HISTORY].append(p_location)

        if p_delivery is not None:
            self.lifecycle_record[self.C_LSTAGE_DELIVERED] = p_delivery
            # TODO Raise an event here






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DeliveryManagement(Workflow):
    # This task shall serve as the dynamic event manager in the whole system

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_delivery_id,
                 p_delivery_name,
                 **p_kwargs):

        Workflow.__init__(self,
                      p_id=p_delivery_id,
                      p_name = p_delivery_name)


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
    def _update_delivery(self, p_orders, p_info):
        """

        :param p_orders:
        :param p_info:
        :return:
        """
        # updates all the orders in the delivery
        # event handler for all the delivery related events

        # Logic Here
        # ...
        # ...





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Shipment(Task):


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_shipment_id = None,
                 p_shipment_name = '',
                 p_orders:[Order] = None,
                 p_delivery_manager:DeliveryManagement = None,
                 p_info = None):

        Task.__init__(self,
                      p_id = p_shipment_id,
                      p_name = p_shipment_name)

        self.orders = p_orders
        self.delivery_manager = p_delivery_manager
        self.info = p_info
        self.location_history = []


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def create_shipment(p_orders, p_delivery_manager, p_info):

        shipment = Shipment(p_orders = p_orders,
                            p_delivery_manager=p_delivery_manager,
                            p_info=p_info)

        return shipment


## -------------------------------------------------------------------------------------------------
    def get_shipment_name(self):
        """

        :return:
        """

        return self.get_name()


## -------------------------------------------------------------------------------------------------
    def get_shipment_id(self):
        """

        :return:
        """

        return self.get_id()

## -------------------------------------------------------------------------------------------------
    def update_shipment_nodes(self, p_location):
        """

        :return:
        """

        self.location_history.append(p_location)