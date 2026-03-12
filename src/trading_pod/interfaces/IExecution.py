from abc import ABC, abstractmethod
from typing import List
from core.datatypes import OrderRequest, TradeReceipt

class IExecution(ABC):
    """
    Contract for broker communication. 
    Strictly handles the routing and confirmation of live orders.
    """

    @abstractmethod
    def cancel_all_open_orders(self) -> None:
        """
        Cancels all pending or open orders at the broker to free up buying power.
        """
        pass

    @abstractmethod
    def execute_orders(self, orders: List[OrderRequest]) -> List[TradeReceipt]:
        """
        Translates a list of standardized OrderRequests into API calls.
        """
        pass