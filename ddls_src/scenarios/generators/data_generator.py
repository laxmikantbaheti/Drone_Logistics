from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseDataGenerator(ABC):
    """
    Abstract base class for all initial simulation data generators.
    All concrete data generators must inherit from this class and implement
    the 'generate_data' method.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the base data generator with configuration specific to the generator.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the generator.
        """
        self.config = config
        print(f"BaseDataGenerator initialized with config: {config}")

    @abstractmethod
    def generate_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Abstract method to generate initial simulation data.
        Concrete implementations must provide the logic to create data for
        nodes, edges, trucks, drones, micro_hubs, and orders.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary where keys are entity types
                                             (e.g., 'nodes', 'edges') and values are
                                             lists of dictionaries, each representing
                                             raw data for an entity.
                                             Example:
                                             {
                                                 'nodes': [{'id': 0, 'coords': [0,0], ...}],
                                                 'edges': [...],
                                                 'trucks': [...],
                                                 'orders': [...],
                                                 'initial_time': 0.0 # Optional, but recommended
                                             }
        """
        pass

