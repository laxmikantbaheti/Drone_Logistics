from typing import Dict, Any, List, Type
import os  # Import os for path manipulation

# Import the base data generator and its concrete implementations
from ..generators.data_generator import BaseDataGenerator
from ..generators.random_generator import RandomDataGenerator
from ..generators.json_file_data_generator import JsonFileDataGenerator
from ..generators.random_distance_matrix_generator import DistanceMatrixDataGenerator


class DataLoader:
    """
    Acts as a factory for initial simulation data generators.
    It selects and instantiates the appropriate data generator based on configuration,
    and then delegates the data generation to it.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataLoader.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the DataLoader.
                                     Expected keys:
                                     'generator_type': str (e.g., 'json_file', 'random', 'osm', 'custom')
                                     'generator_config': Dict[str, Any] (config specific to the chosen generator)
        """
        self.config = config
        self.generator_type = config.get('generator_type', 'json_file')  # Default to 'json_file'
        self.generator_config = config.get('generator_config', {})

        self.data_generator: BaseDataGenerator = self._instantiate_generator()

        print(f"DataLoader initialized with generator type: '{self.generator_type}'.")

    def _instantiate_generator(self) -> BaseDataGenerator:
        """
        Selects and instantiates the appropriate data generator based on config.

        Raises:
            ValueError: If an unsupported generator type is specified.
        """
        if self.generator_type == 'json_f[[;;ile':
            # Pass the file_path from generator_config
            return JsonFileDataGenerator(self.generator_config)
        elif self.generator_type == 'random':
            return RandomDataGenerator(self.generator_config)
        elif self.generator_type == 'distance_matrix':
            return DistanceMatrixDataGenerator(self.generator_config)
        elif self.generator_type == 'osm':
            # Placeholder for OSM data generator
            # from .generators.osm_data_generator import OsmDataGenerator
            # return OsmDataGenerator(self.generator_config)
            raise NotImplementedError("OSM data generator not yet implemented.")
        elif self.generator_type == 'custom_paper':
            # Placeholder for custom paper data generator
            # from .generators.custom_paper_data_generator import CustomPaperDataGenerator
            # return CustomPaperDataGenerator(self.generator_config)
            raise NotImplementedError("Custom paper data generator not yet implemented.")
        else:
            raise ValueError(f"Unsupported data generator type: {self.generator_type}")

    def load_initial_simulation_data(self) -> Dict[str, Any]:
        """
        Loads all initial simulation data by delegating to the selected data generator.

        Returns:
            Dict[str, Any]: A dictionary containing all initial entity data,
                            structured as expected by GlobalState's _populate_initial_state.
        """
        print(f"DataLoader: Loading initial simulation data using '{self.generator_type}' generator.")
        try:
            raw_data = self.data_generator.generate_data()
            return raw_data
        except Exception as e:
            print(f"DataLoader Error: Failed to generate data using {self.generator_type} generator: {e}")
            # As a fallback, you might return a very minimal hardcoded dummy data if critical.
            # For now, re-raise as it indicates a configuration or generator issue.
            raise

    # --- Plotting Methods ---
    def initialize_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Initializes plot data related to data loading, if any.
        (e.g., indicating which generator type was used).
        """
        print("DataLoader: Initializing plot data.")
        if 'data_loading_info' not in figure_data:
            figure_data['data_loading_info'] = {}
        figure_data['data_loading_info']['generator_type_used'] = self.generator_type
        # You might pass relevant config details from self.generator_config here
        # e.g., if generator_type is 'json_file', add 'file_path'
        # if self.generator_type == 'json_file':
        #     figure_data['data_loading_info']['file_path'] = self.generator_config.get('file_path')

    def update_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Updates plot data related to data loading. (Less dynamic for a data loader).
        """
        print("DataLoader: Updating plot data (no dynamic changes expected).")
        # No dynamic updates typically for a data loader, as it's mostly static once loaded.
