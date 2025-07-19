import json
import os
from typing import Dict, Any, List
from .data_generator import BaseDataGenerator  # Import the base class


class JsonFileDataGenerator(BaseDataGenerator):
    """
    A concrete data generator that loads initial simulation data from a specified JSON file.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the JsonFileDataGenerator.

        Args:
            config (Dict[str, Any]): Configuration for JSON file loading. Expected keys:
                                     'file_path': str (path to the JSON file containing entity data).
        """
        super().__init__(config)
        self.file_path = config.get('file_path')
        if not self.file_path:
            raise ValueError("JsonFileDataGenerator: 'file_path' must be provided in config.")

        # Ensure path is absolute if relative paths are passed
        if not os.path.isabs(self.file_path):
            # Resolve path relative to the current working directory, or a specified base path
            # For this example, we assume file_path is correctly relative to main.py or absolute.
            pass

        print(f"JsonFileDataGenerator initialized for file: {self.file_path}")

    def generate_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Loads initial simulation data from the configured JSON file.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary containing raw entity data.
        """
        data = {}
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            print(f"JsonFileDataGenerator: Successfully loaded data from {self.file_path}.")
        except FileNotFoundError:
            print(f"JsonFileDataGenerator Error: File not found at {self.file_path}.")
            raise  # Re-raise to indicate a critical failure in data loading
        except json.JSONDecodeError:
            print(f"JsonFileDataGenerator Error: Could not decode JSON from {self.file_path}. Check file format.")
            raise  # Re-raise to indicate a critical failure

        # Basic validation to ensure expected keys are present
        expected_keys = ["nodes", "edges", "trucks", "drones", "micro_hubs", "orders"]
        for key in expected_keys:
            if key not in data or not isinstance(data[key], list):
                print(
                    f"JsonFileDataGenerator Warning: Missing or invalid '{key}' in loaded data. Providing empty list.")
                data[key] = []

        # Ensure initial_time is present
        if 'initial_time' not in data:
            data['initial_time'] = 0.0  # Default if not specified in JSON

        return data

