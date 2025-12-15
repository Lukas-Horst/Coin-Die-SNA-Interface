__author__ = 'Lukas Horst'

import os
from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime


class PipelineWrapper(ABC):
    """
    Abstract base class for all pipeline wrappers.
    Ensures a consistent interface and handles output directory structure.
    """

    def __init__(self, pipeline_name: str, install_path: str, parameters: Dict[str, Any],
                 raw_source_path: str, work_dir: str, base_output_dir: str, target_side: str):
        """
        Initializes the wrapper and creates a timestamped run directory.

        Args:
            pipeline_name (str): Name of the pipeline (e.g., 'auto_die_studies'). Used for folder naming.
            install_path (str): Absolute path to the installation directory of the external pipeline.
            parameters (Dict): Dictionary containing pipeline-specific parameters.
            raw_source_path (str): The specific source path of the images.
            work_dir (str): Path to the temporary directory for flattened images.
            base_output_dir (str): The root output directory from config (e.g., './results').
            target_side (str): The targeted coin side (obverse/reverse).
        """
        self.pipeline_name = pipeline_name
        self.install_path = os.path.abspath(install_path)
        self.parameters = parameters
        self.raw_source_path = os.path.abspath(raw_source_path)
        self.work_dir = os.path.abspath(work_dir)
        self.target_side = target_side

        # --- NEW: Directory Structure Logic ---
        # 1. Create a timestamp: YYYY-MM-DD_HH-MM-SS
        self.timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        # 2. Construct the full path: results/pipeline_name/timestamp/
        self.run_output_dir = os.path.join(os.path.abspath(base_output_dir), pipeline_name,
                                           self.timestamp)

        # 3. Create the specific run directory
        if not os.path.exists(self.run_output_dir):
            os.makedirs(self.run_output_dir)
            print(f"Created run output directory: {self.run_output_dir}")

        # Validation
        if not os.path.exists(self.install_path):
            raise FileNotFoundError(f"Pipeline install path not found: {self.install_path}")

    @abstractmethod
    def run(self) -> str:
        """
        Executes the pipeline logic.
        Returns: The absolute path to the final standardized JSON file.
        """
        pass