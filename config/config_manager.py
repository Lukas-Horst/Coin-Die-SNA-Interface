__author__ = 'Lukas Horst'

import json
import os
from typing import Dict, Any


class ConfigManager:
    """
    Handles loading and parsing of the central configuration file (config.json).
    Acts as a single source of truth for paths and pipeline settings.
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Args:
            config_path (str): Path to the JSON configuration file. Defaults to 'config.json'.
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Loads the JSON configuration file from disk.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error parsing config.json: {e}")

    def get_source_image_path(self) -> str:
        """
        Determines the correct source image path based on the 'target_side' setting
        defined in 'analysis_settings'.

        Returns:
            str: The absolute path to the selected image folder (obverse or reverse).
        """
        settings = self.config.get("analysis_settings", {})
        paths = self.config.get("paths", {})

        # Get target side (default to obverse if missing)
        target_side = settings.get("target_side", "obverse").lower()

        if target_side == "obverse":
            path = paths.get("images_obverse")
        elif target_side == "reverse":
            path = paths.get("images_reverse")
        else:
            raise ValueError(
                f"Invalid value for 'target_side': '{target_side}'. Must be 'obverse' or 'reverse'.")

        # Validate path existence
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"The path for '{target_side}' images does not exist: {path}")

        return os.path.abspath(path)

    def get_temp_path(self) -> str:
        """
        Returns the path for the temporary flattened images directory.
        Creates the directory structure if it doesn't exist (but does NOT clear it).
        """
        path = self.config.get("paths", {}).get("temp_dir", "flattened_images")
        path = os.path.abspath(path)

        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_output_dir(self) -> str:
        """
        Returns the general output directory.
        Creates the directory if it doesn't exist.
        """
        path = self.config.get("paths", {}).get("output_dir", "results")
        path = os.path.abspath(path)

        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_active_pipeline_name(self) -> str:
        """
        Returns the key of the currently active pipeline (e.g., 'die_study_tool').
        """
        return self.config.get("analysis_settings", {}).get("active_pipeline")

    def get_pipeline_config(self, pipeline_name: str) -> Dict[str, Any]:
        """
        Retrieves the configuration object for a specific pipeline.

        Args:
            pipeline_name (str): The key of the pipeline (e.g., 'auto_die_studies').

        Returns:
            Dict containing 'install_path' and 'parameters'.
        """
        pipelines = self.config.get("pipelines", {})

        if pipeline_name not in pipelines:
            raise ValueError(
                f"Pipeline '{pipeline_name}' is not defined in config.json pipelines section.")

        return pipelines[pipeline_name]


# Example usage for testing
if __name__ == "__main__":
    try:
        cfg = ConfigManager()
        print(f"Active Pipeline: {cfg.get_active_pipeline_name()}")
        print(f"Source Path: {cfg.get_source_image_path()}")
        print(f"Output Dir: {cfg.get_output_dir()}")
    except Exception as e:
        print(f"Config Error: {e}")