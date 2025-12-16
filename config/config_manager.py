__author__ = 'Lukas Horst'

import json
import os
import re
from typing import Dict, Any

from data_utils import get_files_from_directory_recursive, find_file


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
        self.target_side = "obverse"

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
        self.target_side = settings.get("target_side", "obverse").lower()

        if self.target_side == "obverse":
            path = paths.get("images_obverse")
        elif self.target_side == "reverse":
            path = paths.get("images_reverse")
        else:
            raise ValueError(
                f"Invalid value for 'target_side': '{self.target_side}'. Must be 'obverse' or 'reverse'.")

        # Validate path existence
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"The path for '{self.target_side}' images does not exist: {path}")

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

    def set_config_value(self, key_path: str, value: Any) -> None:
        """
        Sets a configuration value using a dot-separated key path (e.g., 'paths.temp_dir').

        This method updates the configuration dictionary loaded in memory (self.config).

        Args:
            key_path (str): Dot-separated string indicating the path to the key
                            (e.g., 'analysis_settings.target_side').
            value (Any): The new value to set.
        """
        keys = key_path.split('.')
        current_level = self.config

        # Traverse the dictionary structure down to the parent of the final key
        for key in keys[:-1]:
            # If a sub-dictionary does not exist, create it
            if key not in current_level or not isinstance(current_level[key], dict):
                # We need to create a new dictionary to insert the value
                current_level[key] = {}
            current_level = current_level[key]

        # Set the final key's value
        final_key = keys[-1]
        current_level[final_key] = value

    def save_config(self, save_path: str = None) -> None:
        """
        Saves the current configuration dictionary back to a JSON file.

        Args:
            save_path (str, optional): The file path to save the configuration to.
                                        If None, it uses the original file path (self.config_path).
        """
        path_to_save = save_path if save_path is not None else self.config_path

        # Check if the directory exists before saving
        directory = os.path.dirname(path_to_save)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(path_to_save, 'w', encoding='utf-8') as f:
            # Use indent=4 for a human-readable format
            json.dump(self.config, f, indent=4)

    def get_coin_die_analysis_files(self):
        """
        Retrieves a list of all JSON files generated by the pipelines.

        The search is performed recursively within the general output directory
        as defined in the configuration (paths.output_dir).

        Returns:
            list[str]: A list of full file paths (strings) pointing to the
                       found JSON analysis files.
                """
        output_dir = self.get_output_dir()
        analysis_files = get_files_from_directory_recursive(output_dir, '.json')
        return analysis_files

    def get_current_analysis_file(self, side='r'):
        """
        Finds the full absolute path to the currently configured analysis result file
        (JSON format) based on the specified coin side.

        The search is performed recursively within the general output directory
        using the file name defined in the configuration (e.g., dataset-reverse).

        Args:
            side (str, optional): The target side of the analysis file.
                                  Use 'r' for reverse or
                                  'a' for obverse.
                                  Defaults to 'r'.

        Returns:
            str or None: The full absolute path to the found analysis file,
                         or None if the file is not found.
        """
        if side == 'r':
            file = find_file(self.get_output_dir(), self.config["paths"]["dataset-reverse"])
        elif side == 'a':
            file = find_file(self.get_output_dir(), self.config["paths"]["dataset-obverse"])
        return file

    def get_combined_analysis_file_name(self) -> str:
        """
        Creates a combined file name for analysis results based on the names
        of the obverse and reverse analysis files defined in the config.

        The function extracts the unique timestamp from each file name and
        combines them to ensure a unique result file name.

        Returns:
            str: A unique file name for the combined analysis (e.g.,
                 'combined_analysis_TS_REV_TS_OBV').
        """
        reverse_dataset_name = self.config["paths"]["dataset-reverse"]
        obverse_dataset_name = self.config["paths"]["dataset-obverse"]

        # Define a regex pattern to find the timestamp (e.g., 15-12-2025_16-06-32)
        # The pattern captures the timestamp part.
        timestamp_pattern = r'(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})'

        # 1. Extract the Reverse timestamp
        match_rev = re.search(timestamp_pattern, reverse_dataset_name)
        if match_rev:
            timestamp_rev = match_rev.group(1)
        else:
            # Fallback if no timestamp is found
            timestamp_rev = "no_ts_rev"
            print(f"WARNING: No timestamp found in reverse dataset name: {reverse_dataset_name}")

        # 2. Extract the Obverse timestamp
        match_obv = re.search(timestamp_pattern, obverse_dataset_name)
        if match_obv:
            timestamp_obv = match_obv.group(1)
        else:
            # Fallback if no timestamp is found
            timestamp_obv = "no_ts_obv"
            print(f"WARNING: No timestamp found in obverse dataset name: {obverse_dataset_name}")

        # 3. Construct the new, combined file name
        combined_filename = f"combined_analysis_R_{timestamp_rev}_A_{timestamp_obv}"
        return combined_filename

    def get_images_by_id_and_side(self, coin_id1, coin_id2, side):
        """
        Finds the full paths to the image files for two specified coin IDs (coin_id1 and coin_id2)
        by searching for a filename pattern starting with the ID.

        The function uses a regular expression to match files that begin with the coin ID,
        followed by common separators ('_', '-', or '.'), and any file extension/suffix.

        Args:
            coin_id1 (str or int): The first coin's unique ID.
            coin_id2 (str or int): The second coin's unique ID.
            side (str): The target side of the image: 'a' for obverse or
                        'r' for reverse.

        Returns:
            tuple[Optional[str], Optional[str]]: A tuple containing the full path
                                                 to coin1's image and coin2's image.
                                                 Returns None for an image if it is not found.

        Raises:
            ValueError: If an invalid side ('a' or 'r') is provided.
        """
        paths = self.config.get("paths", {})

        # Getting the image path for the given side
        if side == "a":
            image_path = paths.get("images_obverse")
        elif side == "r":
            image_path = paths.get("images_reverse")
        else:
            # Handle invalid side input
            raise ValueError("Invalid side specified. Must be 'a' (obverse) or 'r' (reverse).")

        # Define common image file extensions
        image_extensions = r'(?:\.jpe?g|\.png|\.tif|\.tiff|\.gif)'

        # 1. Construct the Regex Pattern for coin_id1

        # We use re.escape() to ensure that if coin_id1 contains regex special characters,
        # they are treated as literals.
        pattern1 = rf'^{re.escape(str(coin_id1))}[_\-.].*{image_extensions}$'

        # Use find_file with the file_pattern argument
        coin1_file = find_file(image_path, file_pattern=pattern1)

        # 2. Construct the Regex Pattern for coin_id2
        pattern2 = rf'^{re.escape(str(coin_id2))}[_\-.].*{image_extensions}$'

        # Use find_file with the file_pattern argument
        coin2_file = find_file(image_path, file_pattern=pattern2)

        # Return both paths
        return coin1_file, coin2_file


# Example usage for testing
if __name__ == "__main__":
    try:
        cfg = ConfigManager()
        print(f"Active Pipeline: {cfg.get_active_pipeline_name()}")
        print(f"Source Path: {cfg.get_source_image_path()}")
        print(f"Output Dir: {cfg.get_output_dir()}")
    except Exception as e:
        print(f"Config Error: {e}")
