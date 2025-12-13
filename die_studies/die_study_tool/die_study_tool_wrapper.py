__author__ = 'Lukas Horst'

import os
import sys
from unittest.mock import MagicMock

import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# Import local modules
import data_utils
from die_studies.pipeline_wrapper import PipelineWrapper


class DieStudyToolWrapper(PipelineWrapper):
    """
    Wrapper for the 'DieStudyTool' (Fiedler).

    Implements the pipeline logic directly (Preprocessing -> Matching -> Clustering).
    Includes logic to handle small datasets safely (adjusts n_clusters automatically)
    and mocks the 'Orange' dependency if not present.
    """

    def run(self) -> str:
        print(f"--- Pipeline Start: {self.pipeline_name} ---")

        # ---------------------------------------------------------
        # STEP 0: Imports & Setup
        # ---------------------------------------------------------
        self._setup_imports()

        try:
            import utils
            from kornia_matcher import extract_kornia_matches_in_directory
        except ImportError as e:
            raise ImportError(f"Could not import DieStudyTool modules. Error: {e}")

        # ---------------------------------------------------------
        # STEP 1: Flattening
        # ---------------------------------------------------------
        print(f"1. Flattening images from '{self.raw_source_path}' to '{self.work_dir}'...")
        data_utils.flatten_image_directory(self.raw_source_path, self.work_dir)

        # ---------------------------------------------------------
        # STEP 2: Preprocessing
        # ---------------------------------------------------------
        print("2. Running Preprocessing Pipeline...")
        final_images_path = self.preprocess_images(utils, input_dir=self.work_dir)

        # ---------------------------------------------------------
        # STEP 3: Matching
        # ---------------------------------------------------------
        print("3. Computing Matches & Distances...")

        n_clusters = self.parameters.get("number_of_clusters", 50)
        match_method = self.parameters.get("matching_computation_method", 4)

        matching_filename = f"matching_clusters{n_clusters}_method{match_method}.csv"
        matching_csv_path = os.path.join(self.run_output_dir, matching_filename)

        self.compute_matches(matcher_func=extract_kornia_matches_in_directory, utils_module=utils,
            images_path=final_images_path, output_csv_path=matching_csv_path)

        # ---------------------------------------------------------
        # STEP 4: Clustering
        # ---------------------------------------------------------
        print("4. Running Clustering...")

        clustering_filename = f"clustering_clusters{n_clusters}_method{match_method}.csv"
        clustering_csv_path = os.path.join(self.run_output_dir, clustering_filename)

        self.compute_clustering(utils_module=utils, matching_csv_path=matching_csv_path,
            output_csv_path=clustering_csv_path, images_folder_path=self.work_dir)

        # ---------------------------------------------------------
        # STEP 5: Standardization
        # ---------------------------------------------------------
        print("5. Standardizing Results...")

        json_filename = f"sna_data_clusters{n_clusters}_method{match_method}.json"
        final_json_path = os.path.join(self.run_output_dir, json_filename)

        data_utils.convert_csv_to_sna_json(csv_path=clustering_csv_path,
            cluster_col="final_obverse_CL",  # Standard column name in DieStudyTool
            image_col="object_number",  # Standard column name for filenames
            output_json_path=final_json_path)

        print(f"--- Pipeline Finished. JSON saved to: {final_json_path} ---")
        return final_json_path

    def _setup_imports(self):
        if self.install_path not in sys.path:
            sys.path.append(self.install_path)

    def preprocess_images(self, utils_module, input_dir: str) -> str:
        """
        Executes the 5-stage preprocessing chain using functions from utils.py.
        """
        # Define output paths within our run directory
        exp1 = os.path.join(self.run_output_dir, "01_grayscale")
        exp2 = os.path.join(self.run_output_dir, "02_histogram_equalization")
        exp3 = os.path.join(self.run_output_dir, "03_denoise")
        exp4 = os.path.join(self.run_output_dir, "04_histogram_equalization_2")
        exp5 = os.path.join(self.run_output_dir, "05_circle_crop")

        folders = [exp1, exp2, exp3, exp4, exp5]
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # --- Pipeline Execution ---
        print(f"  - Grayscale -> {exp1}")
        if hasattr(utils_module, 'grayscale_directory'):
            utils_module.grayscale_directory(input_dir, exp1)

        print(f"  - CLAHE (1) -> {exp2}")
        if hasattr(utils_module, 'clahe_directory'):
            utils_module.clahe_directory(exp1, exp2)
        elif hasattr(utils_module, 'histogram_equalization_directory'):
            utils_module.histogram_equalization_directory(exp1, exp2)

        print(f"  - Denoise -> {exp3}")
        if hasattr(utils_module, 'apply_denoise_tv_chambolle_directory'):
            utils_module.apply_denoise_tv_chambolle_directory(exp2, exp3, weight=0.5)
        else:
            print("WARNING: Denoise function not found. Fallback: Copying images.")
            import shutil
            for f in os.listdir(exp2):
                src = os.path.join(exp2, f)
                if os.path.isfile(src):
                    shutil.copy(src, os.path.join(exp3, f))

        print(f"  - CLAHE (2) -> {exp4}")
        if hasattr(utils_module, 'clahe_directory'):
            utils_module.clahe_directory(exp3, exp4)

        print(f"  - Circle Crop -> {exp5}")
        if hasattr(utils_module, 'circle_crop_directory'):
            utils_module.circle_crop_directory(exp4, exp5)

        return exp5

    def compute_matches(self, matcher_func, utils_module, images_path: str, output_csv_path: str):
        method_id = self.parameters.get("matching_computation_method", 4)
        print(f"  - Calculating matches using method ID {method_id}...")

        # Call Kornia Matcher
        distances_df = matcher_func(images_path, method=method_id, print_log=True)
        distances_df.to_csv(output_csv_path)

        # Add paths to DataFrame (mapping back to filenames)
        df2 = pd.read_csv(output_csv_path)
        paths = utils_module.get_paths(images_path)
        df2 = utils_module.add_path_to_df(df2, paths)
        df2.to_csv(output_csv_path)
        print(f"  - Matches saved to {output_csv_path}")

    def compute_clustering(self, utils_module, matching_csv_path: str, output_csv_path: str,
                           images_folder_path: str):
        # 1. Load requested parameters
        requested_clusters = self.parameters.get("number_of_clusters", 50)
        dist_func_id = self.parameters.get("distance_computation_method", 2)
        linkage_method = self.parameters.get("method", "complete")

        # 2. Determine number of samples from CSV to prevent Scikit-Learn errors
        try:
            # index_col=0 is usually the name/index. len(df) gives the row count.
            temp_df = pd.read_csv(matching_csv_path, index_col=0)
            n_samples = len(temp_df)
        except Exception as e:
            print(f"WARNING: Could not determine sample count from CSV: {e}")
            n_samples = requested_clusters

        # 3. Logic: Cannot have more clusters than samples
        if requested_clusters > n_samples:
            print(
                f"WARNING: Configuration requests {requested_clusters} clusters, but only {n_samples} samples found.")
            print(f"         Adjusting n_clusters to {n_samples} to prevent crash.")
            n_clusters = n_samples
        else:
            n_clusters = requested_clusters

        if n_clusters < 1:
            # Handle edge case if directory was empty or only 1 image
            n_clusters = 1
            print("WARNING: n_clusters was < 1. Set to 1.")

        print(f"  - Clustering with {n_clusters} clusters (Linkage: {linkage_method})...")

        # 4. Initialize Clusterer with corrected cluster count
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method,
            metric='precomputed')

        # 5. Execute Clustering
        clustering_df = utils_module.compute_clustering(matching_csv_path, clusterer=clusterer,
            distance_function=dist_func_id)

        # Add original filenames for readability
        paths = utils_module.get_paths(images_folder_path)
        clustering_df = utils_module.add_path_to_df(clustering_df, paths,
            name_column='object_number', set_index=False)

        clustering_df.to_csv(output_csv_path)
        print(f"  - Clustering results saved to {output_csv_path}")
