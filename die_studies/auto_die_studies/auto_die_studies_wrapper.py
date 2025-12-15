__author__ = 'Lukas Horst'

import os
import sys
import numpy as np
from typing import Optional

# Import local modules
import data_utils
from die_studies.pipeline_wrapper import PipelineWrapper


class AutoDieStudiesWrapper(PipelineWrapper):
    """
    Wrapper for the 'Auto-Die-Studies' pipeline.
    """

    def run(self) -> str:
        print(f"--- Pipeline Start: {self.pipeline_name} ---")

        # ---------------------------------------------------------
        # STEP 1: Flattening Images
        # ---------------------------------------------------------
        print(f"1. Flattening images from '{self.raw_source_path}' to '{self.work_dir}'...")
        data_utils.flatten_image_directory(self.raw_source_path, self.work_dir)

        # ---------------------------------------------------------
        # STEP 2: Setup Environment
        # ---------------------------------------------------------
        print(f"2. Loading Pipeline Modules from {self.install_path}...")
        if self.install_path not in sys.path:
            sys.path.append(self.install_path)

        try:
            from extract_features import compute_xfeat_sim, compute_roma_sim
            from clustering import AGLP_clustering, proj_hdbscan, dissim_hdbscan
        except ImportError as e:
            raise ImportError(f"Could not import Auto-Die-Studies modules. Error: {e}")

        # ---------------------------------------------------------
        # STEP 3: Matching & Clustering
        # ---------------------------------------------------------
        valid_extensions = ('.jpg', '.png')
        images = sorted([os.path.join(self.work_dir, f) for f in os.listdir(self.work_dir) if
            f.lower().endswith(valid_extensions)])

        if not images:
            raise ValueError(f"No images found in work directory: {self.work_dir}")

        # Save matches.npy inside the specific run directory
        clustering_algo = self.parameters.get("clustering_algorithm", "AGLP")
        matching_algo = self.parameters.get("matching_algorithm", "XFeat")
        sim_matrix_path = os.path.join(self.run_output_dir, f"{self.pipeline_name}_"
                                                            f"{self.target_side}_matching_"
                                                            f"{matching_algo}_{clustering_algo}_"
                                                            f"{self.timestamp}.npy")

        match_params = self.parameters.get("matching_params", {})
        filtering = self.parameters.get("filtering", True)

        print(f"3. Running Matching ({matching_algo})...")

        if matching_algo == "XFeat":
            top_k = int(match_params.get("XFeat-TopK", 10000))
            compute_xfeat_sim.save_matches(images, top_k=top_k, filtering=filtering,
                                           fname=sim_matrix_path)
        elif matching_algo == "RoMa":
            threshold = float(match_params.get("RoMa-Threshold", 0.9))
            compute_roma_sim.save_matches(images, threshold=threshold, fname=sim_matrix_path)
        else:
            raise ValueError(f"Unknown matching algorithm: {matching_algo}")

        print("4. Running Clustering...")
        if not os.path.exists(sim_matrix_path):
            raise FileNotFoundError(f"Similarity matrix missing: {sim_matrix_path}")

        sim_matrix = np.load(sim_matrix_path)

        partition = []
        if clustering_algo == 'AGLP':
            partition = AGLP_clustering(sim_matrix)
        elif clustering_algo == 'HDBSCAN-Proj':
            partition = proj_hdbscan(sim_matrix)
        elif clustering_algo == 'HDBSCAN-Dissim':
            partition = dissim_hdbscan(sim_matrix)
        else:
            raise ValueError(f"Unknown clustering algorithm: {clustering_algo}")

        # ---------------------------------------------------------
        # STEP 4: Standardization (Output)
        # ---------------------------------------------------------
        print("5. Standardizing Results...")

        # A) CSV (inside run directory)
        csv_path = os.path.join(self.run_output_dir, f"{self.pipeline_name}_{self.target_side}"
                                                     f"_clustering_{matching_algo}_"
                                                     f"{clustering_algo}_{self.timestamp}.csv")

        data_utils.create_clustering_csv(cluster_ids=partition, images_dir=self.work_dir,
            output_csv_path=csv_path, cluster_column_name="cluster_id")

        # B) Final JSON with Parameter Names
        # Construct filename: e.g., sna_data_XFeat_AGLP.json
        json_filename = (f"{self.pipeline_name}_{self.target_side}_sna_data_{matching_algo}_"
                         f"{clustering_algo}_{self.timestamp}.json")
        final_json_path = os.path.join(self.run_output_dir, json_filename)

        data_utils.convert_csv_to_sna_json(csv_path=csv_path, cluster_col="cluster_id",
            image_col="object_number", output_json_path=final_json_path)

        print(f"--- Pipeline Finished. Results saved to: {self.run_output_dir} ---")
        return final_json_path
