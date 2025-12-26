__author__ = 'Lukas Horst'

import os
import shutil
import sys
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from joblib import Parallel, delayed

# Import base class & utilities
import data_utils
from die_studies.die_study_tool.die_study_tool_wrapper import DieStudyToolWrapper
from die_studies.pipeline_wrapper import PipelineWrapper


class DieStudyWrapper(PipelineWrapper):
    """
    Wrapper for the 'Die_Study' (Hybrid Pipeline).

    Workflow:
    1. Preprocessing (Hybrid): Grayscale -> ROF Denoise -> CLAHE -> ROF Denoise -> Circle Crop.
    2. Matching: Uses 'matcher.py' from the Die_Study installation.
    3. Clustering: Delegates logic to 'DieStudyTool' but handles robustness issues
       (e.g., switching to Euclidean distance for sparse data, NaN handling).
    """

    def __init__(self, pipeline_name: str, install_path: str, die_study_tool_path: str,
                 parameters: dict, raw_source_path: str, work_dir: str, base_output_dir: str,
                 target_side: str):
        super().__init__(pipeline_name, install_path, parameters, raw_source_path, work_dir,
                         base_output_dir, target_side)

        self.die_study_tool_path = os.path.abspath(die_study_tool_path)

        if not os.path.exists(self.die_study_tool_path):
            raise FileNotFoundError(f"DieStudyTool path not found: {self.die_study_tool_path}")

    def run(self) -> str:
        print(f"--- Pipeline Start: {self.pipeline_name} (Hybrid) ---")

        # ---------------------------------------------------------
        # STEP 0: Imports & Setup
        # ---------------------------------------------------------
        self.__setup_imports()

        try:
            import utils as dst_utils
            import matcher
            import rof
        except ImportError as e:
            debug_path = "\n".join(sys.path)
            raise ImportError(
                f"Could not import required modules. Error: {e}\nSys.Path: {debug_path}")

        # ---------------------------------------------------------
        # STEP 1: Flattening
        # ---------------------------------------------------------
        print(f"1. Flattening images from '{self.raw_source_path}' to '{self.work_dir}'...")
        data_utils.flatten_image_directory(self.raw_source_path, self.work_dir)

        # ---------------------------------------------------------
        # STEP 2: Preprocessing (Hybrid Chain)
        # ---------------------------------------------------------
        print("2. Running Preprocessing Pipeline...")
        final_images_path = self.preprocess_hybrid(dst_utils, rof, input_dir=self.work_dir)

        # ---------------------------------------------------------
        # STEP 3: Matching
        # ---------------------------------------------------------
        print("3. Computing Matches (Die_Study Matcher)...")

        # User output file (contains file paths, human-readable)
        matching_csv_path = os.path.join(self.run_output_dir, f"{self.pipeline_name}_"
                                                              f"{self.target_side}_matching_"
                                                              f"{self.timestamp}.csv")

        self.compute_matches(matcher, dst_utils, final_images_path, matching_csv_path)

        # ---------------------------------------------------------
        # STEP 4: Clustering
        # ---------------------------------------------------------
        print("4. Running Clustering (Delegating to DieStudyTool)...")
        clustering_csv_path = os.path.join(self.run_output_dir,
                                           f"{self.pipeline_name}_{self.target_side}"
                                           f"_clustering_{self.timestamp}.csv")

        # Instantiate DieStudyToolWrapper just for this step
        # This will create a folder 'temp_clustering' inside our current output directory
        tool_wrapper = DieStudyToolWrapper(pipeline_name="temp_clustering",
            install_path=self.die_study_tool_path, parameters=self.parameters,
            raw_source_path=self.raw_source_path, work_dir=self.work_dir,
            base_output_dir=self.run_output_dir, target_side=self.target_side)

        try:
            tool_wrapper.compute_clustering(utils_module=dst_utils,
                matching_csv_path=matching_csv_path, output_csv_path=clustering_csv_path,
                images_folder_path=self.work_dir)
        except Exception as e:
            print(f"CRITICAL: Clustering failed. ({e})")
            if os.path.exists(matching_csv_path):
                df_debug = pd.read_csv(matching_csv_path, index_col=0)
                if df_debug.isnull().values.any():
                    print("DEBUG: Input CSV contained NaNs.")
            raise e
        finally:
            # --- CLEANUP ---
            # Remove the temporary 'temp_clustering' folder created by the delegated wrapper
            temp_dir = os.path.join(self.run_output_dir, "temp_clustering")
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print("  - Cleaned up temporary artifacts (temp_clustering folder removed).")
                except Exception as cleanup_err:
                    print(f"  WARNING: Could not remove temp folder: {cleanup_err}")

        # ---------------------------------------------------------
        # STEP 5: Standardization
        # ---------------------------------------------------------
        print("5. Standardizing Results...")

        n_clusters = self.parameters.get("number_of_clusters", 50)
        dist_func_id = self.parameters.get("distance_computation_method", 13)
        json_filename = (f"{self.pipeline_name}_{self.target_side}_"
                         f"sna_data_clusters{n_clusters}_distFunc{dist_func_id}"
                         f"_{self.timestamp}.json")
        final_json_path = os.path.join(self.run_output_dir, json_filename)

        data_utils.convert_csv_to_sna_json(csv_path=clustering_csv_path,
                                           cluster_col="final_obverse_CL",
                                           image_col="object_number",
                                           output_json_path=final_json_path)

        print(f"--- Pipeline Finished. JSON saved to: {final_json_path} ---")
        return final_json_path

    def __setup_imports(self):
        """Adds DieStudyTool and local image-processing paths to sys.path."""
        if self.die_study_tool_path not in sys.path:
            sys.path.append(self.die_study_tool_path)
        if self.install_path not in sys.path:
            sys.path.append(self.install_path)

        # Locate local 'image-processing/packages' relative to this script
        wrapper_dir = os.path.dirname(os.path.abspath(__file__))
        img_proc_dir = os.path.join(wrapper_dir, "image-processing")
        packages_dir = os.path.join(img_proc_dir, "packages")

        if os.path.exists(packages_dir) and packages_dir not in sys.path:
            sys.path.append(packages_dir)
        elif os.path.exists(img_proc_dir) and img_proc_dir not in sys.path:
            sys.path.append(img_proc_dir)

    def preprocess_hybrid(self, dst_utils, rof_module, input_dir: str) -> str:
        """Executes the specific preprocessing chain: Grayscale -> ROF -> CLAHE -> ROF -> Crop."""
        # Output directories
        exp1 = os.path.join(self.run_output_dir, "01_grayscale")
        exp2 = os.path.join(self.run_output_dir, "02_denoise_rof")
        exp3 = os.path.join(self.run_output_dir, "03_clahe")
        exp4 = os.path.join(self.run_output_dir, "04_denoise_rof_2")
        exp5 = os.path.join(self.run_output_dir, "05_circle_crop")

        for f in [exp1, exp2, exp3, exp4, exp5]:
            if not os.path.exists(f): os.makedirs(f)

        # Execution
        print(f"  - Step 1: Grayscale")
        dst_utils.grayscale_directory(input_dir, exp1)

        print(f"  - Step 2: ROF Denoising")
        self.__apply_denoise_parallel(rof_module, exp1, exp2)

        print(f"  - Step 3: CLAHE")
        if hasattr(dst_utils, 'clahe_directory'):
            dst_utils.clahe_directory(exp2, exp3)
        else:
            dst_utils.histogram_equalization_directory(exp2, exp3)

        print(f"  - Step 4: ROF Denoising (2nd pass)")
        self.__apply_denoise_parallel(rof_module, exp3, exp4)

        print(f"  - Step 5: Circle Crop")
        dst_utils.circle_crop_directory(exp4, exp5)

        return exp5

    def __apply_denoise_parallel(self, rof_module, src_dir, target_dir):
        """Applies ROF denoising in parallel."""
        tasks = []
        for root, _, files in os.walk(src_dir):
            for name in files:
                if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    tasks.append((os.path.join(root, name), os.path.join(target_dir, name)))

        if not tasks: return

        print(f"    Processing {len(tasks)} images (Parallel)...")
        Parallel(n_jobs=4, prefer="threads")(
            delayed(self.__rof_worker)(t[0], t[1], rof_module) for t in tasks)

    @staticmethod
    def __rof_worker(src_path, target_path, rof_module):
        """Static worker for ROF to avoid pickling issues."""
        lbda, tolerance, max_iterations = 0.12, 1e-5, 100

        img = cv2.imread(src_path)
        if img is None: return

        # Preprocess for ROF (Float [0,1], RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float64) / 255.0
        if len(img.shape) == 2: img = np.dstack([img] * 3)

        # Apply Algorithm
        try:
            denoised = rof_module.denoise_image(img, lbda, tolerance, max_iterations,
                                                verbose=False, ev_stop=None)
        except TypeError:
            denoised = rof_module.denoise_image(img, lbda, tolerance, max_iterations)

        # Save (UInt8 [0,255], BGR)
        denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
        denoised = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
        cv2.imwrite(target_path, denoised)

    def compute_matches(self, matcher_module, dst_utils, images_path: str, output_csv_path: str):
        """Calculates matches and creates two CSVs: one for display (with paths) and one for
        calculation."""
        print(f"  - Calculating matches...")
        try:
            df = matcher_module.extract_matches_in_directory(images_path, count=True)
        except TypeError:
            df = matcher_module.extract_matches_in_directory(images_path)

        # 1. NaN Handling: Fill missing values with 0.0 (No Match)
        if df.isnull().values.any():
            df.fillna(0.0, inplace=True)

        # 2. Standardize Index
        df.index.name = 'name'

        # 3. Add Paths (for User Output)
        try:
            paths = dst_utils.get_paths(images_path)
            path_map = {os.path.basename(p): p for p in paths}
            df['path'] = df.index.map(path_map)
        except Exception:
            pass

        df.to_csv(output_csv_path)
        print(f"  - Matches saved to {output_csv_path}")
