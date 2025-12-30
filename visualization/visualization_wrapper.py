__author__ = 'Lukas Horst'

import sys
import os
import cv2
import numpy as np
import torch
import imageio.v2 as io
from io import BytesIO


class VisualizationWrapper:
    """
    Wrapper for visualizing matches between two coin images.

    Features:
    - XFeat: Deep Learning based matching (via Auto-Die-Studies).
    - Classic: Uses 'distance_matcher.py' from DieStudyTool (ORB, SIFT, etc.).
    - Caching: Stores result images to disk to speed up repeated requests.
    """

    def __init__(self, config_manager):
        self.config_manager = config_manager

        # 1. Load Visualization Settings
        self.vis_settings = self.config_manager.config.get("visualization_settings", {})
        self.method_id = self.vis_settings.get("method_id", 1)
        self.use_cache = self.vis_settings.get("caching", False)

        # 2. Prepare Cache Directory
        self.cache_dir = None
        if self.use_cache:
            base_cache = self.config_manager.config.get("paths", {}).get("cache_dir", "Cache")
            self.cache_dir = os.path.join(base_cache, "visualization", f"method{self.method_id}")
            os.makedirs(self.cache_dir, exist_ok=True)

        # 3. Module placeholders
        self.XFeat = None
        self.distance_matcher = None  # Placeholder for the external module

        # Load Pipeline Configs
        try:
            self.ads_config = self.config_manager.get_pipeline_config("auto_die_studies")
        except:
            self.ads_config = {}

        try:
            self.dst_config = self.config_manager.get_pipeline_config("die_study_tool")
        except:
            self.dst_config = {}

    def __import_xfeat(self):
        """Imports XFeat from Auto-Die-Studies path."""
        if self.XFeat is not None: return

        install_path = self.ads_config.get("install_path")
        if not install_path or not os.path.exists(install_path):
            pass  # Fallback to global env
        elif install_path not in sys.path:
            sys.path.append(install_path)

        try:
            from extract_features.xfeat_cache import XFeat
            self.XFeat = XFeat
        except ImportError as e:
            raise ImportError(f"Could not import XFeat from {install_path}: {e}")

    def __import_distance_matcher(self):
        """Imports distance_matcher.py from DieStudyTool path."""
        if self.distance_matcher is not None: return

        install_path = self.dst_config.get("install_path")
        if not install_path or not os.path.exists(install_path):
            raise FileNotFoundError(f"DieStudyTool path not found: {install_path}")

        if install_path not in sys.path:
            sys.path.append(install_path)

        try:
            import distance_matcher
            self.distance_matcher = distance_matcher
        except ImportError as e:
            raise ImportError(f"Could not import distance_matcher from {install_path}: {e}")

    def run(self, image_path1, image_path2, coin_id1, coin_id2, side):
        """
        Main method: Checks cache first, then calculates matches based on method_id.
        """
        # 1. Caching Check
        cache_path = None
        if self.use_cache and self.cache_dir:
            filename = f"{coin_id1}_{coin_id2}_{side}.jpg"
            cache_path = os.path.join(self.cache_dir, filename)
            if os.path.exists(cache_path):
                print(f"Loading visualization from cache: {cache_path}")
                return self._load_image_to_buffer(cache_path)

        # 2. Calculation Dispatcher
        print(f"Calculating matches for {coin_id1} vs {coin_id2} (Method {self.method_id})...")

        try:
            if self.method_id == 1:
                # XFeat (Internal logic calling external class)
                score, canvas = self._visualize_xfeat(image_path1, image_path2)

            else:
                # Classic Methods (Delegating to distance_matcher.py)
                score, canvas = self.__visualize_via_tool(image_path1, image_path2, self.method_id)

        except Exception as e:
            print(f"Error during matching calculation: {e}")
            import traceback
            traceback.print_exc()
            return 0, None

        # 3. Save to Cache
        if self.use_cache and cache_path and canvas is not None:
            try:
                cv2.imwrite(cache_path, canvas)
            except Exception as e:
                print(f"Warning: Could not write to cache: {e}")

        # 4. Return Buffer
        if canvas is None:
            return 0, None

        _, buffer = cv2.imencode(".jpg", canvas)
        bytes_buffer = BytesIO(buffer.tobytes())
        bytes_buffer.seek(0)

        return score, bytes_buffer

    def _load_image_to_buffer(self, path):
        img = cv2.imread(path)
        if img is None: return 0, None
        _, buffer = cv2.imencode(".jpg", img)
        bytes_buffer = BytesIO(buffer.tobytes())
        bytes_buffer.seek(0)
        return -1, bytes_buffer

    # =========================================================================
    # ID 1: XFeat (Deep Learning)
    # =========================================================================
    def _visualize_xfeat(self, img_path1, img_path2):
        self.__import_xfeat()
        params = self.ads_config.get("parameters", {})
        top_k = params.get("matching_params", {}).get("XFeat-TopK", 4096)
        filtering = params.get("filtering", True)

        im1_cv = cv2.imread(img_path1)
        im2_cv = cv2.imread(img_path2)
        if im1_cv is None or im2_cv is None: raise FileNotFoundError("Image load failed")

        images_raw = [io.imread(img_path1), io.imread(img_path2)]
        im_tensor_list = []
        for im in images_raw:
            if len(im.shape) == 3:
                im_tensor_list.append(torch.tensor(im.transpose(2, 0, 1)).float())
            else:
                im_tensor_list.append(torch.tensor(im[None, :, :]).float())

        xfeat_instance = self.XFeat(top_k=top_k)
        xfeat_instance.cache_feats(im_tensor_list)
        matches_list = xfeat_instance.match_xfeat_star_from_cache(0, 1)
        mkpts_0, mkpts_1 = matches_list[0], matches_list[1]

        # Draw
        canvas = self.__warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1_cv, im2_cv)

        score = len(mkpts_0)
        if filtering and len(mkpts_0) >= 4:
            _, mask = cv2.findHomography(mkpts_0, mkpts_1, method=cv2.USAC_MAGSAC,
                                         ransacReprojThreshold=8)
            if mask is not None: score = mask.sum()

        return score, canvas

    def __warp_corners_and_draw_matches(self, ref_points, dst_points, img1, img2):
        """Draws the matches and the transformed bounding box (Homography) for XFeat."""
        if len(ref_points) < 4:
            return self.__draw_simple_matches(ref_points, dst_points, img1, img2)

        H, mask = cv2.findHomography(ref_points, dst_points, method=cv2.USAC_MAGSAC,
                                     ransacReprojThreshold=8)

        if H is None:
            return self.__draw_simple_matches(ref_points, dst_points, img1, img2)

        score = mask.sum()
        mask = mask.flatten()

        h, w = img1.shape[:2]
        corners_img1 = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                                dtype=np.float32).reshape(-1, 1, 2)

        try:
            warped_corners = cv2.perspectiveTransform(corners_img1, H)
        except Exception:
            return self.__draw_simple_matches(ref_points, dst_points, img1, img2)

        img2_with_corners = img2.copy()
        for i in range(len(warped_corners)):
            start_point = tuple(warped_corners[i - 1][0].astype(int))
            end_point = tuple(warped_corners[i][0].astype(int))
            cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)

        keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
        keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
        matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

        img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches,
                                      None, matchColor=(0, 255, 0), flags=2)

        text_pos = (img_matches.shape[1] - 300, 50)
        cv2.putText(img_matches, str(int(score)), text_pos, cv2.FONT_HERSHEY_DUPLEX, 1.5,
                    (237, 114, 50), 2)

        return img_matches

    def __draw_simple_matches(self, ref_points, dst_points, img1, img2):
        kp1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
        kp2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
        matches = [cv2.DMatch(i, i, 0) for i in range(len(ref_points))]
        return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

    # =========================================================================
    # ID 2-6: Classic Methods (via DieStudyTool distance_matcher.py)
    # =========================================================================
    def __visualize_via_tool(self, img_path1, img_path2, method_id):
        """
        Delegates the calculation to the imported 'distance_matcher' module
        and handles the visualization. Robust against varying return values.
        """
        self.__import_distance_matcher()
        dm = self.distance_matcher

        # Load color images for drawing (tool loads grayscale internally)
        img1_color = cv2.imread(img_path1)
        img2_color = cv2.imread(img_path2)
        if img1_color is None or img2_color is None:
            raise FileNotFoundError("Image load failed")

        matches = []
        kp1, kp2 = [], []

        # -- Dispatch to distance_matcher functions --

        if method_id == 2:  # ORB + Hamming2
            res = dm.detect_keypoints_and_match(img_path1, img_path2)
            matches, kp1, kp2 = res[0], res[2], res[4]

        elif method_id == 3:  # SIFT + L2
            res = dm.detect_keypoints_and_match_SIFT(img_path1, img_path2)
            matches, kp1, kp2 = res[0], res[2], res[4]

        elif method_id == 4:  # SIFT + FLANN
            res = dm.flann_matcher(img_path1, img_path2)
            raw_matches = res[0]
            kp1, kp2 = res[2], res[4]

            matches = []

            # Safety check: Handle empty results
            if not raw_matches:
                matches = []

            # Case A: Standard KNN output (List of lists/tuples -> [[m, n], ...])
            # Check if the first element is a list or tuple (iterable pair)
            elif isinstance(raw_matches[0], (list, tuple)):
                for entry in raw_matches:
                    if len(entry) == 2:
                        m, n = entry
                        # Lowe's Ratio Test (0.75)
                        if m.distance < 0.75 * n.distance:
                            matches.append(m)

            # Case B: Flat list of DMatch objects ([m, ...])
            # The matcher likely already filtered them or isn't using KNN.
            else:
                matches = raw_matches

        elif method_id == 5:  # SIFT + KNN
            res = dm.detect_keypoints_and_descriptors_knn_match(img_path1, img_path2)
            raw_matches = res[0]
            kp1 = res[2]
            kp2 = res[4]

            matches = []

            # Safety check: Handle empty results
            if not raw_matches:
                matches = []

            # Case A: Standard KNN output (List of lists/tuples -> [[m, n], ...])
            # We need to apply the Ratio Test manually.
            elif isinstance(raw_matches[0], (list, tuple)):
                for entry in raw_matches:
                    if len(entry) == 2:
                        m, n = entry
                        if m.distance < 0.75 * n.distance:
                            matches.append(m)

            # Case B: Flat list of DMatch objects ([m, ...])
            # The tool likely already filtered them.
            else:
                matches = raw_matches

        elif method_id == 6:  # ORB + Hamming
            res = dm.detect_keypoints_match_hamming(img_path1, img_path2)
            matches, kp1, kp2 = res[0], res[2], res[4]

        else:
            raise ValueError(f"Unknown ID for distance_matcher: {method_id}")

        # -- Visualization --

        # Sort matches by distance (best first)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw top 50 matches using the Color Images we loaded
        img_matches = cv2.drawMatches(img1_color, kp1, img2_color, kp2, matches[:50], None,
            flags=2)

        return len(matches), img_matches
