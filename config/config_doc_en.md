# Configuration Documentation (`config.json`)

This file controls the entire interface between the raw image data, the analysis pipelines, and 
the Coin-Die-SNA.

## 1. Overview
The configuration is divided into four main sections:
1.  **`paths`**: Where is the data located and which analysis files are active?
2.  **`analysis_settings`**: What should be analyzed in the next run?
3.  **`visualization_settings`**: How should matches be visualized?
4.  **`pipelines`**: Specific settings for the algorithms.

---

## 2. Paths
Defines storage locations for input/output data and selects the active result files.

> **Warning regarding Path Resolution:**  Be cautious when using **relative paths** (e.g., `./results` or `../Data`). Since the configuration file is accessed by scripts running in different directories (e.g., the main project root vs. the `Coin-Die-SNA-Interface` submodule), relative paths may resolve incorrectly depending on where the script is executed.  
> **Using absolute paths is strongly recommended** to ensure consistent file access across the entire application.

| Key | Type | Description                                                                                                            | Example                                    |
| :--- | :--- |:-----------------------------------------------------------------------------------------------------------------------|:-------------------------------------------|
| `images_obverse` | String | Absolute path to the folder containing obverse images.                                                                 | `"Data/Coins/Obverse"`                     |
| `images_reverse` | String | Absolute path to the folder containing reverse images.                                                                 | `"Data/Coins/Reverse"`                     |
| `findspot_data` | String | Path to the CSV file containing geographical findspot data (coordinates/metadata).                                     | `"Data/numisdata.csv"`                     |
| `die_ground_truth` | String | Path to the CSV file containing the Ground Truth (known die identities) for validation metrics.                        | `"Data/ground_truth.csv"`                  |
| `output_dir` | String | Path where results (JSON, CSV) will be saved.                                                                          | `"results"`                                |
| `temp_dir` | String | Temporary folder for flattened images. This directory is cleared and recreated upon every execution.                   | `"temp_flattened"`                         |
| `cache_dir` | String | Folder used for caching all results of the Coin-Die-SNA like the graphs and the visualization.                         | `"Cache"`                                  |
| `dataset-obverse` | String | **Active Result File**: The filename of the specific JSON clustering result used for obverse analysis in the frontend. | `"die_studie_obverse_12_aglp.json"`        |
| `dataset-reverse` | String | **Active Result File**: The filename of the specific JSON clustering result used for reverse analysis in the frontend. | `"die_studie_reverse_14_projhdbscan.json"` |

---

## 3. Analysis Settings
Determines the scope of the current analysis run (creation of new results).

| Key | Type | Allowed Values                                          | Description |
| :--- | :--- |:--------------------------------------------------------| :--- |
| `target_side` | String | `"obverse"`, `"reverse"`                                | Specifies which of the two image folders (`paths`) will be processed. |
| `active_pipeline` | String | `"die_study_tool"`, `"auto_die_studies"`, `"die_study"` | Selects the pipeline to be executed. The name must match a key in the `pipelines` section. |

---

## 4. Visualization Settings
Controls the behavior of the visual matching comparison (`/coinmatching` endpoint).

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `method_id` | Integer | `1` | ID of the visualization algorithm (see table below). |
| `caching` | Boolean | `true` | If `true`, generated matching images are saved to `cache_dir` to speed up future requests. |

### Available Visualization Methods
The parameter `method_id` accepts the following IDs.
> **Important:** Ensure that the `install_path` for the corresponding backend is correctly set in the `pipelines` section of your configuration file!
> * **ID 1** requires the path for `auto_die_studies`.
> * **IDs 2–6** require the path for `die_study_tool`.

| ID | Method | Backend | Description |
| :--- | :--- | :--- | :--- |
| `1` | **XFeat** | Auto-Die-Studies | Deep Learning based matching. Robust to rotation and scale changes. Requires Torch/GPU for speed. |
| `2` | **ORB + Hamming2** | DieStudyTool | Fast binary descriptor. Uses `NORM_HAMMING2` metric. Standard for ORB. |
| `3` | **SIFT + Euclidean** | DieStudyTool | Classic SIFT features with L2 (Euclidean) distance. High precision. |
| `4` | **SIFT + FLANN** | DieStudyTool | SIFT with FLANN-based matcher (Approximate Nearest Neighbor). Faster for large datasets. |
| `5` | **SIFT + KNN** | DieStudyTool | SIFT with K-Nearest Neighbors and Lowe's Ratio Test (0.75). Produces very clean visualizations with few outliers. |
| `6` | **ORB + Hamming** | DieStudyTool | Alternative ORB implementation using standard `NORM_HAMMING`. |

---

## 5. Pipelines (Algorithm Configuration)
This section defines specific parameters for the individual pipeline wrappers.

### 5.1 DieStudyTool (Fiedler)
**Installation:**
```bash
git clone [https://github.com/Urjarm/DieStudyTool.git](https://github.com/Urjarm/DieStudyTool.git)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `install_path` | String | | Path to the tool's installation folder. |
| `number_of_clusters` | Integer | `50` | The desired number of clusters (dies). |
| `matching_computation_method` | Integer | `4` | ID of the matching algorithm (see table below). |
| `distance_computation_method` | Integer | `2` | ID of the distance function (see table below). |

#### Reference: All Available Matching Methods
The parameter `matching_computation_method` accepts the following IDs:

| ID | Method / Algorithm | ID | Method / Algorithm |
| :--- | :--- | :--- | :--- |
| `0` | ORB | `16` | Kornia HyNet Descriptor |
| `1` | Matcher NN (Nearest Neighbor) | `17` | Kornia TFeat Descriptor |
| `2` | Matcher MNN (Mutual Nearest Neighbor) | `18` | Kornia SOSNet Descriptor |
| `3` | Matcher SNN (Standard Nearest Neighbor) | `19` | Detector & Desc. SOLD2 |
| `4` | **Matcher SMNN** (SIFT MNN - Default) | `20` | Detector & Desc. DeDoDe |
| `5` | Matcher FGINN | `21` | Detector & Desc. DISK |
| `6` | Matcher AdaLAM | `22` | Detector & Desc. SIFTFeature |
| `7` | Matcher LightGlue | `23` | Det. & Desc. GFTTAffNetHardNet |
| `8` | Matcher LoFTR | `24` | Det. & Desc. KeyNetAffNetHardNet |
| `9` | Detector GTFF Response | `25` | OpenCV 2nn |
| `10` | Detector DoG Response Single | `26` | Testing (Experimental) |
| `11` | Kornia Dense SIFT Descriptor | `27` | SMNN Abs Count |
| `12` | Kornia SIFT Descriptor | `28` | Kornia NN ORB |
| `13` | Kornia MKD Descriptor | `29` | OpenCV SMNN |
| `14` | Kornia HardNet Descriptor | `30` | SMNN DISK (USAC_ACCURATE) |
| `15` | Kornia HardNet8 Descriptor | | |

#### Reference: All Available Distance Functions
The parameter `distance_computation_method` accepts the following IDs:

| ID | Function | ID | Function |
| :--- | :--- | :--- | :--- |
| `0` | No Distance Function | `10` | SciPy Spatial Chebyshev |
| `1` | Spearman | `11` | SciPy Spatial Cityblock (Manhattan) |
| `2` | **Pearson** (Default) | `12` | SciPy Spatial Correlation |
| `3` | Cosine | `13` | SciPy Spatial Euclidean |
| `4` | SciPy Stats Linregress | `14` | SciPy Spatial Jensen-Shannon |
| `5` | SciPy Stats Pointbiserialr | `15` | SciPy Spatial Minkowski |
| `6` | SciPy Stats Kendalltau | `16` | SciPy Spatial Seuclidean |
| `7` | SciPy Stats Somersd | `17` | SciPy Spatial SqEuclidean |
| `8` | SciPy Spatial Braycurtis | `18` | SciPy Spatial Yule |
| `9` | SciPy Spatial Canberra | | |

### 5.2 Auto-Die-Studies (Cornet et al.)
**Installation:**
```bash
git clone --recurse-submodules [https://github.com/ClementCornet/Auto-Die-Studies.git](https://github.com/ClementCornet/Auto-Die-Studies.git)
```

| Parameter | Type | Allowed Values / Description |
| :--- | :--- | :--- |
| `install_path` | String | Path to the pipeline's folder. |
| `matching_algorithm` | String | `"XFeat"`, `"RoMa"` |
| `filtering` | Boolean | `true` / `false` (Enables filtering of bad matches). |
| `clustering_algorithm`| String | `"AGLP"`, `"HDBSCAN-Dissim"`, `"HDBSCAN-Proj"`, `"ConnectedComponents"` |
| **`matching_params`** | Object | Specific parameters for the matcher: |
| - `XFeat-TopK` | Integer | Number of keypoints (only for XFeat). Recommended: `10000` |
| - `RoMa-Threshold` | Float | Confidence threshold for RoMa (0.0 to 1.0). Recommended: `0.9` |

### 5.3 Die_Study (Hybrid)
A hybrid pipeline that uses its own preprocessing chain (Grayscale -> Denoise -> CLAHE -> Crop) but relies on the clustering algorithms of the `DieStudyTool`.

**Installation:**
```bash
git clone [https://github.com/Frankfurt-BigDataLab/2023_CAA_ClaReNet.git](https://github.com/Frankfurt-BigDataLab/2023_CAA_ClaReNet.git)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `install_path` | String | | Path to the scripts of this pipeline (containing `matcher.py`, `image-processing/`). |
| `die_study_tool_path` | String | | **Important:** Absolute path to the installation folder of the `DieStudyTool` (Fiedler), as its clustering logic is used. |
| `number_of_clusters` | Integer | `50` | The desired number of clusters (dies). |
| `distance_computation_method` | Integer | `13` | ID of the distance function (see table below). |

#### Recommended Distance Functions for Die_Study
Since this pipeline often produces matching results with many zeros (sparse data), correlation-based metrics (like Pearson) are unsuitable. Use one of the following **spatial** distance functions:

| ID | Function | Description |
| :--- | :--- | :--- |
| `13` | **SciPy Spatial Euclidean** (Recommended) | Standard Euclidean distance. Very stable with zeros. |
| `17` | SciPy Spatial SqEuclidean | Squared Euclidean distance. Penalizes larger differences more heavily. |
| `11` | SciPy Spatial Cityblock | "Manhattan" distance. Sum of absolute differences. |
| `10` | SciPy Spatial Chebyshev | Maximum difference along any coordinate dimension. |
| `15` | SciPy Spatial Minkowski | Generalization of Euclidean and Manhattan distances. |
| `8`  | SciPy Spatial Braycurtis | Often used for ecological/count data, stable for positive values. |
| `9`  | SciPy Spatial Canberra | Weighted version of Manhattan, sensitive to small values near zero. |

---

## 6. Example `config.json`

```json
{
  "paths": {
    "images_obverse": "Data/Coins/Obverse",
    "images_reverse": "Data/Coins/Reverse",
    "findspot_data": "Data/Coins/numisdata.csv",
    "die_ground_truth": "Data/Coins/ground_truth.csv",
    "output_dir": "results",
    "temp_dir": "temp_flattened",
    "cache_dir": "cache",
    "dataset-obverse": "die_studie_obverse_12_aglp.json",
    "dataset-reverse": "die_studie_reverse_14_projhdbscan.json"
  },
  "visualization_settings": {
    "method_id": 5,
    "caching": true
  },
  "analysis_settings": {
    "target_side": "obverse",
    "active_pipeline": "auto_die_studies"
  },
  "pipelines": {
    "die_study_tool": {
        "install_path": "../DieStudyTool",
        "parameters": {
            "n_clusters": 50,
            "min_cluster_size": 2,
            "method": "hierarchical"
        }
    },
    "auto_die_studies": {
        "install_path": "../Auto-Die-Studies",
        "parameters": {
            "matching_algorithm": "XFeat",
            "matching_params": {
                "XFeat-TopK": 10000,
                "RoMa-Threshold": 0.9
            },
            "filtering": true,
            "clustering_algorithm": "AGLP"
        }
    }
    "die_study": {
        "install_path": "../2023_CAA_ClaReNet/Die_Study",
        "die_study_tool_path": "../DieStudyTool",
        "parameters": {
            "number_of_clusters": 50,
            "distance_computation_method": 13
        }
    }
  }
}