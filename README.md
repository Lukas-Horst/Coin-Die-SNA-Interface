# Coin-Die-SNA-Interface

This repository serves as a modular interface to execute various numismatic die study analysis pipelines. It acts as the backend processing layer for the **Coin-Die-SNA** web application, standardizing the execution logic and output formats of different computer vision algorithms.

## Supported Analysis Pipelines

This interface currently supports the following pipelines. Please ensure you have the respective repositories installed or linked if you intend to use them:

* **Auto-Die-Studies** (Cornet et al.)
    * [https://github.com/ClementCornet/Auto-Die-Studies](https://github.com/ClementCornet/Auto-Die-Studies)
* **DieStudyTool** (Urjarm / Fiedler)
    * [https://github.com/Urjarm/DieStudyTool](https://github.com/Urjarm/DieStudyTool)
* **Die_Study** (ClaReNet / Frankfurt-BigDataLab)
    * [https://github.com/Frankfurt-BigDataLab/2023_CAA_ClaReNet/tree/main/Die_Study](https://github.com/Frankfurt-BigDataLab/2023_CAA_ClaReNet/tree/main/Die_Study)

## Configuration & Usage

To run an analysis, you must configure the central `config.json` file. This file controls input paths, output locations, and the selection of the active algorithm.

For a successful execution, the following keys in the `paths` and `analysis_settings` sections are essential:

* **`images_obverse`**: Absolute path to the folder containing obverse images.
* **`images_reverse`**: Absolute path to the folder containing reverse images.
* **`output_dir`**: Directory where the resulting JSON/CSV files will be saved.
* **`temp_dir`**: Temporary directory for image processing (cleared on every run).
* **`target_side`**: Defines which side to analyze in the current run (`"obverse"` or `"reverse"`).
* **`active_pipeline`**: Selects the algorithm to use (e.g., `"auto_die_studies"`, `"die_study_tool"`, or `"die_study"`).

Depending on the `active_pipeline` chosen, you must also configure the specific section for that tool within the `pipelines` object (e.g., setting the `install_path`).

## Documentation

For a detailed explanation of every configuration parameter, including specific settings for the individual pipelines and **full configuration examples**, please refer to the documentation:

[**📖 Configuration Documentation (config_doc_en.md)**](https://github.com/Lukas-Horst/Coin-Die-SNA-Interface/blob/main/config/config_doc_en.md)