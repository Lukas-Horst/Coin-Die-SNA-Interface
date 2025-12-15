__author__ = 'Lukas Horst'

import csv
import json
import os
import re
import shutil
from typing import Optional, List, Tuple

from tqdm import tqdm


def flatten_image_directory(source_path: str, target_path: str,
                            allowed_extensions: Optional[List[str]] = None) -> None:
    """
    Copies all images with specific extensions from a potentially nested source directory
    into a flat target directory. Displays a progress bar in the terminal.

    If the target directory already exists, its CONTENT will be cleared before copying.
    The directory itself remains to avoid permission errors.

    Args:
        source_path (str): The path to the source directory containing images (can be nested).
        target_path (str): The path where the flattened images should be stored.
        allowed_extensions (List[str], optional): A list of file extensions to include.
                                                  Defaults to ['.jpg', '.png'].
    """
    # Set default extensions if none provided
    if allowed_extensions is None:
        allowed_extensions = ['.jpg', '.png']

    # Ensure extensions are lowercase for comparison
    allowed_extensions = [ext.lower() for ext in allowed_extensions]

    # Preparing target directory
    if os.path.exists(target_path):
        print(f"Cleaning content of target directory '{target_path}'...")
        # Iterate over all items in the directory to delete them individually
        for filename in os.listdir(target_path):
            file_path = os.path.join(target_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except PermissionError:
                # Warn the user, that he has no permission
                print(f"Warning: Permission denied. Could not delete '{file_path}'. Skipping.")
            except Exception as e:
                print(f"Warning: Failed to delete '{file_path}'. Reason: {e}")
    else:
        # If directory does not exist, create it
        try:
            os.makedirs(target_path)
        except OSError as e:
            print(f"Error: Could not create directory '{target_path}'. {e}")
            return

    print(f"Analyzing source directory '{source_path}'...")

    # 1. First Pass: Count valid files to initialize the progress bar correctly
    total_files = 0
    for root, _, files in os.walk(source_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in allowed_extensions):
                total_files += 1

    if total_files == 0:
        print("No matching images found in source directory.")
        return

    print(f"Found {total_files} images. Starting copy process...")

    copied_count = 0
    skipped_count = 0

    # 2. Second Pass: Copy files with progress bar
    with tqdm(total=total_files, unit="img", desc="Flattening") as pbar:
        for root, _, files in os.walk(source_path):
            for file in files:
                # Filter by extension
                if any(file.lower().endswith(ext) for ext in allowed_extensions):
                    source_file = os.path.join(root, file)
                    target_file = os.path.join(target_path, file)

                    # Check for duplicate filenames to prevent overwriting
                    if os.path.exists(target_file):
                        # Skip duplicates and log it
                        tqdm.write(
                            f"Warning: Duplicate filename '{file}'. Skipping {source_file}.")
                        skipped_count += 1
                        pbar.update(1)
                        continue

                    # Copy the file
                    try:
                        shutil.copy2(source_file, target_file)
                        copied_count += 1
                    except Exception as e:
                        tqdm.write(f"Error copying {file}: {e}")

                    # Update progress bar by 1
                    pbar.update(1)

    print(f"\nProcess completed.")
    print(f"Successfully copied: {copied_count}")
    print(f"Skipped (duplicates): {skipped_count}")


def read_cluster_ids_from_txt(file_path: str) -> List[int]:
    """
    Reads a text file containing line-separated cluster IDs and converts them into a list of integers.

    This function is specifically designed to parse the output format of the 'Auto-Die-Studies' pipeline,
    which provides a raw list of IDs without filenames.

    Args:
        file_path (str): The absolute or relative path to the text file.

    Returns:
        List[int]: A list containing the cluster IDs as integers. Returns an empty list if an error occurs.
    """
    cluster_ids = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Remove whitespace and newline characters
                clean_line = line.strip()

                # Skip empty lines to prevent errors
                if clean_line:
                    try:
                        cluster_id = int(clean_line)
                        cluster_ids.append(cluster_id)
                    except ValueError:
                        print(
                            f"Warning: Could not convert line '{clean_line}' to integer. Skipping.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return []

    return cluster_ids


def create_clustering_csv(cluster_ids: List[int], images_dir: str, output_csv_path: str,
                          cluster_column_name: str = "final_CL",
                          valid_extensions: Optional[List[str]] = None) -> None:
    """
        Creates a CSV file mapping image filenames to cluster IDs.

        This function is used to standardize the output of pipelines that only provide 
        an implicit ordered list of cluster IDs (like 'Auto-Die-Studies'). It matches 
        the IDs with the alphanumerically sorted images from the source directory.

        Args:
            cluster_ids (List[int]): The list of cluster IDs.
            images_dir (str): Path to the directory containing the images.
            output_csv_path (str): Full path (including filename) where the CSV will be saved.
            cluster_column_name (str, optional): Name of the cluster ID column. 
                                                 Defaults to "final_CL".
            valid_extensions (List[str], optional): List of allowed file extensions.
                                                    Defaults to ['.jpg', '.png'].
        """
    # 1. Setup extensions
    if valid_extensions is None:
        valid_extensions = ['.jpg', '.png']

    # Convert to tuple for endswith() and ensure lowercase
    extensions_tuple = tuple(ext.lower() for ext in valid_extensions)

    # 2. Get and sort image files
    try:
        image_files = sorted(
            [f for f in os.listdir(images_dir) if f.lower().endswith(extensions_tuple)])
    except FileNotFoundError:
        print(f"Error: Image directory '{images_dir}' not found.")
        return

    # 3. Validation
    if len(cluster_ids) != len(image_files):
        print(f"Error: Mismatch between number of cluster IDs ({len(cluster_ids)}) "
              f"and found images ({len(image_files)}).")
        print("Please check if the image directory matches the one used for the analysis.")
        return

    # 4. Create CSV Data
    # Structure: object_number, [cluster_column_name], path
    rows = []
    for filename, cluster_id in zip(image_files, cluster_ids):
        # Construct the relative path as used in the example CSV
        # We ensure forward slashes for compatibility
        file_path = os.path.join(images_dir, filename).replace("\\", "/")

        rows.append(
            {"object_number": filename, cluster_column_name: cluster_id, "path": file_path})

    # 5. Write to CSV
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["object_number", cluster_column_name, "path"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(rows)

        print(f"Successfully created CSV at '{output_csv_path}' with {len(rows)} entries.")

    except Exception as e:
        print(f"Error writing CSV file: {e}")


def convert_csv_to_sna_json(csv_path: str, cluster_col: str, image_col: str,
                            output_json_path: str) -> None:
    """
    Converts a clustering CSV file into the JSON format required by the 'Coin-Die-SNA' application.

    It attempts to robustly extract the coin ID from the filename by splitting at common
    delimiters like '_', '-', '.', or spaces.

    Args:
        csv_path (str): Path to the source CSV file.
        cluster_col (str): The column name containing the cluster IDs (e.g., 'final_obverse_CL').
        image_col (str): The column name containing the image filenames (e.g., 'object_number').
        output_json_path (str): Path where the resulting JSON file will be saved.
    """
    results = {}

    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        return

    try:
        with open(csv_path, mode='r', encoding='utf-8') as csvfile:
            # Analyze CSV dialect (delimiter, etc.) to support both comma and semicolon
            try:
                dialect = csv.Sniffer().sniff(csvfile.read(1024))
                csvfile.seek(0)
                reader = csv.DictReader(csvfile, dialect=dialect)
            except csv.Error:
                # Fallback if sniffing fails (e.g. file too short): assume standard comma
                csvfile.seek(0)
                reader = csv.DictReader(csvfile)

            # Validate columns
            if not reader.fieldnames or cluster_col not in reader.fieldnames or image_col not in reader.fieldnames:
                print(f"Error: Columns '{cluster_col}' or '{image_col}' not found in CSV.")
                print(f"Available columns: {reader.fieldnames}")
                return

            for row in reader:
                filename = row[image_col]
                cluster_id = row[cluster_col]

                if not filename:
                    continue

                # Logic to extract Coin ID:
                try:
                    # 1. Remove file extension first (e.g. "132_a.jpg" -> "132_a")
                    name_stem = os.path.splitext(filename)[0]

                    # 2. Split by common delimiters: underscore (_), hyphen (-), dot (.), or space ( )
                    # The regex [_\-\.\s]+ matches one or more of these characters.
                    parts = re.split(r'[_\-\.\s]+', name_stem)

                    # 3. Take the first part as the ID
                    if parts:
                        coin_id = parts[0]
                    else:
                        # Fallback: if split returns empty (unlikely), take the whole stem
                        coin_id = name_stem

                    # Add to result dict
                    results[coin_id] = str(cluster_id)
                except Exception as e:
                    print(f"Warning: Could not parse filename '{filename}': {e}")

        # Ensure output directory exists
        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Write to JSON
        with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(results, jsonfile, indent=4)

        print(
            f"Successfully converted CSV to JSON: '{output_json_path}' ({len(results)} entries).")

    except Exception as e:
        print(f"Error converting CSV to JSON: {e}")


def get_files_from_directory_recursive(directory_path: str, allowed_extensions: Optional[Tuple[
    str, ...]] = None) -> List[Tuple[str, str]]:
    """
    Retrieves a list of full file paths and file names from a specified directory
    and all its subdirectories (recursive search).
    Optionally filters the files based on a tuple of allowed extensions.

    :param directory_path: The string path to the directory to search.
    :param allowed_extensions: A tuple of strings representing allowed file extensions (e.g., ('.png', '.jpg')).
                               If None, all files are returned.
    :return: A list of tuples, where each tuple is (full_file_path, file_name).
    """
    file_list: List[Tuple[str, str]] = []

    # Check if the directory exists to prevent runtime errors
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return []

    # os.walk iterates through the directory tree rooted at directory_path
    # It yields a 3-tuple (dirpath, dirnames, filenames) for each directory
    for root, _, file_names in os.walk(directory_path):
        # 'root' is the current directory path being walked
        # 'file_names' is a list of file names in the current 'root' directory

        for file_name in file_names:
            # Construct the full file path by joining the current root and the filename
            full_path = os.path.join(root, file_name)

            # Check for allowed extensions if a filter is provided
            if allowed_extensions:
                # To ensure case-insensitive comparison, we convert the filename to lowercase.
                if file_name.lower().endswith(allowed_extensions):
                    # Append the tuple (full_path, file_name)
                    file_list.append((full_path, file_name))
            else:
                # If no filter is set, add every file found
                # Append the tuple (full_path, file_name)
                file_list.append((full_path, file_name))

    return file_list


def find_file(start_path: str, file_name: str) -> str or None:
    """
    Recursively searches for a specific file starting from a given directory path.

    Args:
        start_path (str): The directory path where the search should begin.
        file_name (str): The name of the file to search for (including extension).

    Returns:
        str or None: The full path of the file if found, otherwise None.
    """
    # Walk through the directory tree starting from startPath
    for root, dirs, files in os.walk(start_path):
        # Check if the target file name is in the list of files in the current directory (root)
        if file_name in files:
            # If found, return the full path by joining the current root directory and the file name
            return os.path.join(root, file_name)

    # If the loop finishes without finding the file, return None
    return None


if __name__ == "__main__":
    pass
