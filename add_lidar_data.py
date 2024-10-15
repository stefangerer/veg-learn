import os
import shutil
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from joblib import Parallel, delayed

def clip_feature_to_patch(small_tif_path, bounds, lidar_path, feature_name, clipped_patch_dir):
    """Function to clip a single feature to a patch."""
    with rasterio.open(lidar_path) as lidar_tif:
        window = from_bounds(*bounds, transform=lidar_tif.transform)
        lidar_data = lidar_tif.read(window=window, resampling=Resampling.nearest)
        out_path = os.path.join(clipped_patch_dir, f"{feature_name}_{os.path.basename(small_tif_path)}")
        out_meta = lidar_tif.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": window.height,
            "width": window.width,
            "transform": rasterio.windows.transform(window, lidar_tif.transform)
        })
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(lidar_data)

def clip_features_to_patches(small_patch_dir, lidar_feature_folder, clipped_patch_dir, feature_list):
    """Clips LIDAR feature TIFFs to the bounds of each small patch TIFF using parallel processing with joblib."""

    lidar_features = {}
    for f in os.listdir(lidar_feature_folder):
        if f.endswith('.tif') and f.split('.')[0] in feature_list:
            base_name = f.split('.')[0]
            full_path = os.path.join(lidar_feature_folder, f)
            lidar_features[base_name] = full_path


    def get_bounds(tif_path):
        with rasterio.open(tif_path) as tif:
            return tif.bounds

    tasks = [(os.path.join(small_patch_dir, filename), 
            get_bounds(os.path.join(small_patch_dir, filename)), 
            lidar_path, 
            feature_name, 
            clipped_patch_dir) 
            for filename in os.listdir(small_patch_dir) if filename.endswith(".tif")
            for feature_name, lidar_path in lidar_features.items()]


    # Execute tasks in parallel
    Parallel(n_jobs=-1)(delayed(clip_feature_to_patch)(*task) for task in tasks)

def process_file(small_tiff_path, clipped_tiff_paths, output_file_path):
    """Function to merge one small TIFF with its corresponding clipped TIFFs, using the value of the pixel to the left for padding."""
    with rasterio.open(small_tiff_path) as small_tiff:
        small_data = small_tiff.read().astype(np.float32)
        new_profile = small_tiff.profile
        band_count = small_tiff.count

        max_height = small_data.shape[1]
        max_width = small_data.shape[2]

        # Determine maximum dimensions
        for clipped_tiff_path in clipped_tiff_paths:
            with rasterio.open(clipped_tiff_path) as clipped_tiff:
                max_height = max(max_height, clipped_tiff.height)
                max_width = max(max_width, clipped_tiff.width)

        # Read and combine data from all clipped TIFFs, with padding if necessary
        for clipped_tiff_path in clipped_tiff_paths:
            with rasterio.open(clipped_tiff_path) as clipped_tiff:
                clipped_data = clipped_tiff.read().astype(np.float32)
                if clipped_data.shape[1] != max_height or clipped_data.shape[2] != max_width:
                    # Pad the data array using the value of the pixel to the left
                    padded_data = np.full((clipped_tiff.count, max_height, max_width), np.nan, dtype=np.float32)
                    padded_data[:, :clipped_data.shape[1], :clipped_data.shape[2]] = clipped_data
                    # Apply padding for each band
                    for band in range(clipped_tiff.count):
                        for row in range(clipped_data.shape[1]):
                            # Start padding from the first column not covered by the original data
                            for col in range(clipped_data.shape[2], max_width):
                                padded_data[band, row, col] = padded_data[band, row, col-1]
                    clipped_data = padded_data

                small_data = np.concatenate((small_data, clipped_data), axis=0)
                band_count += clipped_tiff.count

        new_profile.update(count=band_count, dtype='float32')

        with rasterio.open(output_file_path, 'w', **new_profile) as dst:
            dst.write(small_data)

def merge_tiff_with_indices(small_patch_dir, clipped_patch_dir, output_dir, feature_list):
    """
    Merges small patch TIFF files with corresponding clipped patch TIFF files, incorporating additional band information.
    Uses parallel processing to enhance performance.
    """
    #if os.path.exists(output_dir):
    #    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    # Prepare tasks for parallel processing
    for filename in os.listdir(small_patch_dir):
        if filename.endswith(".tif"):
            small_tiff_path = os.path.join(small_patch_dir, filename)
            output_file_path = os.path.join(output_dir, filename)
            clipped_tiff_paths = [os.path.join(clipped_patch_dir, f"{feature}_{filename}")
                                  for feature in feature_list if os.path.exists(os.path.join(clipped_patch_dir, f"{feature}_{filename}"))]

            if clipped_tiff_paths:
                tasks.append((small_tiff_path, clipped_tiff_paths, output_file_path))

    # Execute merging in parallel
    Parallel(n_jobs=-1)(delayed(process_file)(*task) for task in tasks)

def process_lidar_with_patches(small_patches_base_folder, lidar_feature_folder, output_base_folder, feature_list):
    """
    Processes LIDAR features by clipping them to small patches in subfolders and then merging the clipped features with the original patches.
    """
    
    clipped_patch_dir = os.path.join(output_base_folder, "temp_clipped_patches")
    os.makedirs(clipped_patch_dir, exist_ok=True)

    # Process each subfolder within the base folder
    for subdir in os.listdir(small_patches_base_folder):
        subdir_path = os.path.join(small_patches_base_folder, subdir)
        if os.path.isdir(subdir_path):
            clip_features_to_patches(subdir_path, lidar_feature_folder, clipped_patch_dir, feature_list)
            merge_tiff_with_indices(subdir_path, clipped_patch_dir, os.path.join(output_base_folder, subdir), feature_list)
    # Remove the temporary directory
    shutil.rmtree(clipped_patch_dir)


def process_lidar_with_patches_map(small_patches_base_folder, lidar_feature_folder, output_base_folder, feature_list):
    """
    Processes LIDAR features by clipping them to small patches in subfolders and then merging the clipped features with the original patches.
    """
    
    clipped_patch_dir = os.path.join(small_patches_base_folder, "temp_clipped_patches")
    os.makedirs(clipped_patch_dir, exist_ok=True)

   
    clip_features_to_patches(small_patches_base_folder, lidar_feature_folder, clipped_patch_dir, feature_list)
    merge_tiff_with_indices(small_patches_base_folder, clipped_patch_dir, output_base_folder, feature_list)
    # Remove the temporary directory
    shutil.rmtree(clipped_patch_dir)