import os
import shutil
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

def clip_features_to_patches(small_patch_dir, lidar_feature_folder, clipped_patch_dir, feature_list):
    """
    Clips LIDAR feature TIFFs to the bounds of each small patch TIFF within subfolders.
    """
    lidar_features = {f.split('.')[0]: os.path.join(lidar_feature_folder, f)
                      for f in os.listdir(lidar_feature_folder) if f.endswith('.tif') and f.split('.')[0] in feature_list}

    for filename in os.listdir(small_patch_dir):
        if filename.endswith(".tif"):
            small_tif_path = os.path.join(small_patch_dir, filename)
            with rasterio.open(small_tif_path) as small_tif:
                bounds = small_tif.bounds
                for feature_name, lidar_path in lidar_features.items():
                    with rasterio.open(lidar_path) as lidar_tif:
                        window = from_bounds(*bounds, transform=lidar_tif.transform)
                        lidar_data = lidar_tif.read(window=window, resampling=Resampling.nearest)
                        out_path = os.path.join(clipped_patch_dir, f"{feature_name}_{filename}")
                        out_meta = lidar_tif.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": window.height,
                            "width": window.width,
                            "transform": rasterio.windows.transform(window, lidar_tif.transform)
                        })
                        with rasterio.open(out_path, "w", **out_meta) as dest:
                            dest.write(lidar_data)
                            print(f"Clipped and wrote data to {out_path}")

    print(f"Completed clipping to {clipped_patch_dir}")

def merge_tiff_with_indices(small_patch_dir, clipped_patch_dir, output_dir, feature_list):
    """
    Merges small patch TIFF files with corresponding clipped patch TIFF files, incorporating additional band information.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    def process_file(small_tiff_path, clipped_tiff_paths, output_file_path):
        with rasterio.open(small_tiff_path) as small_tiff:
            small_data = small_tiff.read().astype(np.float32)
            new_profile = small_tiff.profile
            band_count = small_tiff.count

            # Read and combine data from all clipped TIFFs
            for clipped_tiff_path in clipped_tiff_paths:
                with rasterio.open(clipped_tiff_path) as clipped_tiff:
                    clipped_data = clipped_tiff.read().astype(np.float32)
                    small_data = np.concatenate((small_data, clipped_data), axis=0)
                    band_count += clipped_tiff.count

            new_profile.update(count=band_count, dtype='float32')

            with rasterio.open(output_file_path, 'w', **new_profile) as dst:
                dst.write(small_data)

    # Process each file in the small patch directory
    for filename in os.listdir(small_patch_dir):
        if filename.endswith(".tif"):
            small_tiff_path = os.path.join(small_patch_dir, filename)
            output_file_path = os.path.join(output_dir, filename)

            # Gather all corresponding clipped TIFF paths
            clipped_tiff_paths = [os.path.join(clipped_patch_dir, f"{feature}_{filename}")
                                  for feature in feature_list if os.path.exists(os.path.join(clipped_patch_dir, f"{feature}_{filename}"))]

            if clipped_tiff_paths:
                process_file(small_tiff_path, clipped_tiff_paths, output_file_path)

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

