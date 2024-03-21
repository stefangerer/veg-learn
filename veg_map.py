import os
import rasterio
from rasterio.windows import Window
import numpy as np
from joblib import load
import config
import create_patches
import extract_features
import re

def create_map_folders(map_folder):
    """
    Create necessary folders for the map if they don't exist already.

    Args:
        map_folder (str): Path to the main map folder.
        
    Returns:
        tuple: Paths to all_patches_folder and all_patches_veg_folder.
    """
    if not os.path.exists(map_folder):
        os.makedirs(map_folder)

    patches_folder = os.path.join(map_folder, "patches")
    if not os.path.exists(patches_folder):
        os.makedirs(patches_folder)

    all_patches_folder = os.path.join(patches_folder, "all_patches")
    all_patches_veg_folder = os.path.join(patches_folder, "all_patches_veg")

    for folder in [all_patches_folder, all_patches_veg_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    print(f"Created necessary folders in: {map_folder}")
    return all_patches_folder, all_patches_veg_folder

def extract_center_square_meters(input_tif_path, output_folder, square_size_meters):
    """
    Extract a centered square of specified size from a TIFF file and save it to the specified output folder.

    Args:
        input_tif_path (str): Path to the input TIFF file.
        output_folder (str): Path to the output folder.
        square_size_meters (int): Size of the square in meters.
    """
    with rasterio.open(input_tif_path) as src:
        pixel_width_meters = src.transform[0]
        pixel_height_meters = abs(src.transform[4])
        pixels_across_width = int(square_size_meters / pixel_width_meters)
        pixels_across_height = int(square_size_meters / pixel_height_meters)
        center_x, center_y = src.width // 2, src.height // 2
        offset_x = pixels_across_width // 2
        offset_y = pixels_across_height // 2
        window = Window(center_x - offset_x, center_y - offset_y, pixels_across_width, pixels_across_height)
        data = src.read(window=window)
        new_transform = src.window_transform(window)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": pixels_across_height,
            "width": pixels_across_width,
            "transform": new_transform
        })
        output_file_name = f"{square_size_meters}_center_ortho.tif"
        output_tif_path = os.path.join(output_folder, output_file_name)
        with rasterio.open(output_tif_path, 'w', **out_meta) as dst:
            dst.write(data)

        print(f"Saved TIFF to {output_tif_path}")

def create_all_patches(input_tif, output_folder):
    """
    Create patches from the input TIFF file.

    Args:
        input_tif (str): Path to the input TIFF file.
        output_folder (str): Path to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with rasterio.open(input_tif) as src:
        patch_size_meters = config.config['patch_creation']['patch_size'] * 2
        patch_size_pixels = int(patch_size_meters / abs(src.transform[0]))
        patch_number = 1

        for y in range(0, src.height, patch_size_pixels):
            for x in range(0, src.width, patch_size_pixels):
                if x + patch_size_pixels > src.width or y + patch_size_pixels > src.height:
                    continue

                window = Window(x, y, patch_size_pixels, patch_size_pixels)
                patch_data = src.read(window=window)
                patch_path = os.path.join(output_folder, f"patch_{patch_number}.tif")
                with rasterio.open(
                    patch_path, 'w',
                    driver='GTiff',
                    height=patch_size_pixels,
                    width=patch_size_pixels,
                    count=src.count,
                    dtype=src.dtypes[0],
                    crs=src.crs,
                    transform=rasterio.windows.transform(window, src.transform)
                ) as patch_dst:
                    patch_dst.write(patch_data)
                
                patch_number += 1

def load_features(folder_path: str):
    """
    Load features from TIFF files in the specified folder.

    Args:
        folder_path (str): Path to the folder containing TIFF files.

    Returns:
        numpy.ndarray: Array of features.
    """
    X = []

    patch_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.tif')],
        key=lambda f: int(re.search(r'\d+', f).group())
    )

    for file_name in patch_files:
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(('.tiff', '.tif')) and os.path.isfile(file_path):
            features = extract_features.extract_features_from_tiff(file_path)
            if features is not None:
                X.append(features)

    if X:
        X = np.stack(X, axis=0)
    else:
        X = np.array([])

    return X

def preprocess_predictions(prediction_vector):
    """
    Preprocess the prediction vector.

    Args:
        prediction_vector (list): List of prediction values.

    Returns:
        list: Preprocessed prediction vector.
    """
    processed_vector = []
    for prediction in prediction_vector:
        if prediction == "no_veg":
            processed_vector.append(9999)
        else:
            processed_vector.append(int(prediction))
    return processed_vector

def generate_output_raster(folder_path, prediction_vector, output_tif_path):
    """
    Generate an output raster based on prediction vector.

    Args:
        folder_path (str): Path to the folder containing patches.
        prediction_vector (list): List of predicted values.
        output_tif_path (str): Path to the output raster file.
    """
    prediction_vector = preprocess_predictions(prediction_vector)
    patch_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.tif')],
        key=lambda f: int(re.search(r'\d+', f).group())
    )

    with rasterio.open(os.path.join(folder_path, patch_files[0])) as first_patch:
        patch_size = first_patch.width
        metadata = first_patch.meta.copy()
        pixel_width = first_patch.transform.a
        pixel_height = -first_patch.transform.e
        crs = first_patch.crs
        transform = first_patch.transform

    num_patches_per_row = int(np.sqrt(len(patch_files)))

    output_width = num_patches_per_row * patch_size
    output_height = output_width

    output_array = np.zeros((output_height, output_width), dtype=np.int32)

    for i, prediction in enumerate(prediction_vector):
        row = i // num_patches_per_row
        column = i % num_patches_per_row
        start_row = row * patch_size
        start_col = column * patch_size
        output_array[start_row:start_row + patch_size, start_col:start_col + patch_size] = prediction

    metadata.update({
        'driver': 'GTiff',
        'height': output_height,
        'width': output_width,
        'transform': rasterio.transform.from_origin(transform.c, transform.f, pixel_width, pixel_height),
        'crs': crs,
        'dtype': 'int32'
    })

    with rasterio.open(output_tif_path, 'w', **metadata) as dst:
        dst.write(output_array, 1)

def create_map(input_tif, map_size, map_folder, model_path): 
    """
    Create a map based on the input data.

    Args:
        input_tif (str): Path to the input TIFF file.
        map_size (int): Size of the map.
        map_folder (str): Path to the main map folder.
        model_path (str): Path to the trained model.
    """
    # Create necessary folders for the map
    all_patches_folder, all_patches_veg_folder = create_map_folders(map_folder)

    # Extract a centered square from the input TIFF
    extract_center_square_meters(input_tif, map_folder, map_size)
    
    # Path to the centered square TIFF
    center_square_tif = os.path.join(map_folder, f"{map_size}_center_ortho.tif")
    
    # Create patches from the centered square TIFF
    create_all_patches(center_square_tif, all_patches_folder)
    
    # Add vegetation indices bands to the patches
    create_patches.add_vegetation_indices_bands(all_patches_folder, config.config["feature_extraction"]["indices"], all_patches_veg_folder)
    
    # Load the trained model
    model = load(model_path)
    
    # Load features from the vegetation patches
    features = load_features(all_patches_veg_folder)
    if features.size > 0:
        print("Features loaded")
        # Predict classes using the loaded features
        predictions = model.predict(features)
        print("Classes predicted")
    else:
        print("No features to load. Exiting.")
        return

    # Path to the prediction map
    prediction_map = os.path.join(map_folder, f"{map_size}_prediction_map.tif")

    # Generate the output raster based on the predictions
    generate_output_raster(all_patches_veg_folder, predictions, prediction_map)

    print("Map generated")


