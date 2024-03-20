import os
import rasterio
from rasterio.windows import Window
import config
import numpy as np 
import create_patches
import extract_features
from joblib import load
import re

def create_map_folders(map_folder):
    
    # Base map folder
    if not os.path.exists(map_folder):
        os.makedirs(map_folder)
    
    # Patches folder within the map folder
    patches_folder = os.path.join(map_folder, "patches")
    if not os.path.exists(patches_folder):
        os.makedirs(patches_folder)
    
    # Subfolders for patches: "all_patches" and "all_patches_veg"
    all_patches_folder = os.path.join(patches_folder, "all_patches")
    all_patches_veg_folder = os.path.join(patches_folder, "all_patches_veg")
    
    for folder in [all_patches_folder, all_patches_veg_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    print(f"Created necessary folders in: {map_folder}")
    return all_patches_folder, all_patches_veg_folder


def extract_center_square_meters(input_tif_path, output_folder, square_size_meters):
    """Extracts a centered square of specified size from a TIFF file and saves it to the specified output folder."""
    with rasterio.open(input_tif_path) as src:
        pixel_width_meters = src.transform[0]  # Pixel width in meters
        pixel_height_meters = abs(src.transform[4])  # Pixel height in meters
        
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
        
        # Construct file name
        output_file_name = f"{square_size_meters}_center_ortho.tif"

        # Construct the full path to save the output TIFF
        output_tif_path = os.path.join(output_folder, output_file_name)
        
        # Save the extracted square as a new TIFF
        with rasterio.open(output_tif_path, 'w', **out_meta) as dst:
            dst.write(data)

        print(f"Saved TIFF to {output_tif_path}")


def create_all_patches(input_tif, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with rasterio.open(input_tif) as src:
        # Convert patch size from meters to pixels
        patch_size_meters = config.config['patch_creation']['patch_size'] * 2
        patch_size_pixels = int(patch_size_meters / abs(src.transform[0]))
        
        # Initialize a counter for the patch filenames
        patch_number = 1

        for y in range(0, src.height, patch_size_pixels):
            for x in range(0, src.width, patch_size_pixels):
                if x + patch_size_pixels > src.width or y + patch_size_pixels > src.height:
                    continue  # Skip patches that extend beyond the image boundary

                window = Window(x, y, patch_size_pixels, patch_size_pixels)
                patch_data = src.read(window=window)
                
                # Check if the patch contains only valid pixels
                # Adjust this condition based on your definition of a valid pixel value
                # Here, we're assuming that a valid pixel is non-zero and not NaN
                #if np.all(patch_data == 255, axis=0).any():
                #    continue
                
                # Use the patch_number for the filename instead of x and y coordinates
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
                
                # Increment the patch counter after writing each patch
                patch_number += 1

def load_features(folder_path: str):
    X = []

    # Adjust the lambda function to work with the new naming convention 'features_patch_{patch_number}.tif'
    patch_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.tif')],
        key=lambda f: int(f.split('_')[2].split('.')[0])
    )

    for file_name in patch_files:
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(('.tiff', '.tif')) and os.path.isfile(file_path):
            # Extract features from the TIFF file
            features = extract_features.extract_features_from_tiff(file_path)
            if features is not None:
                X.append(features)

    # Convert list of features to numpy array for machine learning processing
    if X:
        X = np.stack(X, axis=0)
    else:
        # Handle the case where no features were extracted to avoid errors
        X = np.array([])

    return X

def preprocess_predictions(prediction_vector):
    """Preprocess the prediction vector, converting 'no_veg' to 9999 and ensuring all values are integers."""
    processed_vector = []
    for prediction in prediction_vector:
        print(prediction)
        if prediction == "no_veg":
            processed_vector.append(9999)
        else:
            processed_vector.append(int(prediction))
    return processed_vector

def generate_output_raster(folder_path, prediction_vector, output_tif_path):
    prediction_vector = preprocess_predictions(prediction_vector)
    patch_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.tif')],
        key=lambda f: int(f.split('_')[2].split('.')[0])
    )

    with rasterio.open(os.path.join(folder_path, patch_files[0])) as first_patch:
        patch_size = first_patch.width  # Assuming square patches
        metadata = first_patch.meta.copy()
        pixel_width = first_patch.transform.a
        pixel_height = -first_patch.transform.e
        crs = first_patch.crs
        transform = first_patch.transform

    # No need to calculate num_patches_per_row based on the first_patch.width / patch_size
    # Since we are now naming files sequentially, we just need to know the total number of patches
    num_patches_per_row = int(np.sqrt(len(patch_files)))  # Assuming a square grid of patches

    output_width = num_patches_per_row * patch_size
    output_height = output_width  # Assuming a square layout

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
    all_patches_folder, all_patches_veg_folder = create_map_folders(map_folder)

    extract_center_square_meters(input_tif, map_folder, map_size)
    
    center_square_tif = os.path.join(map_folder, f"{map_size}_center_ortho.tif")
    
    create_all_patches(center_square_tif, all_patches_folder)
    create_patches.add_vegetation_indices_bands(all_patches_folder, config.config["feature_extraction"]["indices"], all_patches_veg_folder)
    
    model = load(model_path)
    features = load_features(all_patches_veg_folder)
    if features.size > 0:
        print("features_loaded")
        predictions = model.predict(features)
        print("classes predicted")
    else:
        print("No features to load. Exiting.")
        return
        # Handle the situation appropriately, perhaps by exiting or skipping the prediction step.

    prediction_map = os.path.join(map_folder, f"{map_size}_prediction_map.tif")

    generate_output_raster(all_patches_veg_folder, predictions, prediction_map)

    print("map generated")


 