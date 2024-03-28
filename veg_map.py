import os
import rasterio
from rasterio.windows import Window
import config
import numpy as np 
import create_patches
import extract_features
from joblib import load, Parallel, delayed
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

def create_all_patches(input_tif, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get necessary information from the input_tif
    with rasterio.open(input_tif) as src:
        patch_size_meters = config.config['patch_creation']['patch_size'] * 2
        patch_size_pixels = int(patch_size_meters / abs(src.transform[0]))
        num_patches_width = (src.width + patch_size_pixels - 1) // patch_size_pixels
        num_patches_height = (src.height + patch_size_pixels - 1) // patch_size_pixels

        # Generate patch coordinates
        patch_coords = []
        for y in range(0, src.height, patch_size_pixels):
            for x in range(0, src.width, patch_size_pixels):
                if x + patch_size_pixels <= src.width and y + patch_size_pixels <= src.height:
                    patch_coords.append((x, y))

        # Define a function to create a patch
        def create_patch(patch_coord):
            x, y = patch_coord
            with rasterio.open(input_tif) as src:
                window = Window(x, y, patch_size_pixels, patch_size_pixels)
                patch_data = src.read(window=window)
                patch_number = y // patch_size_pixels * num_patches_width + x // patch_size_pixels + 1
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
        
        # Create patches in parallel
        Parallel(n_jobs=-1)(delayed(create_patch)(patch_coord) for patch_coord in patch_coords)


def load_features_parallel(folder_path: str):
    def load_features_from_file(file_path):
        with rasterio.open(file_path) as patch:
            patch_data = patch.read(range(1, 11))  # Read the first 10 bands
            if np.all(patch_data == 255):  # Check if all bands are equal to 255
                print(f"Ignoring invalid patch: {file_path}")
                return None
            else:
                features = extract_features.extract_features_from_tiff(file_path)
                print(f"Feature loaded: {file_path}")
                return features

    patch_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.tif')],
        key=lambda f: int(f.split('_')[2].split('.')[0])
    )

    # Adjust the number of workers according to your system's capabilities
    num_workers = min(len(patch_files), os.cpu_count())
    print(f"Number of workers: {num_workers}")

    features_list = Parallel(n_jobs=num_workers)(
        delayed(load_features_from_file)(os.path.join(folder_path, file_name)) for file_name in patch_files
    )

    return features_list

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

def generate_output_raster(folder_path, prediction_vector, valid_indices, output_folder):
    patch_files = sorted(os.listdir(folder_path))

    with rasterio.open(os.path.join(folder_path, patch_files[0])) as first_patch:
        patch_size = first_patch.width  # Assuming square patches
        metadata = first_patch.meta.copy()
        pixel_width = first_patch.transform.a
        pixel_height = -first_patch.transform.e
        crs = first_patch.crs
        transform = first_patch.transform

    num_patches_per_row = int(np.sqrt(len(patch_files)))  # Assuming a square grid of patches

    output_width = num_patches_per_row * patch_size
    output_height = output_width  # Assuming a square layout

    output_array = np.full((output_height, output_width), fill_value=9999, dtype=np.int32)

    # Loop through all patch files
    for i, idx in enumerate(valid_indices):
        patch_file = patch_files[idx]
        with rasterio.open(os.path.join(folder_path, patch_file)) as patch:
            row = i // num_patches_per_row
            column = i % num_patches_per_row
            start_row = row * patch_size
            start_col = column * patch_size
            output_array[start_row:start_row + patch_size, start_col:start_col + patch_size] = prediction_vector[i]

    # Save the output raster
    output_tif_path = os.path.join(output_folder, "prediction_map.tif")
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

def create_maps(input_folder, output_folder, model_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = load(model_path)

    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            tif_path = os.path.join(input_folder, filename)
            map_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
            create_map_single(tif_path, map_folder, model)

def create_map_single(input_tif, map_folder, model):
    all_patches_folder, all_patches_veg_folder = create_map_folders(map_folder)

    create_all_patches(input_tif, all_patches_folder)
    create_patches.add_vegetation_indices_bands(all_patches_folder, config.config["feature_extraction"]["indices"], all_patches_veg_folder)

    # Load features from valid patches only using parallel processing
    features_list = load_features_parallel(all_patches_veg_folder)
    valid_indices = [i for i, f in enumerate(features_list) if f is not None]
    features = np.array([f for f in features_list if f is not None])
    print(features)
    if not features.any():
        print("No valid features to load. Exiting.")
        return

    # Make predictions only on valid features
    predictions = model.predict(features)

    # Preprocess the predictions
    processed_predictions = preprocess_predictions(predictions)

    # Generate output raster
    generate_output_raster(all_patches_veg_folder, processed_predictions, valid_indices, map_folder)

    print("Map generated")



#input_folder = r"D:\DATA\Schrankogel\Metashape_Project_Schrankogel\Ortho\LZW_10000x10000px_tiles"
#output_folder = r"D:\DATA\Masterthesis Angerer\veg_learn\results\20_03_24\maps"
#model_path = r"D:\DATA\Masterthesis Angerer\veg_learn\results\20_03_24\loocv_test_t510_2\rf_model.joblib"
#create_maps(input_folder, output_folder, model_path)



 