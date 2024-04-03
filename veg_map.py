import os
import rasterio
from rasterio.windows import Window
import config
import numpy as np 
import create_patches
import extract_features
from joblib import load, Parallel, delayed
from rasterio.transform import from_origin
import re
import shutil  # Used for deleting directories

# Function to create necessary folders for map creation process
def create_map_folders(map_folder):
    """
    Creates the base folder structure needed for map generation.
    :param map_folder: Base directory for the map generation process.
    :return: Paths to the all_patches and all_patches_veg folders.
    """
    
    # Ensure base map folder exists
    if not os.path.exists(map_folder):
        os.makedirs(map_folder)
    
    # Create subdirectory for patches
    patches_folder = os.path.join(map_folder, "patches")
    if not os.path.exists(patches_folder):
        os.makedirs(patches_folder)
    
    # Further subdivide into specific patch types
    all_patches_folder = os.path.join(patches_folder, "all_patches")
    all_patches_veg_folder = os.path.join(patches_folder, "all_patches_veg")
    
    # Create the specific patch folders if they do not exist
    for folder in [all_patches_folder, all_patches_veg_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    print(f"Created necessary folders in: {map_folder}")
    return all_patches_folder, all_patches_veg_folder

# Function to create patches from input TIFF
def create_all_patches(input_tif, output_folder):
    """
    Generates image patches from the input TIFF file.
    :param input_tif: Path to the input TIFF file.
    :param output_folder: Directory where patches will be stored.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open input TIFF to calculate patch size and number of patches
    with rasterio.open(input_tif) as src:
        patch_size_meters = config.config['patch_creation']['patch_size'] * 2
        patch_size_pixels = int(patch_size_meters / abs(src.transform[0]))
        num_patches_width = (src.width + patch_size_pixels - 1) // patch_size_pixels

        # Generate coordinates for each patch
        patch_coords = []
        for y in range(0, src.height, patch_size_pixels):
            for x in range(0, src.width, patch_size_pixels):
                if x + patch_size_pixels <= src.width and y + patch_size_pixels <= src.height:
                    patch_coords.append((x, y))

        # Function to create a single patch based on coordinates
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
        
        # Parallelize patch creation for efficiency
        Parallel(n_jobs=-1)(delayed(create_patch)(patch_coord) for patch_coord in patch_coords)

# Function to flag and load valid features
def load_features_from_file(file_path):
    """
    Loads features from a single patch file, checking for validity based on pixel values.
    :param file_path: Path to the patch file.
    :return: Extracted features or None if the patch is invalid, alongside a validity flag.
    """
    with rasterio.open(file_path) as patch:
        patch_data = patch.read(range(1, 11))  # Read the first 10 bands

        # Transpose patch_data for the all-bands-255 check
        patch_data_transposed = np.transpose(patch_data, (1, 2, 0))
        
        # If any pixel has all its bands set to 255, consider the patch invalid
        if np.any(np.all(patch_data_transposed == 255, axis=-1)):
            return None, 0  # Return None and invalid flag
        else:
            features = extract_features.extract_features_from_tiff(file_path)
            return features, 1  # Return features and valid flag

# Function to iterate this process
def load_features_parallel(folder_path: str):
    """
    Parallelizes the loading of features from patch files within a given directory.
    :param folder_path: Directory containing the patch files.
    :return: A list of loaded features from valid patches and a validity array for the patches.
    """
    patch_files = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tif')],
        key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group())
    )

    # Use joblib to process files in parallel
    results = Parallel(n_jobs=-1)(delayed(load_features_from_file)(file_path) for file_path in patch_files)
    
    # Separate features and validity flags
    features_list, patch_validity = zip(*results)

    # Filter out None values from features_list
    valid_features_list = [feature for feature in features_list if feature is not None]
    
    return valid_features_list, np.array(patch_validity, dtype=int)

# Function to preprocess prediction vectors
def preprocess_predictions(prediction_vector):
    """
    Converts 'no_veg' predictions to a specific value and ensures integers.
    :param prediction_vector: The raw prediction vector.
    :return: The processed prediction vector.
    """
    processed_vector = []
    for prediction in prediction_vector:
        print(prediction)
        if prediction == "no_veg":
            processed_vector.append(9999)  # Use 9999 for 'no_veg'
        else:
            processed_vector.append(int(prediction))
    return processed_vector

# Function to generate the output raster from predictions
def generate_output_raster(folder_path, prediction_vector, patch_validity, output_folder):
    """
    Creates the final output raster based on predictions for each valid patch.
    :param folder_path: Directory containing the patches.
    :param prediction_vector: Vector of predictions for the patches.
    :param patch_validity: Array indicating the validity of each patch.
    :param output_folder: Directory where the output raster will be saved.
    """
    # Extract spatial reference and metadata from the first patch
    patch_files = sorted(os.listdir(folder_path))
    with rasterio.open(os.path.join(folder_path, patch_files[0])) as first_patch:
        metadata = first_patch.meta.copy()
        crs = first_patch.crs
        transform = first_patch.transform
        patch_size_pixels = first_patch.width

    num_patches = len(patch_validity)
    num_patches_side = int(np.sqrt(num_patches))  # Assuming square number of patches

    output_width = num_patches_side * patch_size_pixels
    output_height = output_width

    output_array = np.full((output_height, output_width), fill_value=9999, dtype=np.int32)

    prediction_idx = 0
    for idx in range(num_patches):
        if patch_validity[idx]:
            row = (idx // num_patches_side) * patch_size_pixels
            col = (idx % num_patches_side) * patch_size_pixels

            output_array[row:row+patch_size_pixels, col:col+patch_size_pixels] = prediction_vector[prediction_idx]
            prediction_idx += 1

    # Update and save metadata for the output raster
    metadata.update({
        'driver': 'GTiff',
        'height': output_height,
        'width': output_width,
        'transform': from_origin(transform.c, transform.f, transform.a, -transform.e),
        'crs': crs,
        'dtype': 'int32'
    })

    output_tif_path = os.path.join(output_folder, "prediction_map.tif")
    with rasterio.open(output_tif_path, 'w', **metadata) as dst:
        dst.write(output_array, 1)

# Main function to orchestrate map creation from input TIFFs using a specified model
def create_maps(input_folder, output_folder, model_path):
    """
    Processes each TIFF in the input folder to generate maps using the provided model.
    :param input_folder: Directory containing the input TIFF files.
    :param output_folder: Directory where the generated maps will be saved.
    :param model_path: Path to the trained model.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = load(model_path)

    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            tif_path = os.path.join(input_folder, filename)
            map_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
            create_map_single(tif_path, map_folder, model)

# Helper function to create a map for a single input TIFF
def create_map_single(input_tif, map_folder, model):
    """
    Generates a map for a single input TIFF using the provided model.
    :param input_tif: Path to the input TIFF file.
    :param map_folder: Base directory for the map generation process.
    :param model: The trained model used for predictions.
    """
    all_patches_folder, all_patches_veg_folder = create_map_folders(map_folder)

    create_all_patches(input_tif, all_patches_folder)
    create_patches.add_vegetation_indices_bands(all_patches_folder, config.config["feature_extraction"]["indices"], all_patches_veg_folder)

    features_list, patch_validity = load_features_parallel(all_patches_veg_folder)
    features = np.array([f for f in features_list if f is not None])
    if not features.any():
        print("No valid features to load. Exiting.")
        return

    predictions = model.predict(features)
    processed_predictions = preprocess_predictions(predictions)
    generate_output_raster(all_patches_veg_folder, processed_predictions, patch_validity, map_folder)

    # Cleanup: Delete the temporary patch folders
    shutil.rmtree(all_patches_folder)
    shutil.rmtree(all_patches_veg_folder)

    print("Map generated")
