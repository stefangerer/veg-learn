import os
import shutil
import rasterio
import numpy as np
from config import config
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm
import numpy as np
import rasterio
from skimage.feature import graycomatrix, graycoprops
from config import config 

def generate_feature_names():
    specific_bands = config['feature_extraction'].get('specific_bands', [])
    statistics_list = config['feature_extraction']['statistics']
    indices = config['feature_extraction'].get('indices', [])
    include_textures = config['feature_extraction'].get('include_textures', False)
    textures = config['feature_extraction'].get('textures', [])

    feature_names = []
    
    # Generate names for statistical features for specific bands
    for band_idx in specific_bands:
        for stat in statistics_list:
            feature_names.append(f"{stat}_band_{band_idx}")
                    
    # Generate names for statistical features for vegetation indices
    for vi_name in indices:
        for stat in statistics_list:
            feature_names.append(f"{stat}_{vi_name}")

    # Add names for textural features if included
    if include_textures:
        for texture in textures:
            feature_names.append(f"{texture}_NIR")

    return feature_names

def calculate_textural_features(tiff_path):
    """
    Calculates specified textural features from the Near-Infrared (NIR) band of a TIFF file using GLCM.
    
    Parameters:
    - tiff_path: str, path to the TIFF file.

    Returns:
    - A dictionary with keys as feature names and values as the calculated features.
    """
    # Retrieve the list of textural features to calculate from the config
    textures = config['feature_extraction']['textures']
    
    try:
        with rasterio.open(tiff_path) as src:
            # Read the NIR band, assuming it is the 12th band
            nir_band = src.read(12)
            
            # Check if the range of values in NIR band is zero
            if nir_band.ptp() == 0:
                print("Warning: Range of values in NIR band is zero. Cannot perform normalization.")
                return None

            # Normalize NIR band to 8-bit to prepare for GLCM calculation
            nir_band_normalized = ((nir_band - nir_band.min()) / (nir_band.ptp() / 255.0)).astype(np.uint8)

            # Calculate the GLCM matrix
            glcm = graycomatrix(nir_band_normalized, distances=[1], angles=[0, 45, 90, 135], levels=256, symmetric=True, normed=True)
            results = {}

            # Calculate and store each requested textural feature
            for texture in textures:
                results[texture] = graycoprops(glcm, texture)[0, 0]

            return results
            
    except Exception as e:
        # Handle exceptions and print error messages
        print(f"Error calculating textural features for TIFF file: {tiff_path}")
        print(e)
        return None


def extract_features_from_tiff(tiff_path):
    """
    Extracts specified statistical and textural features from a TIFF file.
    
    Parameters:
    - tiff_path: str, path to the TIFF file.

    Returns:
    - A numpy array of extracted features, or None if an error occurs.
    """
    # Retrieve the list of statistics to calculate from the config
    statistics_list = config['feature_extraction']['statistics']
    specific_bands = config['feature_extraction'].get('specific_bands', [])

    try:
        with rasterio.open(tiff_path) as src:
            bands_data = []

            # Process specific bands if defined
            if specific_bands:
                for band_idx in specific_bands:
                    band = src.read(band_idx)
                    bands_data.append(band)

            # Always include bands from the 13th onwards
            for band_idx in range(13, src.count + 1):
                # Avoid re-adding a band if it's already included in specific_bands
                if band_idx not in specific_bands:
                    band = src.read(band_idx)
                    bands_data.append(band)

        # Initialize a dictionary to hold statistics for each band
        band_stats = {stat: [] for stat in statistics_list}

        # Calculate specified statistics for each band
        for band in bands_data:
            if 'mean' in statistics_list:
                band_stats['mean'].append(np.mean(band))
            if 'max' in statistics_list:
                band_stats['max'].append(np.max(band))
            if 'min' in statistics_list:
                band_stats['min'].append(np.min(band))
            if 'std' in statistics_list:
                band_stats['std'].append(np.std(band))
            if 'percentile_25' in statistics_list:
                band_stats['percentile_25'].append(np.percentile(band, 25))
            if 'percentile_50' in statistics_list: 
                band_stats['percentile_50'].append(np.percentile(band, 50))
            if 'percentile_75' in statistics_list:
                band_stats['percentile_75'].append(np.percentile(band, 75))
                
        # Flatten the statistics into a single feature list
        features = [value for stats in band_stats.values() for value in stats]

        # Add textural features if specified in the config
        if config['feature_extraction']['include_textures']:
            textural_features = calculate_textural_features(tiff_path) 
            features.extend(textural_features.values())
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error processing TIFF file: {tiff_path}")
        print(e)
        return None

def load_dataset(folder_path):
    """
    Loads and processes a dataset from a directory structure where each subfolder represents a class,
    and each TIFF image within those subfolders is an instance of that class.
    
    Parameters:
    - folder_path: str, path to the dataset directory.

    Returns:
    - X: np.ndarray, a feature matrix where each row is a feature vector of an image.
    - y: np.ndarray, a label vector where each element is the label of the corresponding row in X.
    - num_bands: int, the number of bands detected in the TIFF images.
    """
    X, y = [], []
    num_bands = 0
        
    # Iterate through each subfolder representing a class
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            label = subfolder  # Folder name is used as the label
            
            # Filter out and process each TIFF image in the subfolder
            for file_name in filter(lambda f: f.endswith(('.tiff', '.tif')), os.listdir(subfolder_path)):
                tiff_path = os.path.join(subfolder_path, file_name)
                
                if num_bands == 0:
                    with rasterio.open(tiff_path) as src:
                        num_bands = src.count
                
                features = extract_features_from_tiff(tiff_path)
                if features is not None:
                    X.append(features)
                    y.append(label)
    
    # Convert lists to numpy arrays for machine learning processing
    X = np.stack(X, axis=0)
    y = np.array(y)
  
    return X, y, num_bands






