import os
import math
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import joblib
import tempfile
import shutil
from config import config
from calculate_indices import calculate_ndvi, calculate_gndvi, calculate_ndre, calculate_mcari, calculate_ndgi, calculate_pri, calculate_psri, calculate_sr, calculate_savi, calculate_vari
from extract_features import extract_features_from_tiff

# Load the pre-trained Random Forest model
model_path = os.path.join(config["classification"]["results_folder"], 'rf_model.joblib')
rf_model = joblib.load(model_path)

def add_vegetation_indices_to_single_patch(input_patch_path, features, output_patch_path):
    """
    Adds specified vegetation indices as new bands to a single TIFF patch and saves the result.
    This function now writes to a new temporary file and replaces the original to avoid permission errors.
    """
    with rasterio.open(input_patch_path) as src:
        bands = [src.read(i) for i in range(1, src.count + 1)]
        new_data = np.array(bands, dtype=np.float32)
        features = config["feature_extraction"]["indices"]

        # Add each requested vegetation index
        for feature in features:
            if 'NDVI' in features:
                ndvi = calculate_ndvi(bands[11].astype(float), bands[8].astype(float))
                new_data = np.concatenate((new_data, np.expand_dims(ndvi, axis=0)), axis=0)

            if 'SAVI' in features:
                savi = calculate_savi(bands[11].astype(float), bands[8].astype(float))
                new_data = np.concatenate((new_data, np.expand_dims(savi, axis=0)), axis=0)
            
            if 'PSRI' in features:
                psri = calculate_psri(bands[8].astype(float), bands[11].astype(float))
                new_data = np.concatenate((new_data, np.expand_dims(psri, axis=0)), axis=0)
                                    
            if 'GNDVI' in features:
                gndvi = calculate_gndvi(bands[11].astype(float), bands[6].astype(float))
                new_data = np.concatenate((new_data, np.expand_dims(gndvi, axis=0)), axis=0)

            if 'NDRE' in features:
                ndre = calculate_ndre(bands[11].astype(float), bands[9].astype(float))
                new_data = np.concatenate((new_data, np.expand_dims(ndre, axis=0)), axis=0)

            if 'VARI' in features:
                vari = calculate_vari(bands[6].astype(float), bands[8].astype(float), bands[3].astype(float))
                new_data = np.concatenate((new_data, np.expand_dims(vari, axis=0)), axis=0)
            
            if 'SR' in features:
                sr = calculate_sr(bands[11].astype(float), bands[8].astype(float))
                new_data = np.concatenate((new_data, np.expand_dims(sr, axis=0)), axis=0)
            
            if 'MCARI' in features:
                mcari = calculate_mcari(bands[10].astype(float), bands[8].astype(float), bands[6].astype(float))
                new_data = np.concatenate((new_data, np.expand_dims(mcari, axis=0)), axis=0)
            
            if 'NDGI' in features:
                ndgi = calculate_ndgi(bands[6].astype(float), bands[8].astype(float))
                new_data = np.concatenate((new_data, np.expand_dims(ndgi, axis=0)), axis=0)
            
            if 'PRI' in features:
                pri = calculate_ndgi(bands[3].astype(float), bands[6].astype(float))
                new_data = np.concatenate((new_data, np.expand_dims(pri, axis=0)), axis=0)

        new_profile = src.profile
        new_profile.update(count=len(new_data))

        fd_out, temp_output_path = tempfile.mkstemp(suffix='.tif')
        os.close(fd_out)  # Close the file descriptor immediately

        with rasterio.open(temp_output_path, 'w', **new_profile) as dst:
            for i, band in enumerate(new_data, start=1):
                dst.write(band, indexes=i)

        # Replace the original file with the new one
        shutil.move(temp_output_path, output_patch_path)

def process_image_to_patches(input_tiff_path, patch_size_meters, model, output_label_path):
    
    with rasterio.open(input_tiff_path) as src:
        
        patch_size_pixels = math.ceil(patch_size_meters / src.res[0])

        output_labels = np.full((src.height // patch_size_pixels, src.width // patch_size_pixels), -1, dtype=float)

        for i in tqdm(range(0, src.height, patch_size_pixels), desc='Row'):
            for j in range(0, src.width, patch_size_pixels):
                dynamic_patch_height = min(patch_size_pixels, src.height - i)
                dynamic_patch_width = min(patch_size_pixels, src.width - j)
                window = Window(j, i, dynamic_patch_width, dynamic_patch_height)

                patch_meta = src.meta.copy()
                patch_meta.update({
                    "driver": "GTiff",
                    "height": dynamic_patch_height,
                    "width": dynamic_patch_width,
                    "transform": rasterio.windows.transform(window, src.transform)
                })

                fd, temp_patch_path = tempfile.mkstemp(suffix='.tif')
                os.close(fd)

                with rasterio.open(temp_patch_path, 'w', **patch_meta) as temp_dst:
                    temp_dst.write(src.read(window=window))

                if config["feature_extraction"]["include_indices"]:
                    add_vegetation_indices_to_single_patch(temp_patch_path, config["feature_extraction"]["indices"], temp_patch_path)
                
                features = extract_features_from_tiff(temp_patch_path)
                print(features)
                if features is not None:
                    patch_class = model.predict(features.reshape(1, -1))
                    output_labels[i // patch_size_pixels, j // patch_size_pixels] = patch_class[0]
                
                os.remove(temp_patch_path)
        
        meta = src.meta.copy()
        meta.update(dtype=rasterio.float32, count=1)
        with rasterio.open(output_label_path, 'w', **meta) as dst:
            dst.write(output_labels.astype(rasterio.int32), 1)

# Configuration and execution
input_tiff_path = r'C:\Users\s.angerer\Privat\Studium\veg_classification\input_data\Ortho_Schrankogel_32632_6cm.tif'
output_label_path = r'C:\Users\s.angerer\Privat\Studium\veg_classification\input_data\Ortho_Schrankogel_32632_6cm.tif'
patch_size_meters = config["patch_creation"]["patch_size"]  # Or another value specified in config

process_image_to_patches(input_tiff_path, patch_size_meters, rf_model, output_label_path)
