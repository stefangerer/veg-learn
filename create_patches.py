import os
import math
import shutil
import geopandas as gpd
import rasterio
import calculate_indices
import numpy as np
from tqdm import tqdm

def create_image_patches(tiff_path, gpkg_path, output_folder, buffer_distance_m, cluster_size, t5_cov_herb_threshold):
        
    # Read the TIFF file to get its CRS
    with rasterio.open(tiff_path) as src:
        tiff_crs = src.crs

    # Read points from the GPKG file
    gdf = gpd.read_file(gpkg_path)

    # Check and report if CRS differs, then reproject
    if gdf.crs != tiff_crs:
        gdf = gdf.to_crs(tiff_crs)

    with rasterio.open(tiff_path) as src:
        # Calculate buffer size in pixels
        buffer_size_pixels = math.ceil(buffer_distance_m / (src.res[0]))

        # Initialize tqdm progress bar
        total_features = len(gdf)
        pbar = tqdm(total=total_features, desc="Processing", unit="image_patch")

        for idx, row in gdf.iterrows():
            point = row['geometry']
            # Extract the feature ID
            feature_id = row['logger_ID']

            # Extract the vegetation cluster ID
            cluster_id = row[cluster_size] 

            # Extract T5_cov_herb value
            t5_cov_herb_value = row['t5_cov_herb']

            # Determine the folder based on T5_cov_herb value
            if t5_cov_herb_value < t5_cov_herb_threshold:
                cluster_folder = os.path.join(output_folder, 'no_vegetation')
            else:
                # Extract the vegetation cluster ID
                cluster_id = row[cluster_size]
                cluster_folder = os.path.join(output_folder, f'{cluster_id}')

            # Ensure the cluster folder exists
            if not os.path.exists(cluster_folder):
                os.makedirs(cluster_folder)

            # Convert the point to pixel coordinates
            point_pixel_coords = src.index(*point.coords[0])

            # Create a square window around the point
            window = rasterio.windows.Window(
                col_off=point_pixel_coords[1] - buffer_size_pixels,
                row_off=point_pixel_coords[0] - buffer_size_pixels,
                width=2 * buffer_size_pixels,
                height=2 * buffer_size_pixels
            )

            # Read the data in the window
            data = src.read(window=window)

            tile_path = os.path.join(cluster_folder, f'tile_{feature_id}.tif')
            with rasterio.open(
                tile_path,
                'w',
                driver='GTiff',
                height=window.height,
                width=window.width,
                count=src.count,
                dtype=src.dtypes[0],
                crs=src.crs,
                transform=rasterio.windows.transform(window, src.transform)
            ) as dst:
                dst.write(data)

            # Update progress bar
            pbar.update(1)

        # Close the progress bar after processing is finished
        pbar.close()
        
def add_vegetation_indices_bands(folder_path, features, output_folder_path):

    # Check if the output folder already exists. If so, delete and recreate it.
    if os.path.exists(output_folder_path):
        shutil.rmtree(output_folder_path)
    os.makedirs(output_folder_path, exist_ok=True)

    total_files = sum(len(files) for _, _, files in os.walk(folder_path))

    with tqdm(total=total_files, desc="Processing", unit="image_patch") as pbar:
        for subdir, _, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith(".tif"):
                    tiff_path = os.path.join(subdir, filename)
                    with rasterio.open(tiff_path) as src:
                        bands = [src.read(i) for i in range(1, src.count + 1)]
                        new_data = np.array(bands, dtype=np.float32)

                        if 'NDVI' in features:
                            ndvi = calculate_indices.calculate_ndvi(bands[11].astype(float), bands[8].astype(float))
                            new_data = np.concatenate((new_data, np.expand_dims(ndvi, axis=0)), axis=0)

                        if 'SAVI' in features:
                            savi = calculate_indices.calculate_savi(bands[11].astype(float), bands[8].astype(float))
                            new_data = np.concatenate((new_data, np.expand_dims(savi, axis=0)), axis=0)
                        
                        if 'PSRI' in features:
                            psri = calculate_indices.calculate_psri(bands[8].astype(float), bands[11].astype(float))
                            new_data = np.concatenate((new_data, np.expand_dims(psri, axis=0)), axis=0)
                                                
                        if 'GNDVI' in features:
                            gndvi = calculate_indices.calculate_gndvi(bands[11].astype(float), bands[6].astype(float))
                            new_data = np.concatenate((new_data, np.expand_dims(gndvi, axis=0)), axis=0)

                        if 'NDRE' in features:
                            ndre = calculate_indices.calculate_ndre(bands[11].astype(float), bands[9].astype(float))
                            new_data = np.concatenate((new_data, np.expand_dims(ndre, axis=0)), axis=0)

                        if 'VARI' in features:
                            vari = calculate_indices.calculate_vari(bands[6].astype(float), bands[8].astype(float), bands[3].astype(float))
                            new_data = np.concatenate((new_data, np.expand_dims(vari, axis=0)), axis=0)
                        
                        if 'SR' in features:
                            sr = calculate_indices.calculate_sr(bands[11].astype(float), bands[8].astype(float))
                            new_data = np.concatenate((new_data, np.expand_dims(sr, axis=0)), axis=0)
                        
                        if 'MCARI' in features:
                            mcari = calculate_indices.calculate_mcari(bands[10].astype(float), bands[8].astype(float), bands[6].astype(float))
                            new_data = np.concatenate((new_data, np.expand_dims(mcari, axis=0)), axis=0)
                        
                        if 'NDGI' in features:
                            ndgi = calculate_indices.calculate_ndgi(bands[6].astype(float), bands[8].astype(float))
                            new_data = np.concatenate((new_data, np.expand_dims(ndgi, axis=0)), axis=0)
                        
                        if 'PRI' in features:
                            pri = calculate_indices.calculate_ndgi(bands[3].astype(float), bands[6].astype(float))
                            new_data = np.concatenate((new_data, np.expand_dims(pri, axis=0)), axis=0)


                        new_profile = src.profile
                        new_profile.update(count=new_profile['count'] + len(features), dtype='float32')

                        # Create a corresponding output subdirectory within the output folder
                        output_subdir = subdir.replace(folder_path, output_folder_path, 1)
                        os.makedirs(output_subdir, exist_ok=True)

                        new_tiff_path = os.path.join(output_subdir, f"features_{filename}")
                        with rasterio.open(new_tiff_path, 'w', **new_profile) as dst:
                            for i, band in enumerate(new_data, start=1):
                                dst.write(band, indexes=i)

                    
                    pbar.update(1)