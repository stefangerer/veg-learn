import os
import rasterio
from rasterio.windows import Window
import config
import numpy as np 
import create_patches



def extract_center_square_meters(input_tif_path, output_tif_path, square_size_meters=100):
    with rasterio.open(input_tif_path) as src:
        # Extract pixel dimensions in meters from the transform (affine) of the raster
        pixel_width_meters = src.transform[0]  # Pixel width in meters
        pixel_height_meters = abs(src.transform[4])  # Pixel height in meters, which should be positive
        
        # Calculate the number of pixels for the specified square size in meters
        pixels_across_width = int(square_size_meters / pixel_width_meters)
        pixels_across_height = int(square_size_meters / pixel_height_meters)
        
        # Calculate the window position (centered)
        center_x, center_y = src.width // 2, src.height // 2
        offset_x = pixels_across_width // 2
        offset_y = pixels_across_height // 2
        
        window = Window(center_x - offset_x, center_y - offset_y, pixels_across_width, pixels_across_height)
        
        # Read the data from the window
        data = src.read(window=window)
        
        # Define the new transform for the square based on the window
        new_transform = src.window_transform(window)
        
        # Update the metadata for the output file
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": pixels_across_height,
            "width": pixels_across_width,
            "transform": new_transform
        })
        
        # Save the extracted square as a new TIFF
        with rasterio.open(output_tif_path, 'w', **out_meta) as dst:
            dst.write(data)

def create_all_patches(input_tif, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read patch size in meters from config and double it
    patch_size_meters = config.config['patch_creation']['patch_size'] * 2
    
    with rasterio.open(input_tif) as src:
        # Convert patch size from meters to pixels
        patch_size_pixels = int(patch_size_meters / abs(src.transform[0]))
        
        for x in range(0, src.width, patch_size_pixels):
            for y in range(0, src.height, patch_size_pixels):
                if x + patch_size_pixels > src.width or y + patch_size_pixels > src.height:
                    continue  # Skip patches that extend beyond the image boundary
                
                window = Window(x, y, patch_size_pixels, patch_size_pixels)
                patch_data = src.read(window=window)
                
                # Check if the patch contains only valid pixels
                # Adjust this condition based on your definition of a valid pixel value
                # Here, we're assuming that a valid pixel is non-zero and not NaN
                if np.all(patch_data == 255, axis=0).any():
                    continue
                
                # Save the patch
                patch_path = os.path.join(output_folder, f"patch_{x}_{y}.tif")
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



big_tif = r"C:\Users\s.angerer\Privat\Studium\veg_classification\input_data\Ortho_Schrankogel_32632_6cm.tif"
small_tif = r"C:\Users\s.angerer\Privat\Studium\veg_classification\input_data\small_Ortho_Schrankogel_32632_6cm.tif"
small_tif_veg = r"C:\Users\s.angerer\Privat\Studium\veg_classification\input_data\small_Ortho_Schrankogel_32632_6cm_veg_indices.tif"
output_folder = r"C:\Users\s.angerer\Privat\Studium\veg_classification\big_patches"
output_folder_veg = r"C:\Users\s.angerer\Privat\Studium\veg_classification\big_patches_veg"

#extract_center_square_meters(big_tif, small_tif)
#create_all_patches(small_tif, output_folder)
create_patches.add_vegetation_indices_bands(output_folder, config.config["feature_extraction"]["indices"], output_folder_veg)



for file in os.listdir(output_folder_veg):



