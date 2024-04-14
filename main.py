import logging
import log_config
from config import config, save_config_to_file
import create_patches
import train_rf
import veg_map
import add_lidar_data
import os

def main():
    # Setup logging
    log_config.setup_logging(config['classification']['results_folder'])
    logger = logging.getLogger(__name__)
    logger.info("Starting main execution")

    # Save config file
    save_config_to_file(config, config['classification']['results_folder'])

    if config['patch_creation']['extract_patches']:
        logger.info("Starting patch extraction.")
        create_patches.create_image_patches(
            config['data_paths']['ortho_tiff'],
            config['data_paths']['vegetation_gpkg'], 
            config['patch_creation']['patch_folder'], 
            config['patch_creation']['patch_size'], 
            config['patch_creation']['cluster_size'],
            config['patch_creation']['t5_cov_herb_threshold']
        )
        logger.info("Patch extraction completed.")
    
    if config['patch_creation']['merge_clusters']:
        create_patches.merge_folders(
            config['patch_creation']['patch_folder'],
            config['patch_creation']['merge_list'],
            True
        )
    
    patch_folder = config['patch_creation']['patch_folder']

    if config["feature_extraction"]["include_lidar_features"]:
        logger.info("Starting lidar feature calculation")    
        add_lidar_data.process_lidar_with_patches(
            patch_folder,
            config['data_paths']['lidar_tiff_folder'],
            config['patch_creation']['lidar_patch_folder'],
            config['feature_extraction']['specific_lidar_features']
        )

        patch_folder = config['patch_creation']['lidar_patch_folder']
        logger.info("Lidar features calcualted.")

    if config['feature_extraction']['include_indices']:
        
        logger.info("Starting vegetation indices calculation")
        
        create_patches.add_vegetation_indices_bands(
            patch_folder,
            config['feature_extraction']['specific_indices'],
            config['patch_creation']['vi_patch_folder']
        )
        
        patch_folder = config['patch_creation']['vi_patch_folder']
        
        logger.info("Vegetation indices calcualted.")
    
    if config['classification']['enable_rf']:        
        
        train_rf.perform_rf_classification(
            patch_folder,
            config['classification']['results_folder'],
            config['classification']['hyperparameters'], 
            config['classification']['outer_splits'],
            config['classification']['inner_cv_type']
        )

    if config["map_generation"]["generate_maps"]:
        veg_map.create_maps(
            config['map_generation']['maps_folder'],
            os.path.join(config['classification']['results_folder'], 'prediction_maps'),
            os.path.join(config['classification']['results_folder'], 'rf_model_all_features.txt.joblib')
        )

if __name__ == "__main__":
    main()