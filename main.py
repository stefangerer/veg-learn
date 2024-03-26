import logging
import log_config
from config import config, save_config_to_file
import create_patches
import train_rf
import veg_map
import os

def main():
    # Setup logging
    log_config.setup_logging(config['classification']['results_folder'])
    logger = logging.getLogger(__name__)
    
    # Start main execution
    logger.info("Starting main execution")

    # Save config file
    save_config_to_file(config, config['classification']['results_folder'])
    logger.info("Config file saved")

    # Patch Extraction
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
    
    # Cluster Merging
    if config['patch_creation']['merge_clusters']:
        logger.info("Starting cluster merging.")
        create_patches.merge_folders(
            config['patch_creation']['patch_folder'],
            config['patch_creation']['merge_list'],
            True
        )
        logger.info("Cluster merging completed.")
    
    # Vegetation Indices Calculation
    if config['feature_extraction']['include_indices']:
        logger.info("Starting vegetation indices calculation")
        create_patches.add_vegetation_indices_bands(
            config['patch_creation']['patch_folder'],
            config['feature_extraction']['indices'],
            config['feature_extraction']['vi_patch_folder']
        )
        logger.info("Vegetation indices calculation completed.")
    
    # Random Forest Classification
    if config['classification']['enable_rf']:
        logger.info("Starting random forest classification.")
        if config['feature_extraction']['include_indices']:
            data_folder = config['feature_extraction']['vi_patch_folder']
        else: 
            data_folder = config['patch_creation']['patch_folder'], 

        train_rf.perform_rf_classification(
            data_folder,
            config['classification']['results_folder'],
            config['classification']['include_hyperparameter_tuning'],
            config['classification']['cv_type'], 
            config['classification']['hyperparameters'], 
            config['classification']['include_class_balancing']
        )
        logger.info("Random forest classification completed.")
    
    # Vegetation Map Generation
    if config["map_generation"]["generate_map"]:
        logger.info("Starting vegetation map generation.")
        veg_map.create_map(
            config['data_paths']['ortho_tiff'],
            config['map_generation']['map_size'],
            config['map_generation']['map_folder'],
            os.path.join(config['classification']['results_folder'], 'rf_model.joblib')
        )
        logger.info("Vegetation map generation completed.")

if __name__ == "__main__":
    main()
