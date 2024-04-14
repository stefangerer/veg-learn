import os

def save_config_to_file(config, results_folder):
    config_file_path = os.path.join(results_folder, 'config.txt')
    with open(config_file_path, 'w') as f:
        for section, options in config.items():
            f.write(f"[{section}]\n")
            for key, value in options.items():
                f.write(f"{key} = {value}\n")
            f.write("\n")

config = {
    "data_paths": {
        "ortho_tiff": r'D:\DATA\Schrankogel\Metashape_Project_Schrankogel\Ortho\bigtiff_LZW\Ortho_Schrankogel_32632_6cm.tif',
        "vegetation_gpkg": r'D:\DATA\Masterthesis Angerer\veg_learn\input_data\vegetation_groups_900plots_EPSG25832.gpkg',
        "lidar_tiff_folder": r'D:\DATA\Masterthesis Angerer\input_data\lidar_tiffs'  
    },
    "patch_creation": {
        "patch_folder": r'D:\DATA\Masterthesis Angerer\veg_learn\patches\bands',
        "lidar_patch_folder": r'D:\DATA\Masterthesis Angerer\veg_learn\patches\bands_lidar',
        "vi_patch_folder": r'D:\DATA\Masterthesis Angerer\veg_learn\patches\bands_vi',
        
        "extract_patches":  False,
        "patch_size": 1.00,
        "cluster_size": "cl_good",
        "t5_cov_herb_threshold": 15,

        "merge_clusters":  False,
        "merge_list": [[7, 10], ["no_veg", 9]]
    },
    "feature_extraction": {
        
        # define which features should be added
        "include_lidar_features": True,
        "include_indices": True,
        "include_textures": False,
        
        # feature name list. the list has to correspond to the features added to the feature vector
        "specific_bands": ['405nm', '430nm', '450nm', '480nm', '510nm', '530nm', '550nm', '570nm', '650nm', '685nm', '710nm', '850nm'],
        "specific_lidar_features": ['HEIGHT', 'ASPECT', 'SLOPE'], # 
        "specific_indices": ["NDVI", "SAVI", "PSRI", "GNDVI", "NDRE", "VARI", "SR", "MCARI", "NDGI", "PRI"],
        "specific_textures": ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation'], 
        
        # define which statistics should calculated over the patches
        "statistics": ['mean', 'max', 'min', 'std', 'percentile_75', 'percentile_25', 'percentile_50'], 
    },
    "classification": {
        "enable_rf": False,
        "results_folder": r"D:\DATA\Masterthesis Angerer\veg_learn\results\test\test5",
        "outer_splits": 3,
        "inner_cv_type": 'stratified', #stratified (uses 5 folds) or leave_one_out
        "hyperparameters": {
            'n_estimators': [100],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [2, 4, 6, 8], 
            'min_samples_split': [2, 5, 10, 15],  
            'min_samples_leaf': [2, 5, 10, 15],  
        },
    },
    "map_generation": {
        "generate_maps": False,
        "maps_folder": r"D:\DATA\Masterthesis Angerer\Result_Maps"
    }
}


