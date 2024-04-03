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
        "vegetation_gpkg": r'D:\DATA\Masterthesis Angerer\veg_learn\input_data\vegetation_groups_900plots_EPSG25832.gpkg'
    },
    "patch_creation": {
        "patch_folder": r'D:\DATA\Masterthesis Angerer\veg_learn\patches\bands',
        "extract_patches":  False,
        "patch_size": 1.00,
        "cluster_size": "cl_good",
        "t5_cov_herb_threshold": 15,
        "merge_clusters":  False,
        "merge_list": [[7, 10], ["no_veg", 9]]
    },
    "feature_extraction": {
        "include_bands": True,
        "specific_bands": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "include_indices": True,
        "vi_patch_folder": r'D:\DATA\Masterthesis Angerer\veg_learn\patches\bands_vi',
        "indices": ["NDVI", "SAVI", "PSRI", "GNDVI", "NDRE", "VARI", "SR", "MCARI", "NDGI", "PRI"], 
        "include_textures": True,
        "textures": ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation'], #
        "statistics": ['mean', 'max', 'min', 'std', 'percentile_75', 'percentile_25', 'percentile_50'], #
    },
    "classification": {
        "enable_rf": True,
        "results_folder": r"D:\DATA\Masterthesis Angerer\veg_learn\results\rf_multispectral_final\multispectral_t515_merge_stratified_10fold",
        "outer_splits": 10,
        "inner_cv_type": 'stratified', #stratified (uses 5 folds) or leave_one_out
        "hyperparameters": {
            'n_estimators': [100, 200],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [2, 5, 10],  
            'min_samples_split': [2, 5, 10, 15, 30],  
            'min_samples_leaf': [2, 5, 10, 15, 30],  
        },
    },
        "map_generation": {
        "generate_maps": False,
        "maps_folder": r"D:\DATA\Masterthesis Angerer\map_test"
    }
}
