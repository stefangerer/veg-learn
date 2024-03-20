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
        "extract_patches": False,
        "patch_size": 1.00,
        "cluster_size": "cl_good",
        "t5_cov_herb_threshold": 10,
        "merge_clusters": False, 
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
        "results_folder": r"D:\DATA\Masterthesis Angerer\veg_learn\results\20_03_24\stratified_test_t510_1",
        "include_hyperparameter_tuning": True,
        "cv_type": 'stratified', #stratified or leave_one_out
        "hyperparameters": {
            'n_estimators': [200],
            'max_features': ['log2', 'sqrt'],
            'max_depth': [4, 6, 8, 10],  
            'min_samples_split': [6, 8, 10],  
            'min_samples_leaf': [6, 8, 10],  
        },
        "include_class_balancing": True
    }
}
