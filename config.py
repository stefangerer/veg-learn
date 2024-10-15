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
        "ortho_tiff": r"D:\DATA\Masterthesis Angerer\input_data\ortho_tiff\Ortho_Schrankogel_32632_6cm.tif",
        "vegetation_gpkg": r"D:\DATA\Masterthesis Angerer\input_data\patch_locations\vegetation_groups_900plots_EPSG25832.gpkg",
        "lidar_tiff_folder": r"D:\DATA\Masterthesis Angerer\input_data\lidar_tiffs_resampled"  
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
        "include_textures": True,
        
        # feature name list. the list has to correspond to the features added to the feature vector
        "specific_bands": ['405nm', '430nm', '450nm', '480nm', '510nm', '530nm', '550nm', '570nm', '650nm', '685nm', '710nm', '850nm'],
        "specific_lidar_features": ['ASPECT', 'HEIGHT', 'SLOPE'], # Full List: ["processed_2d_zrange_100cm","processed_2d_zrange_50cm","processed_3d_2d_densityratio_100cm","processed_3d_2d_densityratio_50cm","processed_3d_zrange_100cm","processed_aspect_100cm","processed_aspect_50cm","processed_geom_curvature_100cm","processed_geom_curvature_50cm","processed_linearity_100cm","processed_linearity_50cm","processed_omniariance_100cm","processed_omniariance_50cm","processed_planarity_100cm","processed_planarity_50cm","processed_reflectance_max_100cm","processed_reflectance_max_50cm","processed_reflectance_mean_100cm","processed_reflectance_mean_50cm","processed_reflectance_min_100cm","processed_reflectance_min_50cm","processed_reflectance_n_100cm","processed_reflectance_n_50cm","processed_reflectance_stddev_100cm","processed_reflectance_stddev_50cm","processed_scatter_100cm","processed_scatter_50cm","processed_slope_100cm","processed_slope_50cm","processed_stddev_plane_100cm","processed_stddev_plane_50cm"]
        "specific_indices": ["NDVI", "SAVI", "PSRI", "GNDVI", "NDRE", "VARI", "SR", "MCARI", "NDGI", "PRI"],
        "specific_textures": ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation'], 
        
        # define which statistics should calculated over the patches
        "statistics": ['mean', 'max', 'min', 'std', 'percentile_75', 'percentile_25', 'percentile_50'], 
    },
    "classification": {
        "enable_rf": True,
        "results_folder": r"D:\DATA\Masterthesis Angerer\veg_learn\results\v2_tests\011024\second_test",
        "outer_splits": 5,
        "inner_cv_type": 'stratified', #stratified (uses 10 folds) or leave_one_out
        "hyperparameters": {
            'n_estimators': [100, 200],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [2, 4], 
            'min_samples_split': [2, 5, 10, 15, 20],  
            'min_samples_leaf': [2, 5, 10, 15, 20],  
        },
    },
    "map_generation": {
        "generate_maps": True,
        "maps_folder": r"D:\DATA\Masterthesis Angerer\veg_learn\map_input"
    }
}


