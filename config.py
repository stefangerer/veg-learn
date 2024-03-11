config = {
    "data_paths": {
        "ortho_tiff": r'C:\Users\s.angerer\Privat\Studium\veg_classification\input_data\Ortho_Schrankogel_32632_6cm.tif',
        "vegetation_gpkg": r'C:\Users\s.angerer\Privat\Studium\veg_classification\input_data\vegetation_groups_EPSG25832.gpkg'
    },
    "patch_creation": {
        "patch_folder": r'C:\Users\s.angerer\Privat\Studium\veg_classification\patches\bands',
        "extract_patches": True,
        "patch_size": 1.00,
        "cluster_size": "cl_good",
        "t5_cov_herb_threshold": 5,
        "merge_clusters": True, 
        "merge_list": [[7, 10], [8, 9], ['no_vegetation', 11]]
    },
    "feature_extraction": {
        "include_bands": True,
        "specific_bands": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "include_indices": True,
        "vi_patch_folder": r'C:\Users\s.angerer\Privat\Studium\veg_classification\patches\bands_vi',
        "indices": ["NDVI", "SAVI", "PSRI", "GNDVI", "NDRE", "VARI", "SR", "MCARI", "NDGI", "PRI"], 
        "include_textures": True,
        "textures": ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation'], #
        "statistics": ['mean', 'max', 'min', 'std', 'percentile_75', 'percentile_25', 'percentile_50'], #
    },
    "classification": {
        "enable_rf": True,
        "results_folder": r"C:\Users\s.angerer\Privat\Studium\veg_classification\results\05_03_24\test_cl_good_t505",
        "include_hyperparameter_tuning": True,
        "hyperparameter_cv_type": "leave_one_out", #stratified or leave_one_out
        "hyperparameters": {
            'n_estimators': [300],
            'max_features': ['log2', 'sqrt'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4], 
        },
        "include_class_balancing": True
    }
}
