config = {
    "data_paths": {
        "ortho_tiff": r'C:\Users\s.angerer\Privat\Studium\Masterarbeit\Daten\Ortho_Schrankogel_32632_6cm.tif',
        "vegetation_gpkg": r'C:\Users\s.angerer\Privat\Studium\Masterarbeit\Daten\vegetation_groups_EPSG25832.gpkg'
    },
    "patch_creation": {
        "patch_folder": r'C:\Users\s.angerer\Privat\Studium\Masterarbeit\Daten\patches',
        "extract_patches": "True",
        "patch_size": 1.00,
        "cluster_size": "cl_good",
    },
    "feature_extraction": {
        "include_bands": True,
        "specific_bands": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "include_indices": True,
        "vi_patch_folder": r'C:\Users\s.angerer\Privat\Studium\Masterarbeit\Daten\patches_vi',
        "indices": ["NDVI", "SAVI", "PSRI", "GNDVI", "NDRE", "VARI", "SR", "MCARI", "NDGI", "PRI"], 
        "include_textures": True,
        "textures": ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation'], #
        "statistics": ['mean', 'max', 'min', 'std', 'percentile_75', 'percentile_25', 'percentile_50'], #
    },
    "classification": {
        "enable_rf": True,
        "results_folder": r"C:\Users\s.angerer\Privat\Studium\Masterarbeit\Daten\res_test_2",
        "include_hyperparameter_tuning": True,
        "hyperparameters": {
            'n_estimators': [100, 200, 300],
            'max_features': ['log2', 'sqrt'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4], 
        },
        "include_class_balancing": True
    }
}
