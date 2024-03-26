import os
import numpy as np
import extract_features
import matplotlib.pyplot as plt
import seaborn as sns
import validate_rf
import logging
import joblib
from sklearn.utils.class_weight import compute_class_weight
from skimage import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, LeaveOneOut
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_curve, roc_auc_score, auc, balanced_accuracy_score

logger = logging.getLogger(__name__)

def perform_hyperparameter_tuning(X_train, y_train, param_grid, class_weights_fold, cv_type):
    """
    Perform hyperparameter tuning for Random Forest.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        param_grid (dict): Grid of hyperparameters to search.
        class_weights_fold (str): Class weights for balancing.
        cv_type (str): Type of cross-validation, either stratified (3-folds) or leave_one_out cross validation.

    Returns:
        tuple: Best parameters, best score, and best estimator.
    """
    if cv_type == 'stratified':
        # Initialize the GridSearchCV object
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42, class_weight=class_weights_fold, oob_score=True),
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1
        )

    elif cv_type == 'leave_on_out':
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42, class_weight=class_weights_fold, oob_score=True),
            param_grid=param_grid,
            cv=LeaveOneOut(), 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
    
    else:
        raise ValueError("Invalid value for cv_type. Choose 'stratified' or 'leave_one_out'.")

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Return the best parameters, score, and estimator
    return grid_search.best_params_, grid_search.best_score_, grid_search.best_estimator_

def perform_rf_classification(data_folder, output_folder, use_hyperparameter_tuning, cv_type, parameter_grid, use_class_balancing):
    """
    Perform Random Forest classification.

    Args:
        data_folder (str): Folder containing the dataset.
        output_folder (str): Folder to save the output.
        use_hyperparameter_tuning (bool): Whether to perform hyperparameter tuning.
        cv_type (str): Type of cross-validation.
        parameter_grid (dict): Grid of hyperparameters.
        use_class_balancing (bool): Whether to use class balancing.
    """
    
    # Load datasets
    X, y, num_bands = extract_features.load_dataset(data_folder)

    # Generate feature names
    feature_names = extract_features.generate_feature_names()

    os.makedirs(output_folder, exist_ok=True)
    fold_output_folder = os.path.join(output_folder, "fold_data")
    os.makedirs(fold_output_folder, exist_ok=True)

    # Initialize dictionaries to store evaluation metrics for each fold
    train_evaluation_metrics = {metric: [] for metric in ['accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'kappa_score', 'balanced_accuracy']}
    test_evaluation_metrics = {metric: [] for metric in ['accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'kappa_score', 'balanced_accuracy']}

    best_models = []
    all_true_labels, all_predictions = [], []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        class_weights_fold = 'balanced' if use_class_balancing else None
        
        if use_hyperparameter_tuning:
            logger.info(f"Hyperparameter-tuning for fold-{fold}")
            best_params, best_score, best_estimator = perform_hyperparameter_tuning(X_train_fold, y_train_fold, parameter_grid, class_weights_fold, cv_type)
            logger.info(f"Score: {best_score}")
            logger.info(f"Parameters: {best_params}")
        else:
            best_estimator = RandomForestClassifier(n_estimators=100, 
                                                    max_depth=8, 
                                                    max_features='log2', 
                                                    min_samples_leaf=2, 
                                                    min_samples_split=8, 
                                                    random_state=42, 
                                                    class_weight=class_weights_fold, 
                                                    oob_score=True)
            best_estimator.fit(X_train_fold, y_train_fold)
        
        best_models.append(best_estimator)

        # Calculate and store test metrics for the current fold
        y_pred_test = best_estimator.predict(X_test_fold)
        test_metrics_fold = validate_rf.calculate_metrics(y_test_fold, y_pred_test)
        for metric, value in test_metrics_fold.items():
            test_evaluation_metrics[metric].append(value)

        # Calculate and store training metrics for the current fold
        y_pred_train = best_estimator.predict(X_train_fold)
        train_metrics_fold = validate_rf.calculate_metrics(y_train_fold, y_pred_train)
        for metric, value in train_metrics_fold.items():
            train_evaluation_metrics[metric].append(value)

        # Collect true lables and predictions
        all_true_labels.extend(y_test_fold.tolist())
        all_predictions.extend(y_pred_test.tolist())

        # Plot Confusion Matrix
        # validate_rf.plot_confusion_matrix(f'Confusion Matrix - Fold {fold}', y_test_fold, y_pred_test, fold_output_folder, y, f'confusion_matrix_fold_{fold}.png')

    # Aggregate metrics across folds
    average_test_metrics, std_test_metrics = validate_rf.aggregate_metrics(test_evaluation_metrics)
    average_train_metrics, std_train_metrics = validate_rf.aggregate_metrics(train_evaluation_metrics)

    # Selecting the best model based on aggregated metrics (example shown for balanced accuracy)
    best_model_index = np.argmax(test_evaluation_metrics['balanced_accuracy'])
    best_model = best_models[best_model_index]

    # Save model for later usage
    rf_save_path = os.path.join(output_folder, 'rf_model.joblib')
    joblib.dump(best_model, rf_save_path)

    # Plotting feature importances
    feature_importances = best_model.feature_importances_
    validate_rf.plot_feature_importances(feature_importances, feature_names, os.path.join(output_folder, 'feature_importance.png'))

    # Plotting learning curve
    validate_rf.plot_learning_curve(best_model, "Learning Curve (Random Forest)", X, y, skf, os.path.join(output_folder, 'learning_curve.png'))

    # Plotting aggregated confusion matrix
    validate_rf.plot_confusion_matrix('Confusion Matrix - Aggregated Over All Folds', all_true_labels, all_predictions, output_folder, y, 'confusion_matrix_aggregated.png')

    # Save aggregated test and training scores 
    validate_rf.save_metrics_and_importances("TRAIN SCORES", average_train_metrics, std_train_metrics, file_path=os.path.join(output_folder, 'scores_and_importances.txt'), mode='w')
    validate_rf.save_metrics_and_importances("TEST SCORES", average_test_metrics, std_test_metrics, feature_names, feature_importances, os.path.join(output_folder, 'scores_and_importances.txt'), mode='a')
    #validate_rf.save_metrics_and_feature_importances_in_order("TEST SCORES", average_test_metrics, std_test_metrics, feature_names, feature_importances, os.path.join(output_folder, 'scores_and_importances_in_order.txt'), mode='w')
