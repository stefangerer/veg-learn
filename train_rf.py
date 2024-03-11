import os
import numpy as np
import extract_features
import matplotlib.pyplot as plt
import seaborn as sns
import validate_rf
import logging
from sklearn.utils.class_weight import compute_class_weight
from skimage import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, LeaveOneOut
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_curve, roc_auc_score, auc, balanced_accuracy_score

logger = logging.getLogger(__name__)

def perform_hyperparameter_tuning(X_train, y_train, param_grid, class_weights_fold, cv_type='stratified'):
    
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

    elif cv_type == "leave_on_out":
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

def perform_rf_classification(data_folder, output_folder, use_hyperparameter_tuning, hyperparameter_cv_type, parameter_grid, use_class_balancing):
    
    X, y, num_bands = extract_features.load_dataset(data_folder)

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
            best_params, best_score, best_estimator = perform_hyperparameter_tuning(X_train_fold, y_train_fold, parameter_grid, class_weights_fold, cv_type=hyperparameter_cv_type)
            logger.info(f"Score Fold: {fold}: {best_score}")
            logger.info(f"Parameters Fold: {fold}: {best_params}")
        else:
            best_estimator = RandomForestClassifier(n_estimators=300, 
                                                    max_depth=10, 
                                                    max_features='sqrt', 
                                                    min_samples_leaf=4, 
                                                    min_samples_split=10, 
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
        validate_rf.plot_confusion_matrix(f'Confusion Matrix - Fold {fold}', y_test_fold, y_pred_test, fold_output_folder, y, f'confusion_matrix_fold_{fold}.png')

        #validate_rf.calculate_and_plot_permutation_importance(best_estimator, X_test_fold, y_test_fold, feature_names, os.path.join(fold_output_folder, f'permutation_importances_{fold}.png'))

    # Aggregate metrics across folds
    average_test_metrics, std_test_metrics = validate_rf.aggregate_metrics(test_evaluation_metrics)
    average_train_metrics, std_train_metrics = validate_rf.aggregate_metrics(train_evaluation_metrics)

    # Selecting the best model based on aggregated metrics (example shown for balanced accuracy)
    best_model_index = np.argmax(test_evaluation_metrics['balanced_accuracy'])
    best_model = best_models[best_model_index]

    feature_importances = best_model.feature_importances_

    validate_rf.plot_feature_importances(feature_importances, feature_names, os.path.join(output_folder, 'feature_importance.png'))

    validate_rf.plot_learning_curve(best_model, "Learning Curve (Random Forest)", X, y, skf, os.path.join(output_folder, 'learning_curve.png'))

    # Plotting the aggregated confusion matrix
    validate_rf.plot_confusion_matrix('Confusion Matrix - Aggregated Over All Folds', all_true_labels, all_predictions, output_folder, y, 'confusion_matrix_aggregated.png')

    validate_rf.save_metrics_and_importances("TRAIN SCORES", average_train_metrics, std_train_metrics, file_path=os.path.join(output_folder, 'scores_and_importances.txt'), mode='w')

    validate_rf.save_metrics_and_importances("TEST SCORES", average_test_metrics, std_test_metrics, feature_names, feature_importances, os.path.join(output_folder, 'scores_and_importances.txt'), mode='a')

    validate_rf.save_metrics_and_feature_importances_in_order("TEST SCORES", average_test_metrics, std_test_metrics, feature_names, feature_importances, os.path.join(output_folder, 'scores_and_importances_in_order.txt'), mode='w')



