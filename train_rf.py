import os
import numpy as np
import extract_features
import matplotlib.pyplot as plt
import seaborn as sns
import validate_rf
import logging
import joblib
from skimage import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, LeaveOneOut

logger = logging.getLogger(__name__)

def perform_hyperparameter_tuning(X_train, y_train, param_grid, cv_type):
# Define the cross-validator based on the cv_type argument more succinctly
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if cv_type == 'stratified' else LeaveOneOut()
    
    # Initialize GridSearchCV with a more concise approach
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight='balanced', oob_score=True),
        param_grid=param_grid,
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=8,
        verbose=1
    )
    
    # Fit the model and return the results
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_, grid_search.best_estimator_


def perform_nested_cv(X, y, feature_names, param_grid, output_folder, outer_splits, inner_cv_type, name):

    logger.info(f"Perform nested cv for {name}")

    # ---> NESTED CROSS VALIDATION <---

    # Initialize dictionaries to store evaluation metrics for each fold
    train_evaluation_metrics = {metric: [] for metric in ['accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'kappa_score', 'balanced_accuracy']}
    test_evaluation_metrics = {metric: [] for metric in ['accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'kappa_score', 'balanced_accuracy']}

    all_true_labels, all_predictions = [], []

    skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
          
        best_params, best_score, best_estimator = perform_hyperparameter_tuning(X_train_fold, y_train_fold, param_grid, inner_cv_type)
        logger.info(f"Score Fold-{fold}: {best_score}")
        logger.info(f"Parameters Fold-{fold}: {best_params}")
      
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

    # Aggregate metrics across folds
    average_test_metrics, std_test_metrics = validate_rf.aggregate_metrics(test_evaluation_metrics)
    average_train_metrics, std_train_metrics = validate_rf.aggregate_metrics(train_evaluation_metrics)

    # ---> GENERALIZED MODEL <---

    # Find final model      
    final_params, final_score, final_estimator = perform_hyperparameter_tuning(X, y, param_grid, inner_cv_type)
    logger.info(f"Generalized Parameters: {final_params}")
    logger.info(f"Generalized Score: {final_score}")

    # ---> SCORINGS <---

    # Plot learning curve for the model trained with LOOCV
    validate_rf.plot_learning_curve(final_estimator, "Learning Curve", X, y, cv=skf, output_path=os.path.join(output_folder, f'learning_curve_{name}.png'))

    # Save and plot feature importances
    feature_importances = final_estimator.feature_importances_
    validate_rf.plot_feature_importances(feature_importances, feature_names, os.path.join(output_folder, f'feature_importance_{name}.png'))

    # Plotting the aggregated confusion matrix
    validate_rf.plot_confusion_matrix('Confusion Matrix', all_true_labels, all_predictions, output_folder, y, f'confusion_matrix_{name}.png')

    # Save aggregated validation scores
    validate_rf.save_metrics_and_importances("TRAIN SCORES", average_train_metrics, std_train_metrics, file_path=os.path.join(output_folder, f'scores_and_importances_{name}.txt'), mode='w')
    validate_rf.save_metrics_and_importances("TEST SCORES", average_test_metrics, std_test_metrics, feature_names, feature_importances, os.path.join(output_folder, f'scores_and_importances_{name}.txt'), mode='a')

    # Save the final model
    rf_save_path = os.path.join(output_folder, f'rf_model_{name}.txt.joblib')
    joblib.dump(final_estimator, rf_save_path)
    logger.info(f"Trained RF model saved to {rf_save_path}")

    return feature_importances


def perform_rf_classification(data_folder, output_folder, parameter_grid, outer_splits, inner_cv_type):
    
    X, y, _ = extract_features.load_dataset(data_folder)

    feature_names = extract_features.generate_feature_names()

    os.makedirs(output_folder, exist_ok=True)

    feature_importance = perform_nested_cv(X, y, feature_names, parameter_grid, output_folder, outer_splits, inner_cv_type, 'all_features')

    # Select the top 10 features
    top10_indices = np.argsort(feature_importance)[::-1][:10]
    X_top10 = X[:, top10_indices]
    top10_feature_names = [feature_names[i] for i in top10_indices]

    perform_nested_cv(X_top10, y, top10_feature_names, parameter_grid, output_folder, outer_splits, inner_cv_type, 'top_10_features')

   



    



