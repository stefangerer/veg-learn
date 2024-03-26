import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_curve, roc_auc_score, auc, balanced_accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
import pandas as pd

logger = logging.getLogger(__name__)

def calculate_and_plot_permutation_importance(model, X_test, y_test, feature_names, output_folder='output', n_repeats=10, random_state=42):
    """
    Calculate permutation importances, plot them, and save the plot to a file.
    
    Parameters:
    - model: The trained model.
    - X_test: The test features.
    - y_test: The test labels.
    - feature_names: A list of feature names.
    - output_folder: The folder where to save the output plot.
    - n_repeats: Number of times to permute a feature.
    - random_state: Seed for random number generator.
    """
    
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Calculate permutation importance
    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    
    # Organize importances
    importances_mean = result.importances_mean
    importances_std = result.importances_std
    
    df_importances = pd.DataFrame({
        'features': feature_names,
        'importance_mean': importances_mean,
        'importance_std': importances_std
    }).sort_values(by='importance_mean', ascending=True)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    plt.barh(df_importances['features'], df_importances['importance_mean'], xerr=df_importances['importance_std'], align='center')
    plt.xlabel('Mean decrease in accuracy')
    plt.title('Permutation Importances (test set)')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_folder, 'permutation_importance.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

def plot_learning_curve(estimator, title, X, y, cv, output_path, scoring='balanced_accuracy'):
    
    # Determine the training and test scores
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring=scoring)
    
    # Plotting the learning curve
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    
    # Calculate means and standard deviations
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    
    # Save the plot to the specified output path
    plt.savefig(output_path)
    plt.close()  # Close the figure to free memory
    
def log_and_print_scores(header, metrics, std_metrics):
    print(f"--- {header} ---")
    logger.info(f"--- {header} ---")
    for metric, avg_score in metrics.items():
        print(f"{metric} | AVG: {avg_score:.3f}:.3f | STD: {std_metrics[metric]:.3f}")
        logger.info(f"{metric} | AVG: {avg_score:.3f}:.3f | STD: {std_metrics[metric]:.3f}")

def plot_confusion_matrix(title, all_true_labels, all_predictions, output_folder, labels, matrix_name):
    cm = confusion_matrix(all_true_labels, all_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.title(title)
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.savefig(os.path.join(output_folder, matrix_name))
    plt.close()

def calculate_metrics(y_true, y_pred, average='macro'):
    metrics = {
        'accuracy_score': accuracy_score(y_true, y_pred),
        'precision_score': precision_score(y_true, y_pred, average=average),
        'recall_score': recall_score(y_true, y_pred, average=average),
        'f1_score': f1_score(y_true, y_pred, average=average),
        'kappa_score': cohen_kappa_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
    }
    return metrics

def aggregate_metrics(metrics_list):
    average_metrics = {metric: np.mean(scores) for metric, scores in metrics_list.items()}
    std_metrics = {metric: np.std(scores) for metric, scores in metrics_list.items()}
    return average_metrics, std_metrics

def plot_feature_importances(importances, feature_names, output_file):

    #logger.info(f"--- FEATURE IMPORTANCE ---")
    #for feature, importance in zip(feature_names, importances):
    #    logger.info(f"{feature}: {importance}")

    # Sort the feature importances in descending order and get the indices
    indices = np.argsort(importances)[::-1]

    # Rearrange the feature names to match the sorted feature importances
    sorted_names = [feature_names[i] for i in indices]

    # Create a bar plot of the feature importances
    plt.figure(figsize=(10, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), sorted_names, rotation=90)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def save_metrics_and_importances(header, metrics, std_metrics, feature_names=None, importances=None, file_path="/mnt/data/report.txt", mode='a'):
    """
    Logs metrics, standard deviations, and optionally feature importances to a specified file.
    Supports both overwriting and appending data to the file.

    Parameters:
    - header: A header or title for the metrics section.
    - metrics: Dictionary of metric names to average scores.
    - std_metrics: Dictionary of metric names to their standard deviations.
    - feature_names: Optional list of feature names. If None, feature importances are not logged.
    - importances: Optional list of importance scores corresponding to the feature names.
    - file_path: Path to the file where the data will be saved.
    - mode: 'w' for overwrite or 'a' for append. Defaults to 'a' for appending.
    """
    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(file_path, mode) as file:
        # Write header and metrics
        file.write(f"\n--- {header} ---\n")
        for metric, avg_score in metrics.items():
            file.write(f"{metric} | AVG: {avg_score:.3f} | STD: {std_metrics[metric]:.3f}\n")
        
        # Optionally write feature importances
        if feature_names is not None and importances is not None:
            file.write("\n--- FEATURE IMPORTANCE ---\n")
            features_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            for rank, (feature, importance) in enumerate(features_importances, start=1):
                file.write(f"Rank {rank}: {feature} - {importance:.3f}\n")

def save_metrics_and_feature_importances_in_order(header, metrics, std_metrics, feature_names=None, importances=None, file_path="/mnt/data/report_ordered_importances.txt", mode='a'):
    """
    Logs metrics, standard deviations, and feature importances to a specified file in their original order.
    Supports both overwriting and appending data to the file.

    Parameters:
    - header: A header or title for the metrics section.
    - metrics: Dictionary of metric names to average scores.
    - std_metrics: Dictionary of metric names to their standard deviations.
    - feature_names: Optional list of feature names. If None, feature importances are not logged.
    - importances: Optional list of importance scores corresponding to the feature names, in their original order.
    - file_path: Path to the file where the data will be saved.
    - mode: 'w' for overwrite or 'a' for append. Defaults to 'a' for appending.
    """
    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(file_path, mode) as file:
        # Write header and metrics
        file.write(f"\n--- {header} ---\n")
        for metric, avg_score in metrics.items():
            file.write(f"{metric} | AVG: {avg_score:.3f} | STD: {std_metrics[metric]:.3f}\n")
        
        # Optionally write feature importances in their original order
        if feature_names is not None and importances is not None:
            file.write("\n--- FEATURE IMPORTANCE (Original Order) ---\n")
            for index, (feature, importance) in enumerate(zip(feature_names, importances)):
                file.write(f"{feature} - {importance:.3f}\n")
