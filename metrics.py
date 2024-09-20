from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def evaluate_model(labels, preds, class_names, num_classes, run_index=None):
    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Calculate overall accuracy
    overall_accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    
    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(labels, preds)
    
    # Compute macro-averaged F1 score
    macro_f1 = f1_score(labels, preds, average='macro')

    return overall_accuracy, balanced_acc, macro_f1


# Function to plot F1 score, Precision, and Recall per class
def plot_f1_score_precision_recall(labels, preds, class_names, run_index=None):
    """
    Plots the F1 score, Precision, and Recall per class, sorted by F1-score, and displays the highest, lowest, and mean values in a summary table.
    """
    # Calculate precision, recall, F1 score, and support for present classes
    precision, recall, f1_scores, support = precision_recall_fscore_support(labels, preds, average=None)

    # Sort indices by F1-score
    sorted_indices = np.argsort(f1_scores)[::-1]

    # Sort class names and metrics by F1-score
    sorted_class_names = [class_names[i] for i in sorted_indices]
    sorted_precision = precision[sorted_indices]
    sorted_recall = recall[sorted_indices]
    sorted_f1_scores = f1_scores[sorted_indices]
    sorted_support = support[sorted_indices]
    
    # Plot precision, recall, and F1 scores per class
    x = np.arange(len(sorted_class_names))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, sorted_precision, width, label='Precision', color='lightblue')
    plt.bar(x, sorted_recall, width, label='Recall', color='lightgreen')
    plt.bar(x + width, sorted_f1_scores, width, label='F1 Score', color='lightcoral')

    plt.xlabel('Classes (sorted by F1-score)')
    plt.ylabel('Scores')
    plt.title('Precision, Recall, and F1 Score per Class')
    plt.xticks(x, sorted_class_names, rotation=90)
    plt.legend(loc='best')
    
    if run_index == None:
        plt.savefig('sorted_precision_recall_f1_scores.png')
    else:
        plt.savefig(f'sorted_precision_recall_f1_scores{run_index}.png')

    plt.close()

    # Create a summary table
    summary_data = {
        'Statistic': ['Highest F1 Score', 'Lowest F1 Score', 'Mean F1 Score'],
        'Value': [max(sorted_f1_scores), min(sorted_f1_scores), np.mean(sorted_f1_scores)]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Plot the summary table
    plt.figure(figsize=(5, 1.5))  # Adjust the figure size as needed
    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    if run_index == None:
        plt.savefig('f1_score_summary_table.png')
    else:
        plt.savefig(f'f1_score_summary_table{run_index}.png')

    plt.close()



def plot_confusion_matrix_best_worst(labels, preds, class_names, top_n=3, run_index=None):
    """
    Plots the confusion matrix for the top N best and worst classes based on F1-score and includes support information.
    """
    # Generate the confusion matrix
    cm = confusion_matrix(labels, preds)

    # Calculate F1-Score for each class
    precision, recall, f1_scores, support = precision_recall_fscore_support(labels, preds, average=None)

    # Identify the best and worst classes based on F1-Score
    best_classes = np.argsort(f1_scores)[-top_n:]  # Top n highest F1-scores
    worst_classes = np.argsort(f1_scores)[:top_n]  # Top n lowest F1-scores

    # Combine the indices of best and worst classes
    selected_classes = np.union1d(best_classes, worst_classes)

    # Filter the confusion matrix for the selected classes
    cm_selected = cm[selected_classes, :][:, selected_classes]
    selected_class_names = [class_names[i] for i in selected_classes]
    selected_support = support[selected_classes]

    # Normalize the confusion matrix
    cm_normalized = cm_selected.astype('float') / cm_selected.sum(axis=1)[:, np.newaxis]

    # Annotate the support values in the class labels
    support_labels = [f'{name}\n(n={sup})' for name, sup in zip(selected_class_names, selected_support)]

    # Plot the confusion matrix
    plt.figure(figsize=(15, 10))
    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                     xticklabels=support_labels, yticklabels=support_labels)

    # Set the colorbar label
    colorbar_label = 'Probabilities'
    cbar = ax.collections[0].colorbar
    cbar.set_label(colorbar_label, rotation=270, labelpad=20)

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix Top/Worst {top_n} Classes)')
    plt.tight_layout()

    if run_index == None:
        plt.savefig('conf_matrix_best_worst.png')
    else:
        plt.savefig(f'conf_matrix_best_worst{run_index}.png')

    plt.close()




def plot_f1_score(labels, preds, class_names, run_index=None):
    """
    Plots the F1 score per class and displays the highest, lowest, and mean F1 scores in a separate table.

    Parameters:
    - labels (array-like): True labels.
    - preds (array-like): Predicted labels.
    - class_names (list): List of class names corresponding to the labels.
    """
    # Get the classes present in the dataset
    present_classes = sorted(set(labels) | set(preds))
    present_class_names = [class_names[i - 1] for i in present_classes]  # Adjust class names for 1-based index

    # Calculate the F1 score for present classes
    f1_scores = f1_score(labels, preds, labels=present_classes, average=None)
    f1 = f1_score(labels, preds, labels=present_classes, average='macro')
    
    # Calculate summary statistics
    highest_f1 = max(f1_scores)
    lowest_f1 = min(f1_scores)
    mean_f1 = sum(f1_scores) / len(f1_scores)

    # Plot the F1 score per class
    plt.figure(figsize=(12, 6))
    plt.bar(present_class_names, f1_scores, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.title('F1 Score per Class')
    plt.xticks(rotation=90)
    
    if run_index==None:
        plt.savefig('f1_score.png')
    else:
        plt.savefig(f'f1_score{run_index}.png') 

    plt.close()

    # Create a summary table
    summary_data = {
        'Statistic': ['Highest F1 Score', 'Lowest F1 Score', 'Mean F1 Score'],
        'Value': [highest_f1, lowest_f1, mean_f1]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Plot the summary table
    plt.figure(figsize=(5, 1.5))  # Adjust the figure size as needed
    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    if run_index==None:
        plt.savefig('f1_score_summary_table.png')
    else:
        plt.savefig('ff1_score_summary_table{run_index}.png')

    plt.close()
    return f1



def plot_micro_averaged_roc(labels, preds, num_classes, run_index=None):
    # Binarize the labels for multi-class ROC
    labels_bin = label_binarize(labels, classes=list(range(num_classes)))
    preds_bin = label_binarize(preds, classes=list(range(num_classes)))

    # Compute the micro-average ROC curve and AUC
    fpr, tpr, _ = roc_curve(labels_bin.ravel(), preds_bin.ravel())
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Micro-Averaged ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-Averaged Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    if run_index == None:
      plt.savefig('micro_avg_roc_curve.png')
    else:
      plt.savefig(f'micro_avg_roc_curve{run_index}.png')

    plt.close()



def plot_confusion_matrix(y_true, y_pred, part_names, title='Confusion Matrix'):
    """
    Plots a confusion matrix using true and predicted labels.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - part_names (list): List of car part names corresponding to the labels.
    - normalize (bool): Whether to normalize the confusion matrix by row (i.e., by true labels).
    - title (str): Title for the confusion matrix plot.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plotting the confusion matrix
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=part_names, yticklabels=part_names)


    # Set the colorbar label
    colorbar_label = 'Probabilities'
    cbar = ax.collections[0].colorbar
    cbar.set_label(colorbar_label, rotation=270, labelpad=20)

    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('conf_matrix.png')
    plt.close()

####################Car model###################
"""def overall_accuracy(predicted_makes, predicted_models, true_makes, true_models):
    correct_make = (predicted_makes == true_makes).sum().item()
    correct_model = (predicted_models == true_models).sum().item()
    total = len(true_makes)

    accuracy_make = correct_make / total
    accuracy_model = correct_model / total

    return accuracy_make, accuracy_model


def per_make_model_accuracy(predicted_makes, predicted_models, true_makes, true_models):
    accuracy_dict = {}
    for make in set(true_makes.cpu().numpy()):
        make_indices = (true_makes == make)
        correct_predictions = (predicted_models[make_indices] == true_models[make_indices]).sum().item()
        total_predictions = make_indices.sum().item()
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        accuracy_dict[make] = accuracy
    return accuracy_dict



def confusion_matrix_per_make(predicted_makes, predicted_models, true_makes, true_models):
    confusion_matrices = {}
    for make in set(true_makes.cpu().numpy()):
        make_indices = (true_makes == make)
        cm = confusion_matrix(true_models[make_indices].cpu().numpy(), 
                              predicted_models[make_indices].cpu().numpy(), 
                              labels=list(range(num_models_per_make)))
        confusion_matrices[make] = cm
    return confusion_matrices



def precision_recall_f1_per_make(predicted_makes, predicted_models, true_makes, true_models):
    metrics_dict = {}
    for make in set(true_makes.cpu().numpy()):
        make_indices = (true_makes == make)
        precision = precision_score(true_models[make_indices].cpu().numpy(), 
                                    predicted_models[make_indices].cpu().numpy(), 
                                    average='weighted')
        recall = recall_score(true_models[make_indices].cpu().numpy(), 
                              predicted_models[make_indices].cpu().numpy(), 
                              average='weighted')
        f1 = f1_score(true_models[make_indices].cpu().numpy(), 
                      predicted_models[make_indices].cpu().numpy(), 
                      average='weighted')
        
        metrics_dict[make] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    return metrics_dict
"""

############### Car Make-Model fine grained classification ###############

def get_best_worst_makes_by_f1(true_makes, predicted_makes, valid_makes):
    """
    Calculate F1 scores for each make and return the best and worst performing makes.
    
    Parameters:
    - true_makes: Array of true make labels.
    - predicted_makes: Array of predicted make labels.
    - valid_makes: List of makes that have more than one model.
    
    Returns:
    - best_makes: List of makes with the highest F1 scores.
    - worst_makes: List of makes with the lowest F1 scores.
    """
    
    true_makes = np.array(true_makes)
    predicted_makes = np.array(predicted_makes)
    # Dictionary to store F1 scores for valid makes
    f1_scores_per_make = {}

    # Iterate over each valid make and calculate its F1 score
    for make in valid_makes:
        # Create masks for the current make
        make_indices = (true_makes == make)

        # Extract true and predicted values for this make
        true_make_labels = true_makes[make_indices]
        predicted_make_labels = predicted_makes[make_indices]
        
        f1 = f1_score(true_make_labels, predicted_make_labels, average='weighted')
        f1_scores_per_make[make] = f1

    # Sort the makes by their F1 scores
    sorted_makes = sorted(f1_scores_per_make.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top 3 best makes and bottom 3 worst makes based on F1 score
    best_makes = [make for make, f1 in sorted_makes[:3]]
    worst_makes = [make for make, f1 in sorted_makes[-3:]]

    return best_makes, worst_makes


def get_best_worst_models_per_make(true_models, predicted_models, true_makes, selected_makes, num_models_per_make):
    """
    Get the best and worst models per make based on confusion matrix and accuracy.

    Parameters:
    - true_models: Array of true model labels.
    - predicted_models: Array of predicted model labels.
    - true_makes: Array of true make labels.
    - selected_makes: List of makes to evaluate.
    - num_models_per_make: List containing the number of models for each make.

    Returns:
    - confusion_matrices: Dictionary of confusion matrices for each make.
    - model_accuracies_per_make: Dictionary with best and worst models for each make.
    """
    
    confusion_matrices = {}
    model_accuracies_per_make = {}

    true_makes = np.array(true_makes)
    true_models = np.array(true_models)
    predicted_models = np.array(predicted_models)

    for make in selected_makes:
        # Create boolean mask for the current make
        make_indices = (true_makes == make)

        # Apply the boolean mask to both true and predicted models
        true_models_make = true_models[make_indices]
        predicted_models_make = predicted_models[make_indices]

        # Skip this make if there are no true or predicted models for it
        if len(true_models_make) == 0 or len(predicted_models_make) == 0:
            print(f"Skipping make {make} due to no true or predicted models.")
            continue

        # Get unique labels that are actually present in true and predicted models
        present_labels = np.unique(np.concatenate([true_models_make, predicted_models_make]))

        # Skip if no present labels are found (to avoid empty confusion matrix)
        if len(present_labels) == 0:
            print(f"Skipping make {make} as no present labels are found.")
            continue

        # Compute the confusion matrix using only the present labels
        try:
            cm = confusion_matrix(true_models_make, predicted_models_make, labels=present_labels)
            confusion_matrices[make] = cm
        except ValueError as e:
            print(f"Error computing confusion matrix for make {make}: {e}")
            continue

        # Calculate accuracy for each model within the make
        model_accuracies = {}
        for model in present_labels:
            model_indices = (true_models_make == model)
            # Avoid division by zero
            if model_indices.sum() == 0:
                model_accuracies[model] = 0
            else:
                accuracy = (predicted_models_make[model_indices] == true_models_make[model_indices]).sum() / model_indices.sum()
                model_accuracies[model] = accuracy

        # Sort the models by accuracy and select best/worst models
        sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)
        best_models = [x[0] for x in sorted_models[:3]]  # Top 3 best models
        worst_models = [x[0] for x in sorted_models[-3:]]  # Bottom 3 worst models
        model_accuracies_per_make[make] = {'best_models': best_models, 'worst_models': worst_models}

    return confusion_matrices, model_accuracies_per_make




def get_class_names(model_mapping, selected_makes, make_mapping):
    """
    Generate class names for the selected makes using the original make labels.

    Parameters:
    - model_mapping: A dictionary where keys are the original make labels (as strings).
    - selected_makes: List of 0-indexed selected makes for plotting confusion matrices (as integers).
    - make_mapping: A dictionary mapping original make labels (as strings) to 0-indexed labels (as integers).

    Returns:
    - class_names_dict: A dictionary where keys are 0-indexed make IDs and values are lists of class names (model names).
    """
    class_names_dict = {}

    # Reverse the make_mapping to map 0-indexed labels back to original labels
    reverse_make_mapping = {v: k for k, v in make_mapping.items()}

    for make in selected_makes:
        # Use the reverse mapping to get the original make label from the 0-indexed label
        original_make = reverse_make_mapping.get(make)
        if original_make is not None:
            if original_make in model_mapping:
                # Extract the model class names for the current make
                models_for_make = model_mapping[original_make]
                class_names = [f"Model_{model}" for model in sorted(models_for_make.keys())]
                class_names_dict[make] = class_names

    return class_names_dict



def plot_confusion_matrix_best_worst_models(confusion_matrices, best_worst_models, class_names_dict, make_mapping):
    """
    Plot the normalized confusion matrices for the best and worst models for the selected makes.
    If there are fewer than six models, it plots the confusion matrix with the available models.
    
    Parameters:
    - confusion_matrices: Dictionary containing confusion matrices for each make.
    - best_worst_models: Dictionary containing best and worst models for each make.
    - class_names_dict: Dictionary mapping make indices to their corresponding model names.
    """
    
    for make, cm in confusion_matrices.items():
        best_models = best_worst_models[make]['best_models']
        worst_models = best_worst_models[make]['worst_models']

        # Combine best and worst models
        combined_models = list(set(best_models + worst_models))

        # Check if there are any models to plot
        if not combined_models:
            print(f"No models to plot for make {make}. Skipping.")
            continue

        # Get class names
        class_names = class_names_dict.get(make)
        if not class_names:
            print(f"No class names found for make {make}. Skipping.")
            continue

        # Check if the confusion matrix is valid
        if cm is None or cm.size == 0:
            print(f"Invalid or empty confusion matrix for make {make}. Skipping.")
            continue

        # Adjust the number of models based on availability
        num_combined_models = min(6, len(combined_models))
        selected_models = combined_models[:num_combined_models]

        # Ensure selected_models indices are valid
        max_index = cm.shape[0] - 1
        selected_models = [i for i in selected_models if i <= max_index]

        if not selected_models:
            print(f"No valid model indices for make {make}. Skipping.")
            continue
        
        model_labels = [class_names[m] for m in selected_models]

        # Extract the sub-matrix for the selected models
        selected_cm = cm[np.ix_(selected_models, selected_models)]

        # Normalize the confusion matrix row-wise
        row_sums = selected_cm.sum(axis=1, keepdims=True)
        normalized_cm = np.divide(selected_cm, row_sums, where=row_sums != 0) 
        
        # Get the number of samples (support) for each model
        selected_support = row_sums.flatten().astype(int)

        # Annotate the support values in the model class labels
        support_labels = [f'{label}\n(n={sup})' for label, sup in zip(model_labels, selected_support)]

        # Reverse the mapping to find the original label corresponding to the 0-index label
        reverse_make_mapping = {v: k for k, v in make_mapping.items()}
        original_make_label = reverse_make_mapping.get(make, str(make))

        # Dynamically adjust figure size based on the number of models
        fig_size = (min(8, num_combined_models + 2), min(6, num_combined_models + 2))

        # Plot the combined confusion matrix (best + worst models)
        plt.figure(figsize=fig_size)
        ax = sns.heatmap(normalized_cm, annot=True, fmt='.2f', cmap='Blues',
                         xticklabels=support_labels, yticklabels=support_labels)
        plt.title(f'Normalized Confusion Matrix for Best and Worst Models (Make {original_make_label})')
        plt.savefig(f'combined_best_worst_models_confusion_make_{original_make_label}.png')
        plt.close()



def plot_precision_recall_f1_for_models(true_models, predicted_models, true_makes, selected_makes, class_names_dict, make_mapping):
    """
    Plot precision, recall, and F1 scores for all models associated with the selected car make classes.
    
    Parameters:
    - true_models: Array of true model labels.
    - predicted_models: Array of predicted model labels.
    - true_makes: Array of true make labels.
    - selected_makes: List of selected car makes for evaluation.
    - class_names_dict: Dictionary containing class names for each make.
    """

    true_models = np.array(true_models)
    predicted_models = np.array(predicted_models)
    selected_makes = np.array(selected_makes)
    for make in selected_makes:
        # Get the mask for the current make
        make_indices = (true_makes == make)
        if not np.any(make_indices):
            print(f"No samples for make {make}. Skipping.")
            continue
        
        # Extract true and predicted models for the current make
        true_models_make = true_models[make_indices]
        predicted_models_make = predicted_models[make_indices]
        
        # Reverse the mapping to find the original label corresponding to the 0-index label
        reverse_make_mapping = {v: k for k, v in make_mapping.items()}
        original_make_label = reverse_make_mapping.get(make, str(make))

        # Get class names for this make
        class_names = class_names_dict.get(make, [])
        # Calculate precision, recall, and F1 scores for all models within this make
        precision, recall, f1, _ = precision_recall_fscore_support(true_models_make, predicted_models_make, labels=range(len(class_names)), zero_division=0)

        # Plot precision, recall, and F1 scores for all models
        x = np.arange(len(class_names))  # Create an array with model indices
        
        plt.figure(figsize=(12, 6))
        
        # Plot precision, recall, and F1
        plt.bar(x - 0.2, precision, width=0.2, label='Precision', align='center')
        plt.bar(x, recall, width=0.2, label='Recall', align='center')
        plt.bar(x + 0.2, f1, width=0.2, label='F1 Score', align='center')

        # Add labels and titles
        plt.xticks(x, class_names, rotation=90)  # Display model names on x-axis
        plt.xlabel('Model Class')
        plt.ylabel('Score')
        plt.title(f'Precision, Recall, F1 for Make {original_make_label}')
        plt.legend()
        
        # Adjust layout and save plot
        plt.tight_layout()
        plt.savefig(f'precision_recall_f1_make_{original_make_label}.png')
        plt.close()



def plot_micro_averaged_roc_selected_models(true_makes, predicted_makes, true_models, predicted_models, selected_makes, num_models_per_make, make_mapping):
    
    true_makes = np.array(true_makes)
    predicted_makes = np.array(predicted_makes)
    true_models = np.array(true_models)
    predicted_models = np.array(predicted_models)
    selected_makes = np.array(selected_makes)

    for make in selected_makes:
        make_indices = (true_makes == make)
        if not np.any(make_indices):
            print(f"No samples for make {make}. Skipping.")
            continue
        
        # Extract true and predicted models for the current make
        true_models_make = true_models[make_indices]
        predicted_models_make = predicted_models[make_indices]
        
        true_models_bin = label_binarize(true_models[make_indices], classes=list(range(num_models_per_make[make])))
        predicted_models_bin = label_binarize(predicted_models[make_indices], classes=list(range(num_models_per_make[make])))

        # Reverse the mapping to find the original label corresponding to the 0-index label
        reverse_make_mapping = {v: k for k, v in make_mapping.items()}
        original_make_label = reverse_make_mapping.get(make, str(make))

        # Compute ROC curve and AUC for models within the make
        if len(np.unique(true_models_make)) > 1:  # Ensure there are both positive and negative samples
            fpr, tpr, _ = roc_curve(true_models_bin.ravel(), predicted_models_bin.ravel())
        else:
            print(f"Skipping ROC curve for make {make} as it has no negative samples.")
            continue
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Micro-Averaged ROC for Make {original_make_label}')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curve_make_{original_make_label}.png')
        plt.close()










