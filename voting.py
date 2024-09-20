import torch
from torchvision import transforms
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from resnet34 import *
from custom_dataset import VotingDataset
from metrics import *


"""Below the implementation of an analogous test function used to obtain results using a weighted voting strategy
   Weighting is based on the mean model balanced accuracy computed for each car part model"""

input_size = 224  # ResNet34 224x224
part_list = ['headlight', 'taillight', 'foglight', 'air_intake', 'console', 'steering_wheel', 'dashboard', 'gear_lever']
models = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

txt_files = {
    'headlight': 'Prog/data/train_test_split/part/test_part_1.txt',
    'taillight': 'Prog/data/train_test_split/part/test_part_2.txt',   
    'foglight': 'Prog/data/train_test_split/part/test_part_3.txt',
    'air_intake': 'Prog/data/train_test_split/part/test_part_4.txt',
    'console':'Prog/data/train_test_split/part/test_part_5.txt',
    'steering_wheel': 'Prog/data/train_test_split/part/test_part_6.txt',
    'dashboard': 'Prog/data/train_test_split/part/test_part_7.txt',
    'gear_lever':'Prog/data/train_test_split/part/test_part_8.txt'
}

data_transforms = {
    'test':transforms.Compose([
      transforms.Resize((input_size, input_size)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# Create the dataset
voting_dataset = VotingDataset(root_dir='Prog/data/part', txt_files=txt_files, part_list=part_list, transform=data_transforms['test'])

# Create a DataLoader for the test set
test_loader = torch.utils.data.DataLoader(voting_dataset, batch_size=64, shuffle=False)

check_results = voting_dataset.create_label_mappings()
num_makes = len(check_results[0])
make_mapping, model_mapping = voting_dataset.create_label_mappings()
model_counts = [len(model_mapping[make]) for make in sorted(make_mapping.keys())]

# Define and load the model for each part
for part in part_list:
    # Create a new model instance for each part
    model = FineGrainedResNet34(num_makes=num_makes, model_counts=model_counts)
    model.to(device)
    model.load_state_dict(torch.load(f'{part}_model.pt'))  # Load the corresponding weights
    model.eval()  # Set to evaluation mode
    
    # Store the model in the models dictionary with the part as the key
    models[part] = model

# Mean balanced model accuracy for each part
mean_balanced_accuracies = {
    'headlight': 0.6667,
    'taillight': 0.7496,   
    'foglight': 0.5365,
    'air_intake': 0.6290,
    'console':0.6540,
    'steering_wheel': 0.1310,
    'dashboard': 0.1346,
    'gear_lever':0.5381
    }

# Normalize the balanced accuracies to sum to 1
total_balanced_accuracy = sum(mean_balanced_accuracies.values())
weights = {part: acc / total_balanced_accuracy for part, acc in mean_balanced_accuracies.items()}


def test(models, test_loader, num_models_per_make, model_mapping, make_mapping, weights):
    correct_make = 0
    correct_model = 0
    total_samples = 0
    valid_model_samples = 0  # Track valid samples for model classification
    top3_correct_model = 0

    # Arrays to store all true/predicted makes (unfiltered)
    true_makes_all, predicted_makes_all = [], []

    # Filtered arrays for makes and models
    true_makes_all_filtered, predicted_makes_all_filtered = [], []
    true_models_all_filtered, predicted_models_all_filtered = [], []

    # Filter makes with only one model (we won't consider these for model classification)
    makes_with_multiple_models = [make for make, count in enumerate(num_models_per_make) if count > 1]

    # To store the filtered indices for each make
    filtered_make_indices = {make: [] for make in makes_with_multiple_models}

    with torch.no_grad():
        for images, make_labels, model_labels, missing_parts in test_loader:
            make_labels = make_labels.to(device)
            model_labels = model_labels.to(device)
            batch_size = len(make_labels)

            # Store predictions from all models for this batch
            all_make_preds = {}
            all_model_features = {}

            # For each part (headlight, taillight, etc.), get the predictions
            for part, model in models.items():
                if part in images and images[part] is not None and not missing_parts[part].all():  # Skip missing parts
                    part_images = images[part].to(device)
                    make_preds, features = model(part_images)

                    # Weight the predictions by the respective part's weight
                    all_make_preds[part] = weights[part] * make_preds.cpu().numpy()  # Shape: [batch_size, num_makes]
                    all_model_features[part] = features  # Shape: [batch_size, feature_dim]
                else:
                    continue  # Skip if this part image is missing

            if not all_make_preds:
                continue  # Skip this batch if no predictions are available

            # Combine the make predictions using weighted voting
            combined_make_preds = np.sum([preds for preds in all_make_preds.values()], axis=0)  # Shape: [batch_size, num_makes]
            combined_make_preds = np.argmax(combined_make_preds, axis=1)
            combined_make_preds = torch.tensor(combined_make_preds).to(device)

            # Calculate total number of make predictions for accuracy
            correct_make += (combined_make_preds == make_labels).sum().item()
            total_samples += make_labels.size(0)

            # Append true and predicted makes to the unfiltered arrays for overall make accuracy
            true_makes_all.extend(make_labels.cpu().numpy())
            predicted_makes_all.extend(combined_make_preds.cpu().numpy())

            # Now handle model classification using the correct make prediction
            for i in range(batch_size):
                true_make_idx = make_labels[i].item()  # True make index
                predicted_make_idx = combined_make_preds[i].item()  # Predicted make index
    
                if true_make_idx not in makes_with_multiple_models:
                    continue

                if predicted_make_idx == true_make_idx:
                    # Append filtered true and predicted makes
                    true_makes_all_filtered.append(true_make_idx)
                    predicted_makes_all_filtered.append(predicted_make_idx)

                    # Model classification (combine model classification predictions)
                    model_preds_combined = []
                    for part in all_model_features:
                        if not missing_parts[part][i]:  # Skip if the part is missing for this sample
                            features = all_model_features[part]
                            # Obtain model predictions for the current make and sample
                            model_preds = models[part].classify_model(predicted_make_idx, features[i].unsqueeze(0))
                            model_preds_combined.append(weights[part] * model_preds.cpu().numpy())  # Shape: [1, num_models]

                    if not model_preds_combined:
                        continue  # Skip if no model predictions are available
 
                    # Sum over parts
                    combined_model_preds = np.sum(model_preds_combined, axis=0)  # Shape: [1, num_models]
                    predicted_model = np.argmax(combined_model_preds, axis=1)[0]

                    # Get top 3 model predictions
                    num_classes = combined_model_preds.shape[1]
                    k = min(3, num_classes)
                    top_k_model_preds = np.argsort(-combined_model_preds, axis=1)[0, :k]
                    true_model = model_labels[i].item()
                    if true_model in top_k_model_preds:
                        top3_correct_model += 1

                    # Append to filtered lists for model accuracy calculation
                    true_models_all_filtered.append(true_model)
                    predicted_models_all_filtered.append(predicted_model)

                    correct_model += int(predicted_model == true_model)
                    valid_model_samples += 1
                    # Keep track of the filtered indices for this specific make
                    filtered_make_indices[true_make_idx].append(len(true_models_all_filtered) - 1)

    # Compute Make Accuracy (not balanced) using unfiltered true/predicted makes
    make_accuracy = correct_make / total_samples
    balanced_make_accuracy = balanced_accuracy_score(true_makes_all, predicted_makes_all)

    # Compute Model Accuracy (not balanced)
    model_accuracy = correct_model / valid_model_samples
    top3_model_accuracy = top3_correct_model / valid_model_samples

    # Compute Balanced Accuracy for Models (per make)
    balanced_accuracy_models_per_make = {}
    for make in makes_with_multiple_models:
        make_indices = filtered_make_indices[make]

        true_models_for_make = [true_models_all_filtered[i] for i in make_indices]
        predicted_models_for_make = [predicted_models_all_filtered[i] for i in make_indices]

        if len(true_models_for_make) > 0 and len(predicted_models_for_make) > 0:
            balanced_accuracy_models_per_make[make] = balanced_accuracy_score(true_models_for_make, predicted_models_for_make)
        else:
            balanced_accuracy_models_per_make[make] = 0.0

    balanced_accuracies = list(balanced_accuracy_models_per_make.values())
    mean_balanced_accuracy = np.mean(balanced_accuracies)
    min_balanced_accuracy = np.min(balanced_accuracies)
    max_balanced_accuracy = np.max(balanced_accuracies)

    # Save results to a text file
    result_save_path = 'accuracy_results.txt'
    with open(result_save_path, 'w') as f:
        f.write(f'Make Accuracy: {make_accuracy:.4f}\n')
        f.write(f'Make Balanced Accuracy: {balanced_make_accuracy:.4f}\n')
        f.write(f'Model Accuracy: {model_accuracy:.4f}\n')
        f.write(f'Top-3 Model Accuracy: {top3_model_accuracy:.4f}\n')
        f.write(f'Mean Balanced Accuracy for Models: {mean_balanced_accuracy:.4f}\n')
        f.write(f'Min Balanced Accuracy: {min_balanced_accuracy:.4f}\n')
        f.write(f'Max Balanced Accuracy: {max_balanced_accuracy:.4f}\n')

    print(f'Results saved to {result_save_path}')

    # Perform the previous analysis using the true and predicted makes/models
    best_makes, worst_makes = get_best_worst_makes_by_f1(
        true_makes_all_filtered, predicted_makes_all_filtered, makes_with_multiple_models
    )

    confusion_matrices, best_worst_models = get_best_worst_models_per_make(
        true_models_all_filtered, predicted_models_all_filtered, true_makes_all_filtered, best_makes + worst_makes, num_models_per_make
    )

    # Use make_mapping to get class names
    class_names_dict = get_class_names(model_mapping, best_makes + worst_makes, make_mapping)

    # Plotting functions...
    plot_confusion_matrix_best_worst_models(confusion_matrices, best_worst_models, class_names_dict, make_mapping)
    plot_precision_recall_f1_for_models(true_models_all_filtered, predicted_models_all_filtered, true_makes_all_filtered, best_makes + worst_makes, class_names_dict, make_mapping)
    plot_micro_averaged_roc_selected_models(true_makes_all_filtered, predicted_models_all_filtered, true_models_all_filtered, predicted_models_all_filtered, best_makes + worst_makes, num_models_per_make, make_mapping)



test(models, test_loader, num_models_per_make=model_counts, model_mapping=model_mapping, make_mapping=make_mapping, weights=weights)



