import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from resnet34 import *
from custom_dataset import CarPartsDataset, VotingDataset
from imbalance_check import *
from metrics import *
from dataset_utility import *
from network_utility import *


# Parameters
input_size = 224  # ResNet34 224x224
batch_size = 64
learning_rate = 0.001  
validation_split = 0.2  # Use 20% of the training data for validation
# Seed
seed = 42
torch.manual_seed(seed)

data_transforms = {
    'train': transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test':transforms.Compose([
      transforms.Resize((input_size, input_size)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

train_dataset_full = CarPartsDataset(txt_file='Prog/data/train_test_split/part/train_part_8.txt', root_dir='Prog/data/part', transform=data_transforms['train'])
test_dataset = CarPartsDataset(txt_file='Prog/data/train_test_split/part/test_part_8.txt', root_dir='Prog/data/part', transform=data_transforms['test'])

# Check dataset
check_results = train_dataset_full.create_label_mappings()
num_makes = len(check_results[0])
num_models_per_make = check_results[1]
print("Number of makes in the used dataset:", num_makes)
print("Number of models for each make label in the used dataset:", num_models_per_make)
make_mapping, model_mapping = train_dataset_full.create_label_mappings()
model_counts = [len(model_mapping[make]) for make in sorted(make_mapping.keys())]
print(make_mapping.keys())
print(train_dataset_full.model_mapping.items())
print("model counts:", model_counts)
make_train_distribution= check_class_distribution(train_dataset_full, "make_model")[0]
model_train_distribution= check_class_distribution(train_dataset_full, "make_model")[4]
print("Number of images for each make", make_train_distribution)
print("Number of images for each model within a make", model_train_distribution)

# Split the train_dataset into train and validation datasets
train_size = int((1 - validation_split) * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

# Create DataLoaders for train, validation, and test datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Assuming `num_makes` and `model_counts` (list of model counts per make) are defined
model = FineGrainedResNet34(num_makes=num_makes, model_counts=model_counts)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load pretrained weights for make classification
pretrained_weights = torch.load('best_model_make.pt')
# Filter out the fc layer weights (since we don't need them)
pretrained_weights = {k: v for k, v in pretrained_weights.items() if 'fc' not in k}
model.resnet.load_state_dict(pretrained_weights)

alpha = compute_alpha_inverse_frequency(make_train_distribution, eta=0.4)
focal_loss = FocalLoss(alpha=alpha, gamma=1.5).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)


def train(model, train_loader, val_loader, optimizer, make_criterion=None, scheduler=None, epochs=10, device='cuda'):
    best_val_loss = float('inf')
    
    # Track training/validation metrics
    train_losses = []
    val_losses = []
    train_make_accuracies = []
    val_make_accuracies = []
    train_model_accuracies = []
    val_model_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        total_valid_samples = 0
        correct_make_predictions = 0
        correct_model_predictions = 0
        total_samples = 0  # Keep track of total samples processed for accuracy normalization

        # Training loop with tqdm progress bar
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}", leave=False)
        for images, make_labels, model_labels in pbar:
            images, make_labels, model_labels = images.to(device), make_labels.to(device), model_labels.to(device)

            optimizer.zero_grad()

            # Predict make
            make_preds, features = model(images)
            #loss_make = F.cross_entropy(make_preds, make_labels)
            loss_make = focal_loss(make_preds, make_labels)
            _, predicted_makes = torch.max(make_preds, 1)

            # Calculate make prediction accuracy
            correct_make_predictions += (predicted_makes == make_labels).sum().item()
            total_samples += make_labels.size(0)  # Track total samples

            # Predict model
            model_loss = 0
            valid_samples = 0  # Reset valid samples counter for each batch

            for i in range(len(make_labels)):
                make_idx = predicted_makes[i].item()  # Get the predicted make index as an integer
                true_make_idx = make_labels[i].item()
                model_label = model_labels[i].item()

                # Skip the model loss if the predicted make is wrong
                if make_idx != true_make_idx:
                    continue

                # Compute the model classification loss
                model_preds = model.classify_model(make_idx, features[i].unsqueeze(0))
                loss_model = F.cross_entropy(model_preds, model_labels[i:i+1])
                model_loss += loss_model

                # Calculate model prediction accuracy
                _, predicted_model = torch.max(model_preds, 1)
                correct_model_predictions += (predicted_model == model_labels[i:i+1]).sum().item()

                valid_samples += 1

            if valid_samples > 0:
                model_loss /= valid_samples
            else:
                model_loss = torch.tensor(0.0).to(device)  # Handle case where no samples were valid    

            loss = loss_make + model_loss  # Correctly add the model_loss after averaging
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total_valid_samples += valid_samples

            # Update the tqdm progress bar
            make_accuracy = correct_make_predictions / total_samples
            model_accuracy = correct_model_predictions / (total_valid_samples if total_valid_samples > 0 else 1)
            pbar.set_postfix({'Loss': train_loss / total_samples, 
                              'Make Acc': make_accuracy, 
                              'Model Acc': model_accuracy})

        # Compute the correct average train_loss
        if total_samples > 0:
            train_loss /= total_samples
        else:
            train_loss = torch.tensor(0.0).to(device)  # Handle case where no samples were valid

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Make Accuracy: {make_accuracy:.4f}, Model Accuracy: {model_accuracy:.4f}')

        # Store training metrics
        train_losses.append(train_loss)
        train_make_accuracies.append(make_accuracy)
        train_model_accuracies.append(model_accuracy)

        # Evaluate the model on the validation set
        val_loss, val_make_acc, val_model_acc = evaluate(model, val_loader)

        print(f'Val Loss: {val_loss:.4f}, Val Make Accuracy: {val_make_acc:.4f}, Val Model Accuracy: {val_model_acc:.4f}')

        # Store validation metrics
        val_losses.append(val_loss)
        val_make_accuracies.append(val_make_acc)
        val_model_accuracies.append(val_model_acc)

        # Step the scheduler based on the epoch
        if scheduler:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

    # After training, plot the losses and accuracies
    # Aggregate make and model accuracies for training and validation respectively
    train_accuracies = [0.5 * (make_acc + model_acc) for make_acc, model_acc in zip(train_make_accuracies, train_model_accuracies)]
    val_accuracies = [0.5 * (make_acc + model_acc) for make_acc, model_acc in zip(val_make_accuracies, val_model_accuracies)]

    plot_loss_accuracy_curves(train_losses, val_losses, train_accuracies, val_accuracies)



def evaluate(model, val_loader):
    model.eval()
    val_loss = 0.0
    total_valid_samples = 0
    correct_make_predictions = 0
    correct_model_predictions = 0
    total_samples = 0  # Track total samples for normalization

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for images, make_labels, model_labels in pbar:
            images, make_labels, model_labels = images.to(device), make_labels.to(device), model_labels.to(device)
            # Predict make
            make_preds, features = model(images)
            loss_make = F.cross_entropy(make_preds, make_labels)
            _, predicted_makes = torch.max(make_preds, 1)

            # Calculate make prediction accuracy
            correct_make_predictions += (predicted_makes == make_labels).sum().item()
            total_samples += make_labels.size(0)

            # Predict model
            model_loss = 0
            valid_samples = 0

            for i in range(len(make_labels)):
                make_idx = predicted_makes[i].item()  # Get the predicted make index as an integer
                true_make_idx = make_labels[i].item()
                model_label = model_labels[i].item()

                # Skip the model loss if the predicted make is wrong
                if make_idx != true_make_idx:
                    continue

                # Compute the model classification loss
                model_preds = model.classify_model(make_idx, features[i].unsqueeze(0))
                loss_model = F.cross_entropy(model_preds, model_labels[i:i+1])
                model_loss += loss_model

                # Calculate model prediction accuracy
                _, predicted_model = torch.max(model_preds, 1)
                correct_model_predictions += (predicted_model == model_labels[i:i+1]).sum().item()

                valid_samples += 1

            if valid_samples > 0:
                model_loss /= valid_samples
            else:
                model_loss = torch.tensor(0.0).to(device)  # Handle case where no samples were valid 

            val_loss += (loss_make + loss_model).item()
            total_valid_samples += valid_samples

            # Update the tqdm progress bar
            make_accuracy = correct_make_predictions / total_samples
            model_accuracy = correct_model_predictions / (total_valid_samples if total_valid_samples > 0 else 1)
            pbar.set_postfix({'Loss': val_loss / total_samples, 
                              'Make Acc': make_accuracy, 
                              'Model Acc': model_accuracy})

    # Calculate average validation loss and accuracy
    val_loss /= total_samples
    val_make_acc = correct_make_predictions / total_samples
    val_model_acc = correct_model_predictions / (total_valid_samples if total_valid_samples > 0 else 1)

    return val_loss, val_make_acc, val_model_acc


train(model, train_loader, val_loader, optimizer=optimizer, scheduler=scheduler, make_criterion=focal_loss, epochs=26)


def test(model, test_loader, num_models_per_make, model_mapping, make_mapping):
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()

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
        for images, make_labels, model_labels in test_loader:
            images, make_labels, model_labels = images.to(device), make_labels.to(device), model_labels.to(device)

            # Predict make
            make_preds, features = model(images)
            _, predicted_makes = torch.max(make_preds, 1)

            # Calculate total number of make predictions for accuracy
            correct_make += (predicted_makes == make_labels).sum().item()
            total_samples += make_labels.size(0)

            # Append true and predicted makes to the unfiltered arrays for overall make accuracy
            true_makes_all.extend(make_labels.cpu().numpy())
            predicted_makes_all.extend(predicted_makes.cpu().numpy())

            #print(f"Unique true makes: {set(true_makes_all)}")
            #print(f"Unique predicted makes: {set(predicted_makes_all)}")

            for i in range(len(make_labels)):
                true_make_idx = make_labels[i].item()  # True make index
                predicted_make_idx = predicted_makes[i].item()  # Predicted make index

                # Only append to filtered arrays if the true make is in makes_with_multiple_models
                if true_make_idx not in makes_with_multiple_models:
                    continue

                # Only filter and process further if the make prediction is correct
                if predicted_make_idx == true_make_idx:
                    # Append filtered true and predicted makes
                    true_makes_all_filtered.append(true_make_idx)
                    predicted_makes_all_filtered.append(predicted_make_idx)

                    # Model classification
                    model_preds = model.classify_model(predicted_make_idx, features[i].unsqueeze(0))
                    _, predicted_model = torch.max(model_preds, 1)

                    # Get top 3 model predictions
                    num_classes = model_preds.size(1)
                    # If there are fewer than 3 model classes, adjust the number of top predictions
                    k = min(3, num_classes)
                    top_k_model_preds = torch.topk(model_preds, k, dim=1).indices.squeeze(0)
                    true_model = model_labels[i].item()
                    if true_model in top_k_model_preds.cpu().numpy():
                      top3_correct_model += 1

                    # Append to filtered lists for model accuracy calculation
                    true_models_all_filtered.append(model_labels[i].item())
                    predicted_models_all_filtered.append(predicted_model.item())

                    # Keep track of the filtered indices for this specific make
                    filtered_make_indices[true_make_idx].append(len(true_models_all_filtered) - 1)

                    correct_model += (predicted_model == model_labels[i]).sum().item()
                    valid_model_samples += 1

    # Compute Make Accuracy (not balanced) using unfiltered true/predicted makes
    make_accuracy = correct_make / total_samples

    # Compute Make Balanced Accuracy using unfiltered true/predicted makes
    balanced_make_accuracy = balanced_accuracy_score(true_makes_all, predicted_makes_all)
    
    # Compute Model Accuracy (not balanced) using filtered makes/models
    model_accuracy = correct_model / valid_model_samples

    # Top-3 Accuracy for models
    top3_model_accuracy = top3_correct_model / valid_model_samples
    
    # Balanced Accuracy for Models (per make)
    balanced_accuracy_models_per_make = {}
    for make in makes_with_multiple_models:
        # Get the filtered indices for this make
        make_indices = filtered_make_indices[make]

        # Now filter the true and predicted models corresponding to these indices
        true_models_for_make = [true_models_all_filtered[i] for i in make_indices]
        predicted_models_for_make = [predicted_models_all_filtered[i] for i in make_indices]

        # Ensure there are models for this make
        if len(true_models_for_make) > 0 and len(predicted_models_for_make) > 0:
            balanced_accuracy_models_per_make[make] = balanced_accuracy_score(
                true_models_for_make, predicted_models_for_make
            )
        else:
            balanced_accuracy_models_per_make[make] = 0.0

    balanced_accuracies = list(balanced_accuracy_models_per_make.values())
    mean_balanced_accuracy = np.mean([acc for acc in balanced_accuracies])
    min_balanced_accuracy = np.min([acc for acc in balanced_accuracies])
    max_balanced_accuracy = np.max([acc for acc in balanced_accuracies])

    # Save results
    result_save_path = f'accuracy_results.txt'
    with open(result_save_path, 'w') as f:
          f.write(f'Make Accuracy (Not Balanced): {make_accuracy:.4f}\n')
          f.write(f'Make Accuracy (Balanced): {balanced_make_accuracy:.4f}\n')
          f.write(f'Model Accuracy (Not Balanced): {model_accuracy:.4f}\n')
          f.write(f'Top-3 Model Accuracy: {top3_model_accuracy:.4f}\n')
          f.write(f'Balanced Accuracy (Models Per Make): {balanced_accuracy_models_per_make}\n')
          f.write(f"Mean Balanced Accuracy: {mean_balanced_accuracy:.4f}\n")
          f.write(f"Min Balanced Accuracy: {min_balanced_accuracy:.4f}\n")
          f.write(f"Max Balanced Accuracy: {max_balanced_accuracy:.4f}\n")

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
    plot_micro_averaged_roc_selected_models(true_makes_all_filtered, predicted_makes_all_filtered, true_models_all_filtered, predicted_models_all_filtered, best_makes + worst_makes, num_models_per_make, make_mapping)


test(model, test_loader, num_models_per_make=model_counts, model_mapping=model_mapping, make_mapping=make_mapping)


"""Below the implementation of an analogous test function used to obtain results using a weighted voting strategy
   Weighting is based on the mean model balanced accuracy computed for each car part model before"""

part_list = ['headlight', 'taillight', 'foglight', 'air_intake', 'console', 'steering_wheel', 'dashboard', 'gear_lever']
models = {}

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

# Create the dataset
voting_dataset = VotingDataset(root_dir='Prog/data/part', txt_files=txt_files, part_list=part_list, transform=data_transforms['test'])

# Create a DataLoader for the test set
test_loader = torch.utils.data.DataLoader(voting_dataset, batch_size=64, shuffle=False)

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



