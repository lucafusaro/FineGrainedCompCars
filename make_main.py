import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import os
from custom_dataset import MakeDataset
from resnet34 import *
from imbalance_check import *
from metrics import *
from dataset_utility import *
from network_utility import *


# Parameters
input_size = 224  # ResNet34 224x224
batch_size = 32
epochs = 25
learning_rate = 0.001  
validation_split = 0.2  # Use 20% of the training data for validation
#num_runs = 3  # Number of different seeds/runs
seeds = [42]  #123, 456  # Different seeds for each run

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load the dataset
train_dataset = MakeDataset(image_paths_file='Prog/data/train_test_split/classification/train.txt',
                              root_dir='Prog/data/image',
                              labels_dir='Prog/data/label',
                              transform=data_transforms['train'])

test_dataset = MakeDataset(image_paths_file='Prog/data/train_test_split/classification/test.txt',
                             root_dir='Prog/data/image',
                             labels_dir='Prog/data/label',
                             transform=data_transforms['test'])

#Plotting of dataset image examples before and after cropping (visualize_examples is in 'dataset_utility.py)
visualize_examples(train_dataset)

# Check number of classes
train_distribution, num_classes = check_class_distribution(train_dataset, "make")
print("Number of classes in the used dataset:", num_classes)
print("Number of images for each make", train_distribution)

# Split the train_dataset into train and validation datasets
train_size = int((1 - validation_split) * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create DataLoaders for train, validation, and test datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Function to train and evaluate the model
def run(seed):

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Define the model
    model = ResNet34(num_classes=num_classes)

    # Load the pretrained weights from torchvision's ResNet34
    pretrained_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    #Moving model to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #Print summary of architecture
    input_shape = (3, 224, 224)
    summary(model, input_shape)
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early_stopping = EarlyStopping(patience=3, path=f'best_checkpoint_make.pt')
    
    print(f"Running model with seed {seed}")
    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, epochs, early_stopping, scheduler
    )
    
    # Plotting loss/accuracy (the function is defined in network_utility.py)
    plot_loss_accuracy_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    #Evaluation
    model.load_state_dict(torch.load(f'best_checkpoint_make.pt'))
    model.to(device)
    # Evaluate the model on the test set
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate metrics
    class_names = [str(label) for label in sorted(train_distribution.keys())]
    
    # Compute Overall Metrics
    overall_accuracy, balanced_acc, macro_f1 = evaluate_model(all_labels, all_preds, class_names, num_classes)

    # Confusion matrix for the n best classes and n worst classes
    plot_confusion_matrix_best_worst(all_labels, all_preds, class_names, top_n=3)

    # 4. Analyze Precision, Recall, and F1-Scores
    plot_f1_score_precision_recall(all_labels, all_preds, class_names)

    # 5. Assess Overall Discrimination Ability Using ROC Curve
    plot_micro_averaged_roc(all_labels, all_preds, num_classes)

    # Save results
    result_save_path = f'results.txt'
    with open(result_save_path, 'w') as f:
        f.write(f'Seed {seed}\n')
        f.write(f'Overall accuracy: {overall_accuracy:.4f}\n')
        f.write(f'Balanced accuracy: {balanced_acc:.4f}\n')
        f.write(f'Macro F1 Score: {macro_f1:.4f}\n')
    
    return overall_accuracy, balanced_acc, macro_f1

# Training and validation function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, early_stopping, scheduler=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        progress_bar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", total=len(train_loader), leave=False)

        for i, (inputs, labels) in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            progress_bar.set_postfix(loss=loss.item(), accuracy=(running_corrects.double() / ((i+1) * batch_size)).item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            # Wrap the val_loader with tqdm to create a progress bar
            for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        # Calculate the average loss and accuracy
        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item())

        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Step the scheduler based on the epoch
        if scheduler:
            scheduler.step()

        # Early stopping check
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_losses, val_losses, train_accuracies, val_accuracies


# Running multiple runs with different seeds
accuracies = []
balanced_accuracies = []
f1_scores = []

for seed in seeds:
    overall_accuracy, balanced_acc, macro_f1 = run(seed)
    accuracies.append(overall_accuracy)
    balanced_accuracies.append(balanced_acc)
    f1_scores.append(macro_f1)
