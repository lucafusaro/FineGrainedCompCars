import matplotlib.pyplot as plt
import random
import os
from PIL import Image
import torch
import torchvision
from imbalance_check import check_class_distribution

# Function to plot images
def plot_images(images, labels, title):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    for i, (img, label) in enumerate(zip(images, labels)):
        ax = axes[i // 3, i % 3]
        ax.imshow(img)
        ax.set_title(f'Car maker: {label}')
        ax.axis('off')
    
    plt.savefig(f'{title}.png')
    plt.close()

# Function to visualize examples
def visualize_examples(dataset, num_examples=6):
    indices = random.sample(range(len(dataset)), num_examples)
    original_images, cropped_images, labels = [], [], []

    for idx in indices:
        # Load image and label
        img_path = os.path.join(dataset.root_dir, dataset.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Construct the full path to the label file
        label_path = os.path.join(dataset.labels_dir, dataset.image_paths[idx].replace('.jpg', '.txt'))

        # Get the bounding box coordinates
        x1, y1, x2, y2 = dataset.get_bounding_box(label_path)

        # Get the car make label from the label path and map it to the new label
        original_label = dataset.get_label_from_path(label_path)
        label = dataset.label_mapping[original_label]
        
        # Append original image and label
        original_images.append(image.copy())
        labels.append(label)
        
        # Crop the image using the bounding box
        cropped_image = image.crop((x1, y1, x2, y2))
        
        """# Apply the transformations if any
        if dataset.transform:
            cropped_image = dataset.transform(cropped_image)
        
        # Convert the image back to PIL for visualization (skip this if the transform is not applied)
        if isinstance(cropped_image, torch.Tensor):
            cropped_image = transforms.ToPILImage()(cropped_image)"""
        
        cropped_images.append(cropped_image)

    # Plot original images
    plot_images(original_images, labels, "Original Images with Labels")

    # Plot cropped images
    plot_images(cropped_images, labels, "Cropped Images with Labels")


def get_low_sample_classes(train_dataset, threshold=50):
    """
    Identify classes with fewer samples than the specified threshold.
    Returns: a list of class labels with fewer samples than the threshold.
    """
    train_distribution = check_class_distribution(train_dataset)[0]
    low_sample_classes = [label for label, count in train_distribution.items() if count < threshold]
    return low_sample_classes

    
##########Car parts classficiation##########

def unnormalize_image(image, mean, std):
    mean = torch.tensor(mean).reshape(1, 3, 1, 1)
    std = torch.tensor(std).reshape(1, 3, 1, 1)
    return image * std + mean

def visualize_loader_examples(data_loader, part_names, num_images=6):
    # Get a batch of images and labels
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Select the first num_images images
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Unnormalize the images
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images = unnormalize_image(images, mean, std)

    # Create a grid of images with padding between them
    grid_img = torchvision.utils.make_grid(images, nrow=3, padding=5, pad_value=255)
    
    # Convert the grid image to numpy for plotting
    plt.figure(figsize=(10, 6))
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    
    # Set the title with the associated part names
    plt.title(" | ".join([part_names[label.item() + 1] for label in labels]))
    plt.axis('off')  # Turn off the axis
    plt.savefig('part_train_examples.png')
    plt.close()

