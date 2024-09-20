from collections import Counter, defaultdict
import os
import torch
import numpy as np
from custom_dataset import *


def check_class_distribution(dataset, mode="make"):
    make_label_counts = Counter()
    model_label_counts = Counter()
    make_to_models = defaultdict(set)
    model_counts_per_make = defaultdict(Counter)

    for idx in range(len(dataset)):
        image_path = dataset.image_paths[idx]

        if mode == "make":
            make_label = dataset.get_label_from_path(image_path) 
            make_label_counts[make_label] += 1
        
        elif mode == "make_model":
            make_label, model_label = dataset.get_label_from_path(image_path)
            make_label_counts[make_label] += 1
            model_label_counts[model_label] += 1
            make_to_models[make_label].add(model_label)
            model_counts_per_make[make_label][model_label] += 1

    num_make_labels_present = len(make_label_counts)
    num_model_labels_present = len(model_label_counts)

    if mode == "make":
        return make_label_counts, num_make_labels_present

    elif mode == "make_model":
        make_to_model_count = {make: len(models) for make, models in make_to_models.items()}
        return make_label_counts, num_make_labels_present, make_to_model_count, num_model_labels_present, model_counts_per_make


#############################################################
#Car parts classification

def check_part_class_distribution(dataset_dir):
    """
    Checks the distribution of classes (car parts) in the dataset.

    Parameters:
    - dataset_dir (str): Path to the dataset directory (e.g., 'dataset/train' or 'dataset/test').

    Returns:
    - class_distribution (Counter): A Counter object containing the count of images for each class.
    - num_classes_present (int): The number of unique classes present in the dataset.
    """
    class_distribution = Counter()

    # Iterate over each class directory in the dataset
    for class_dir in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_dir)
        
        if os.path.isdir(class_path):
            # Count the number of images in the class directory
            num_images = len([img for img in os.listdir(class_path) if img.endswith('.jpg')])
            class_distribution[class_dir] += num_images

    # Count the number of unique classes
    num_classes_present = len(class_distribution)

    return class_distribution, num_classes_present

