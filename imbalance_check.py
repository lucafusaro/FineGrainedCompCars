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

