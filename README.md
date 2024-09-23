This repository contains the code for various implementations related to the car make and fine-grained model classification tasks. Below is a brief description of the key files:

make_main: Implements the car make classification task using the CrossEntropy loss function.

make_focal_main: Similar to make_main, but uses the Focal Loss function with additional augmentations applied to underrepresented classes.

fine_grained: Implements the car make-model classification task. Certain lines are commented out, depending on whether CrossEntropy or Focal Loss is used.

part_fine_grained: Implements the car part classification task. This script should be run 8 times, each time specifying the path to a different car part type (e.g., headlights, taillights) and saving the corresponding trained models.

voting: Implements the weighted voting strategy for car part classification as described in the report. It requires the 8 models trained from the part_fine_grained task.

custom_dataset: Contains the function for creating the dataset used in each task.

dataset_utility: Contains utility functions for visualizing example images from the dataset.

imbalance_check: Provides a function to inspect the training data distribution, including the number of makes and models, and the number of samples per class.

metrics: Contains metric functions used across the tasks to evaluate the models' performance.

Results: in this folder there are plots and accuracies results for each task. For make classification with CrossEntropy metric results from three different runs can be found. For car parts classification both results from single car part model and voting strategy are present. 

Note that in the main implementation codes you have to modify the path according to your system to create the dataset. 
