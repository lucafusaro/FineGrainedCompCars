# FineGrainedCompCars
Here a short explanation of the content of the file:

'make_main' contains the main implementation for car make classification problem using CrossEntropy loss while

'make_focal_main' is an analogous implementation but using Focal Loss and further augmentations for lower sample classes as described in report.

'fine_grained' contains the main implementation of the car make-model classification, here some lines ae deactivated depending if one want to use CrossEntropy or Focal loss.

'part_fine_grained' contains the main implementation for car part classification, this should be runned 8 times changing the correspondent path to the images to select a different car part type and saving the trained models. 

'voting' contains the implementation of the weighted voting strategy for car part classificartion described in the report, here we need the 8 trained models from the point above.

'custom_dataset contains the function to create the dataset used for each task.

'dataset_utility' contains some functions used in th
