import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from collections import Counter
from torch.utils.data.dataloader import default_collate


#########Car make classification#########

class MakeDataset(Dataset):
    def __init__(self, image_paths_file, root_dir, labels_dir, transform=None):
        """
        Args:
            image_paths_file (string): Path to the text file with image paths.
            root_dir (string): Directory with all the images.
            labels_dir (string): Directory with all the label text files.
            transform : Optional transform to be applied on a sample.
        """
        # Load image paths from the provided file
        self.image_paths = []
        with open(image_paths_file, 'r') as f:
            for line in f:
                self.image_paths.append(line.strip())
                
        self.root_dir = root_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # Create the label mapping
        self.label_mapping = self.create_label_mapping()

    def create_label_mapping(self):
        """Create a mapping from original labels to new labels in range [0, num_classes - 1]."""
        all_labels = []
        for idx in range(len(self.image_paths)):
            label_path = os.path.join(self.labels_dir, self.image_paths[idx].replace('.jpg', '.txt'))
            label = self.get_label_from_path(label_path)
            all_labels.append(label)
        
        unique_labels = sorted(set(all_labels))
        label_mapping = {original_label: new_label for new_label, original_label in enumerate(unique_labels)}
        return label_mapping

    def get_bounding_box(self, label_path):
        """Retrieve the bounding box coordinates from the label file."""
        with open(label_path, 'r') as f:
            lines = f.readlines()
            # The bounding box coordinates are on the 3rd line
            x1, y1, x2, y2 = map(int, lines[2].strip().split())
            return x1, y1, x2, y2

    def get_label_from_path(self, img_path):
        """Extract the car make from the image path"""
        parts = img_path.split('/')
        make = parts[-4]   # Fourth last element is the make
        return make

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Construct the full path to the label file
        label_path = os.path.join(self.labels_dir, self.image_paths[idx].replace('.jpg', '.txt'))

        # Get the bounding box coordinates
        x1, y1, x2, y2 = self.get_bounding_box(label_path)

        # Crop the image using the bounding box
        image = image.crop((x1, y1, x2, y2))

        # Get the car make label from the label path and map it to the new label
        original_label = self.get_label_from_path(label_path)
        label = self.label_mapping[original_label]

        # Apply the transformations
        if self.transform:
            image = self.transform(image)

        return image, label


"""The following is used in the try with focal loss to add other augment. to lower samples classes"""

class MakeDatasetWithAugmentation(MakeDataset):
    def __init__(self, image_paths_file, root_dir, labels_dir, transform=None, augment_classes=None):
        """
        Args:
            image_paths_file (string): Path to the text file with image paths.
            root_dir (string): Directory with all the images.
            labels_dir (string): Directory with all the label text files.
            transform: Optional transform to be applied on a sample.
            augment_classes (list): List of class labels that need augmentation.
        """
        super().__init__(image_paths_file, root_dir, labels_dir, transform)
        self.augment_classes = augment_classes
        
        # Define the augmentation transforms for low-sample classes
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=2),
            transforms.RandomPerspective(distortion_scale=0.10, p=0.5)
        ])
        
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        
        # Apply augmentation only if the label is in the augment_classes list
        if self.augment_classes and label in self.augment_classes:
            image = self.augmentation_transforms(image)
        
        return image, label



#########Car Make-Model classification#########

class MMdataset(Dataset):
    def __init__(self, image_paths_file, root_dir, labels_dir, transform=None):
        """
        Args:
            image_paths_file (string): Path to the text file with image paths.
            root_dir (string): Directory with all the images.
            labels_dir (string): Directory with all the label text files.
            transform: Optional transform to be applied on a sample.
        """
        # Load image paths from the provided file
        self.image_paths = []
        with open(image_paths_file, 'r') as f:
            for line in f:
                self.image_paths.append(line.strip())
                
        self.root_dir = root_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # Create the label mappings for both make and model
        self.make_mapping, self.model_mapping = self.create_label_mappings()

    def create_label_mappings(self):
        """Create mappings for car make and model labels to numeric labels."""
        model_mapping = {}
        all_makes = set()
        
        for idx in range(len(self.image_paths)):
            img_path = os.path.join(self.root_dir, self.image_paths[idx])
            make, model = self.get_label_from_path(img_path)
            all_makes.add(make)
            
            if make not in model_mapping:
                model_mapping[make] = {}
            
            if model not in model_mapping[make]:
                model_mapping[make][model] = len(model_mapping[make])

        # Create a make mapping
        sorted_makes = sorted(all_makes)
        make_mapping = {make: idx for idx, make in enumerate(sorted(sorted_makes))}
        
        print("All makes found:", make_mapping)
        print("All models found for each make:", model_mapping)
        
        return make_mapping, model_mapping


    def get_bounding_box(self, label_path):
        """Retrieve the bounding box coordinates from the label file."""
        with open(label_path, 'r') as f:
            lines = f.readlines()
            # The bounding box coordinates are on the 3rd line
            x1, y1, x2, y2 = map(int, lines[2].strip().split())
            return x1, y1, x2, y2


    def get_label_from_path(self, img_path):
        """Extract the car make and model from the image path, ignoring the year."""
        parts = img_path.split('/')
        model = parts[-3]  # Third last element is the model
        make = parts[-4]   # Fourth last element is the make
        return make, model

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
          # Load image
          img_path = os.path.join(self.root_dir, self.image_paths[idx])
          image = Image.open(img_path).convert('RGB')

          # Construct the full path to the label file
          label_path = os.path.join(self.labels_dir, self.image_paths[idx].replace('.jpg', '.txt'))

          # Get the bounding box coordinates
          x1, y1, x2, y2 = self.get_bounding_box(label_path)

          # Crop the image using the bounding box
          image = image.crop((x1, y1, x2, y2))

          # Extract make and model directly from the image path
          make, model = self.get_label_from_path(img_path)
          make_label = self.make_mapping[make]
            
          model_label = self.model_mapping[make][model]  # Use per-make model mapping

          # Apply the transformations
          if self.transform:
              image = self.transform(image)
                
          return image, make_label, model_label



class MMDatasetWithAugmentation(MMdataset):
    def __init__(self, image_paths_file, root_dir, labels_dir, transform=None, augment_classes=None):
        """
        Args:
            image_paths_file (string): Path to the text file with image paths.
            root_dir (string): Directory with all the images.
            labels_dir (string): Directory with all the label text files.
            transform: Optional transform to be applied on a sample.
            augment_classes (list): List of class labels that need augmentation.
        """
        super().__init__(image_paths_file, root_dir, labels_dir, transform)
        self.augment_classes = augment_classes
        
        # Define the augmentation transforms for low-sample classes
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=2),
            transforms.RandomPerspective(distortion_scale=0.10, p=0.5)
        ])
        
    def __getitem__(self, idx):
        image, make_label, model_label = super().__getitem__(idx)
        
        # Apply augmentation only if the label is in the augment_classes list
        if self.augment_classes and make_label in self.augment_classes:
            image = self.augmentation_transforms(image)
        
        return image, make_label, model_label



############## Fine-grained classification with car parts images ##############

class CarPartsDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with image paths.
            root_dir (string): Directory with all the images.
            transform: Optional transform to be applied on a sample.
        """
        with open(txt_file, 'r') as f:
            self.image_paths = [line.strip() for line in f]
        self.root_dir = root_dir
        self.transform = transform
        # Create the label mappings for both make and model
        self.make_mapping, self.model_mapping = self.create_label_mappings()
    
    def __len__(self):
        return len(self.image_paths)
  
    def create_label_mappings(self):
        """Create mappings for car make and model labels to numeric labels."""
        model_mapping = {}
        all_makes = set()
        
        for idx in range(len(self.image_paths)):
            img_path = os.path.join(self.root_dir, self.image_paths[idx])
            make, model = self.get_label_from_path(img_path)
            all_makes.add(make)
            
            if make not in model_mapping:
                model_mapping[make] = {}
            
            if model not in model_mapping[make]:
                model_mapping[make][model] = len(model_mapping[make])

        # Create a make mapping
        sorted_makes = sorted(all_makes)
        make_mapping = {make: idx for idx, make in enumerate(sorted(sorted_makes))}
        
        return make_mapping, model_mapping

    def get_label_from_path(self, img_path):
        """Extract the car make and model from the image path, ignoring the year."""
        parts = img_path.split('/')
        model = parts[-4]  # fourth last element is the model
        make = parts[-5]   # five last element is the make
        return make, model

    def __getitem__(self, idx):
          # Load image
          img_path = os.path.join(self.root_dir, self.image_paths[idx])
          image = Image.open(img_path).convert('RGB')

          # Extract make and model directly from the image path
          make, model = self.get_label_from_path(img_path)
          make_label = self.make_mapping[make]
            
          model_label = self.model_mapping[make][model]  # Use per-make model mapping

          # Apply the transformations
          if self.transform:
              image = self.transform(image)
                
          return image, make_label, model_label


"""Dataset for test phase using voting strategy"""

class VotingDataset(Dataset):
    def __init__(self, root_dir, txt_files, part_list, transform=None):
        """
        Args:
            root_dir (string): Root directory containing all images.
            txt_files (dict): Dictionary mapping car parts (e.g., 'headlight') to their corresponding .txt files containing image paths.
            part_list (list): List of car parts to be used (e.g., ['headlight', 'taillight', 'fog_light']).
            transform: Optional transform to be applied to each image.
        """
        self.root_dir = root_dir
        self.txt_files = txt_files
        self.part_list = part_list
        self.transform = transform if transform else transforms.ToTensor()  # Set a default transform to tensor conversion
        
        # Load image paths
        self.car_images = self._load_image_paths()
        
        # Create the label mappings for both make and model
        self.make_mapping, self.model_mapping = self.create_label_mappings()

    def _load_image_paths(self):
        """ 
        Load image paths from the .txt files and group them by car.
        """
        car_images = {}
        
        for part, txt_file in self.txt_files.items():
            with open(txt_file, 'r') as f:
                for line in f:
                    relative_image_path = line.strip()
                    # Create the full path by concatenating root_dir with the relative path
                    image_path = os.path.join(self.root_dir, relative_image_path)
                    
                    # Extract car make, model, and year information from the path
                    parts = relative_image_path.split(os.sep)
                    car_make = parts[-5]  # Extract make from path
                    car_model = parts[-4]  # Extract model from path
                    car_identifier = f"{car_make}_{car_model}"
                    
                    if car_identifier not in car_images:
                        car_images[car_identifier] = {}
                    
                    # Add the image for this part to the car's entry
                    car_images[car_identifier][part] = image_path
        
        return car_images
    
    def create_label_mappings(self):
        """Create mappings for car make and model labels to numeric labels."""
        model_mapping = {}
        all_makes = set()
        
        for car_identifier in self.car_images:
            make, model = car_identifier.split('_')
            all_makes.add(make)
            
            if make not in model_mapping:
                model_mapping[make] = {}
            
            if model not in model_mapping[make]:
                model_mapping[make][model] = len(model_mapping[make])

        # Create a make mapping
        sorted_makes = sorted(all_makes)
        make_mapping = {make: idx for idx, make in enumerate(sorted(sorted_makes))}
        
        return make_mapping, model_mapping

    def __len__(self):
        return len(self.car_images)

    def __getitem__(self, idx):
        # Get the car's identifier (make_model)
        car_identifier = list(self.car_images.keys())[idx]
        
        images = {}
        missing_parts = {}  # To track whether an image is a placeholder
        for part in self.part_list:
            image_path = self.car_images[car_identifier].get(part, None)
            if image_path is not None:
                image = Image.open(image_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)  # Convert image to tensor
                images[part] = image
                missing_parts[part] = False
            else:
                # If part image is missing, return a placeholder tensor (zeros)
                images[part] = torch.zeros(3, 224, 224)  # Example: 3 channels, 224x224 image
                missing_parts[part] = True

        # Retrieve make and model labels
        make, model = car_identifier.split('_')
        make_label = self.make_mapping[make]
        model_label = self.model_mapping[make][model]

        return images, make_label, model_label, missing_parts



def custom_collate_fn(batch):
    images_batch = {part: [] for part in batch[0][0].keys()}
    missing_parts_batch = {part: [] for part in batch[0][3].keys()}  # Track missing parts
    make_batch = []
    model_batch = []

    for images, make, model, missing_parts in batch:
        for part, image in images.items():
            images_batch[part].append(image)  # No need to check for None, as it's already handled in the Dataset
            missing_parts_batch[part].append(missing_parts[part])
        make_batch.append(make)
        model_batch.append(model)

    # Use default_collate to stack the tensors
    images_batch = {part: default_collate(images) for part, images in images_batch.items()}
    make_batch = default_collate(make_batch)
    model_batch = default_collate(model_batch)
    missing_parts_batch = {part: default_collate(missing) for part, missing in missing_parts_batch.items()}

    return images_batch, make_batch, model_batch, missing_parts_batch




