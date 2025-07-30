import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import random
from typing import Tuple, List

class CIFAR10SSL:
    """
    CIFAR-10 dataset for semi-supervised learning
    """
    
    def __init__(
        self,
        root: str = './data',
        labeled_samples: int = 4000,
        val_samples: int = 5000,
        download: bool = True,
        seed: int = 42
    ):
        """
        Initialize CIFAR-10 SSL dataset
        
        Args:
            root: Data directory
            labeled_samples: Number of labeled samples to use
            val_samples: Number of validation samples
            download: Whether to download dataset
            seed: Random seed for reproducibility
        """
        self.root = root
        self.labeled_samples = labeled_samples
        self.val_samples = val_samples
        self.seed = seed
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Download and load full dataset
        self.full_train_dataset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=download, transform=None
        )
        
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=download, transform=None
        )
        
        # Split data
        self._split_data()
        
        logger.info(f"CIFAR-10 SSL dataset initialized")
        logger.info(f"Labeled samples: {labeled_samples}")
        logger.info(f"Unlabeled samples: {len(self.unlabeled_indices)}")
        logger.info(f"Validation samples: {val_samples}")
        logger.info(f"Test samples: {len(self.test_dataset)}")
    
    def _split_data(self):
        """Split training data into labeled, unlabeled, and validation sets"""
        train_indices = list(range(len(self.full_train_dataset)))
        
        # Create stratified split for labeled data
        labeled_indices = self._stratified_sample(
            train_indices, 
            self.full_train_dataset.targets, 
            self.labeled_samples
        )
        
        # Create validation set
        remaining_indices = [i for i in train_indices if i not in labeled_indices]
        val_indices = self._stratified_sample(
            remaining_indices,
            [self.full_train_dataset.targets[i] for i in remaining_indices],
            self.val_samples
        )
        
        # Remaining samples are unlabeled
        self.unlabeled_indices = [i for i in remaining_indices if i not in val_indices]
        
        # Store indices
        self.labeled_indices = labeled_indices
        self.val_indices = val_indices
    
    def _stratified_sample(self, indices: List[int], targets: List[int], n_samples: int) -> List[int]:
        """
        Create stratified sample of indices
        
        Args:
            indices: List of indices
            targets: List of targets
            n_samples: Number of samples to select
            
        Returns:
            List of selected indices
        """
        # Group indices by class
        class_indices = {}
        for idx, target in zip(indices, targets):
            if target not in class_indices:
                class_indices[target] = []
            class_indices[target].append(idx)
        
        # Sample from each class
        selected_indices = []
        samples_per_class = n_samples // len(class_indices)
        remainder = n_samples % len(class_indices)
        
        for class_id in sorted(class_indices.keys()):
            class_samples = samples_per_class + (1 if remainder > 0 else 0)
            if remainder > 0:
                remainder -= 1
            
            if class_samples <= len(class_indices[class_id]):
                selected = random.sample(class_indices[class_id], class_samples)
            else:
                # If not enough samples in class, take all
                selected = class_indices[class_id]
            
            selected_indices.extend(selected)
        
        return selected_indices
    
    def get_labeled_dataset(self, transform=None) -> Dataset:
        """
        Get labeled dataset
        
        Args:
            transform: Transform to apply
            
        Returns:
            Labeled dataset
        """
        if transform is None:
            transform = self._get_labeled_transform()
        
        dataset = Subset(self.full_train_dataset, self.labeled_indices)
        return TransformedSubset(dataset, transform)
    
    def get_unlabeled_dataset(self, transform=None) -> Dataset:
        """
        Get unlabeled dataset
        
        Args:
            transform: Transform to apply
            
        Returns:
            Unlabeled dataset
        """
        if transform is None:
            transform = self._get_unlabeled_transform()
        
        dataset = Subset(self.full_train_dataset, self.unlabeled_indices)
        return TransformedSubset(dataset, transform)
    
    def get_val_dataset(self, transform=None) -> Dataset:
        """
        Get validation dataset
        
        Args:
            transform: Transform to apply
            
        Returns:
            Validation dataset
        """
        if transform is None:
            transform = self._get_test_transform()
        
        dataset = Subset(self.full_train_dataset, self.val_indices)
        return TransformedSubset(dataset, transform)
    
    def get_test_dataset(self, transform=None) -> Dataset:
        """
        Get test dataset
        
        Args:
            transform: Transform to apply
            
        Returns:
            Test dataset
        """
        if transform is None:
            transform = self._get_test_transform()
        
        return TransformedSubset(self.test_dataset, transform)
    
    def _get_labeled_transform(self):
        """Get transform for labeled data"""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    def _get_unlabeled_transform(self):
        """Get transform for unlabeled data (weak augmentation)"""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    def _get_test_transform(self):
        """Get transform for test data"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

class TransformedSubset(Dataset):
    """Dataset that applies transform to a subset"""
    
    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, index):
        data, target = self.subset[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, target
    
    def __len__(self):
        return len(self.subset)

def create_data_loaders(
    labeled_batch_size: int = 64,
    unlabeled_batch_size: int = 128,
    val_batch_size: int = 128,
    test_batch_size: int = 128,
    num_workers: int = 4,
    labeled_samples: int = 4000,
    val_samples: int = 5000,
    root: str = './data',
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for FlexMatch training
    
    Args:
        labeled_batch_size: Batch size for labeled data
        unlabeled_batch_size: Batch size for unlabeled data
        val_batch_size: Batch size for validation data
        test_batch_size: Batch size for test data
        num_workers: Number of workers for data loading
        labeled_samples: Number of labeled samples
        val_samples: Number of validation samples
        root: Data directory
        seed: Random seed
        
    Returns:
        Tuple of (labeled_loader, unlabeled_loader, val_loader, test_loader)
    """
    # Create dataset
    dataset = CIFAR10SSL(
        root=root,
        labeled_samples=labeled_samples,
        val_samples=val_samples,
        seed=seed
    )
    
    # Create data loaders
    labeled_loader = DataLoader(
        dataset.get_labeled_dataset(),
        batch_size=labeled_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    unlabeled_loader = DataLoader(
        dataset.get_unlabeled_dataset(),
        batch_size=unlabeled_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset.get_val_dataset(),
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset.get_test_dataset(),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return labeled_loader, unlabeled_loader, val_loader, test_loader

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

import logging
logger = logging.getLogger(__name__) 