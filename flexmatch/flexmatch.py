import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlexMatch:
    """
    FlexMatch implementation with Curriculum Pseudo Labeling (CPL)
    
    Key features:
    - Adaptive class-specific thresholds
    - Warm-up phase for initial training
    - Weak/strong augmentation consistency
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 10,
        base_threshold: float = 0.95,
        warmup_iterations: int = 40000,
        lambda_u: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize FlexMatch
        
        Args:
            model: Neural network model
            num_classes: Number of classes (10 for CIFAR-10)
            base_threshold: Base confidence threshold (τ)
            warmup_iterations: Number of iterations for warm-up phase
            lambda_u: Weight for unsupervised loss
            device: Device to run on
        """
        self.model = model.to(device)
        self.num_classes = num_classes
        self.base_threshold = base_threshold
        self.warmup_iterations = warmup_iterations
        self.lambda_u = lambda_u
        self.device = device
        
        # Initialize class-specific learning effects
        self.sigma = torch.zeros(num_classes, device=device)
        self.beta = torch.ones(num_classes, device=device)
        self.thresholds = torch.full((num_classes,), base_threshold, device=device)
        
        # Track unlabeled predictions
        self.unlabeled_predictions = []
        self.current_iteration = 0
        
        # Optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.03,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1024
        )
        
        logger.info(f"FlexMatch initialized with {num_classes} classes")
        logger.info(f"Base threshold: {base_threshold}")
        logger.info(f"Warmup iterations: {warmup_iterations}")
    
    def update_learning_effects(self, unlabeled_predictions: torch.Tensor) -> None:
        """
        Update class-specific learning effects (σ(c))
        
        Args:
            unlabeled_predictions: Predicted classes for unlabeled samples
        """
        # Count predictions per class
        for pred in unlabeled_predictions:
            if pred >= 0:  # Valid prediction
                self.sigma[pred] += 1
    
    def compute_adaptive_factors(self) -> None:
        """
        Compute adaptive factors β(c) and update thresholds T(c)
        """
        max_sigma = torch.max(self.sigma)
        unused_samples = len([p for p in self.unlabeled_predictions if p == -1])
        
        # Check if we're in warm-up phase
        if max_sigma < unused_samples and self.current_iteration < self.warmup_iterations:
            # Warm-up phase
            warmup_factor = 1.0 / np.sqrt(self.current_iteration / self.warmup_iterations)
            self.beta = torch.full((self.num_classes,), warmup_factor, device=self.device)
            logger.debug(f"Warm-up phase: β = {warmup_factor:.3f}")
        else:
            # Normal operation
            if max_sigma > 0:
                self.beta = self.sigma / max_sigma
            else:
                self.beta = torch.ones(self.num_classes, device=self.device)
            logger.debug(f"Normal phase: β = {self.beta.cpu().numpy()}")
        
        # Update thresholds
        self.thresholds = self.base_threshold * self.beta
    
    def weak_augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply weak augmentation (same as FixMatch)
        
        Args:
            x: Input tensor
            
        Returns:
            Weakly augmented tensor
        """
        # Standard weak augmentation for CIFAR-10
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        ])
        
        # Apply transformation
        if len(x.shape) == 4:
            # Batch of images
            augmented = []
            for img in x:
                img_pil = transforms.ToPILImage()(img)
                augmented.append(transforms.ToTensor()(transform(img_pil)))
            result = torch.stack(augmented)
        else:
            # Single image
            img_pil = transforms.ToPILImage()(x)
            result = transforms.ToTensor()(transform(img_pil))
        
        # Move result to the same device as input
        return result.to(x.device)
    
    def strong_augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply strong augmentation (same as FixMatch)
        
        Args:
            x: Input tensor
            
        Returns:
            Strongly augmented tensor
        """
        # Strong augmentation for CIFAR-10
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        # Apply transformation
        if len(x.shape) == 4:
            # Batch of images
            augmented = []
            for img in x:
                img_pil = transforms.ToPILImage()(img)
                augmented.append(transforms.ToTensor()(transform(img_pil)))
            result = torch.stack(augmented)
        else:
            # Single image
            img_pil = transforms.ToPILImage()(x)
            result = transforms.ToTensor()(transform(img_pil))
        
        # Move result to the same device as input
        return result.to(x.device)
    
    def get_pseudo_labels(self, unlabeled_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate pseudo-labels using weak augmentation and adaptive thresholds
        
        Args:
            unlabeled_batch: Batch of unlabeled images
            
        Returns:
            Tuple of (pseudo_labels, mask) where mask indicates which samples to use
        """
        batch_size = unlabeled_batch.size(0)
        pseudo_labels = torch.zeros(batch_size, self.num_classes, device=self.device)
        mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Apply weak augmentation
        weak_aug = self.weak_augment(unlabeled_batch)
        
        with torch.no_grad():
            weak_predictions = self.model(weak_aug)
            confidence, predicted_classes = torch.max(weak_predictions, dim=1)
            
            # Apply class-specific adaptive thresholds
            adaptive_thresholds = self.thresholds[predicted_classes]
            
            # Create mask for samples that pass threshold
            mask = confidence > adaptive_thresholds
            
            # Create pseudo-labels for selected samples
            pseudo_labels[mask] = F.one_hot(predicted_classes[mask], self.num_classes).float()
            
            # Update learning effects
            self.update_learning_effects(predicted_classes)
            self.unlabeled_predictions.extend(predicted_classes.cpu().numpy())
        
        return pseudo_labels, mask
    
    def compute_loss(
        self,
        labeled_batch: torch.Tensor,
        labeled_targets: torch.Tensor,
        unlabeled_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute FlexMatch loss
        
        Args:
            labeled_batch: Labeled images
            labeled_targets: True labels
            unlabeled_batch: Unlabeled images
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Supervised loss on labeled data
        labeled_outputs = self.model(labeled_batch)
        supervised_loss = F.cross_entropy(labeled_outputs, labeled_targets)
        
        # Unsupervised loss on unlabeled data
        pseudo_labels, mask = self.get_pseudo_labels(unlabeled_batch)
        
        if mask.sum() > 0:
            # Apply strong augmentation to selected samples
            strong_aug = self.strong_augment(unlabeled_batch[mask])
            strong_outputs = self.model(strong_aug)
            
            # Consistency loss
            unsupervised_loss = F.cross_entropy(strong_outputs, pseudo_labels[mask])
        else:
            unsupervised_loss = torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = supervised_loss + self.lambda_u * unsupervised_loss
        
        # Update adaptive factors
        self.compute_adaptive_factors()
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'supervised_loss': supervised_loss.item(),
            'unsupervised_loss': unsupervised_loss.item(),
            'mask_ratio': mask.float().mean().item(),
            'current_iteration': self.current_iteration
        }
        
        return total_loss, loss_dict
    
    def train_step(
        self,
        labeled_batch: torch.Tensor,
        labeled_targets: torch.Tensor,
        unlabeled_batch: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            labeled_batch: Labeled images
            labeled_targets: True labels
            unlabeled_batch: Unlabeled images
            
        Returns:
            Dictionary with loss information
        """
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute loss
        total_loss, loss_dict = self.compute_loss(
            labeled_batch, labeled_targets, unlabeled_batch
        )
        
        # Backward pass
        total_loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        # Update iteration counter
        self.current_iteration += 1
        
        return loss_dict
    
    def evaluate(self, test_loader: DataLoader) -> float:
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Test accuracy
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'sigma': self.sigma,
            'beta': self.beta,
            'thresholds': self.thresholds,
            'current_iteration': self.current_iteration,
            'unlabeled_predictions': self.unlabeled_predictions
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load model checkpoint
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.sigma = checkpoint['sigma']
        self.beta = checkpoint['beta']
        self.thresholds = checkpoint['thresholds']
        self.current_iteration = checkpoint['current_iteration']
        self.unlabeled_predictions = checkpoint['unlabeled_predictions']
        logger.info(f"Checkpoint loaded from {filepath}")
    
    def get_adaptive_thresholds(self) -> torch.Tensor:
        """
        Get current adaptive thresholds for each class
        
        Returns:
            Tensor of adaptive thresholds
        """
        return self.thresholds.clone()
    
    def get_learning_effects(self) -> torch.Tensor:
        """
        Get current learning effects for each class
        
        Returns:
            Tensor of learning effects
        """
        return self.sigma.clone() 