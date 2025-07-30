"""
Virtual Adversarial Training (VAT) on CIFAR-10 for Semi-Supervised Learning

This implementation follows the paper "Virtual Adversarial Training: A Regularization Method 
for Supervised and Semi-Supervised Learning" by Miyato et al.

VAT Algorithm:
1. Forward pass: Get predictions p = softmax(model(x))
2. Find adversarial direction d that maximizes KL divergence
3. Apply perturbation r_vadv = ε * d / ||d||
4. Compute KL divergence between p and model(x + r_vadv)
5. Add to supervised loss and backpropagate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
import warnings
import wandb
from tqdm import tqdm

warnings.filterwarnings('ignore')

class CIFAR10CNN(nn.Module):
    """
    Simple CNN for CIFAR-10 classification.
    """
    
    def __init__(self, num_classes: int = 10):
        super(CIFAR10CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class VATTrainer:
    """
    Virtual Adversarial Training trainer for semi-supervised learning.
    """
    
    def __init__(self,
                 model: nn.Module,
                 vat_epsilon: float = 8.0,
                 vat_xi: float = 1e-6,
                 vat_iterations: int = 1,
                 vat_alpha: float = 1.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize VAT trainer.
        
        Args:
            model: Neural network model
            vat_epsilon: Maximum perturbation magnitude
            vat_xi: Small constant for numerical stability
            vat_iterations: Number of power iterations for finding adversarial direction
            vat_alpha: Weight for VAT loss
            device: Device to run training on
        """
        self.model = model.to(device)
        self.vat_epsilon = vat_epsilon
        self.vat_xi = vat_xi
        self.vat_iterations = vat_iterations
        self.vat_alpha = vat_alpha
        self.device = device
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def _kl_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between two probability distributions.
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            KL divergence
        """
        # Add small epsilon to avoid log(0)
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)
        
        # KL divergence: KL(p||q) = sum(p * log(p/q))
        kl_div = torch.sum(p * torch.log(p / (q + 1e-8) + 1e-8), dim=1)
        return kl_div
    
    def _get_virtual_adversarial_perturbation(self, x: torch.Tensor, 
                                            labeled_mask: torch.Tensor) -> torch.Tensor:
        # Initialize random unit vector
        d = torch.randn_like(x, device=self.device)
        d = d / torch.norm(d.view(x.size(0), -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
        
        # Get original predictions (fixed)
        with torch.no_grad():
            p_orig = F.softmax(self.model(x), dim=1)
        
        # Power iteration
        for _ in range(self.vat_iterations):
            # Create perturbation that requires gradient
            r = self.vat_xi * d
            r.requires_grad_(True)
            
            # Perturbed predictions
            x_perturbed = x + r
            p_perturbed = F.softmax(self.model(x_perturbed), dim=1)
            
            # KL divergence
            kl_div = self._kl_divergence(p_orig, p_perturbed).mean()
            
            # Compute gradient w.r.t. r and apply finite difference scaling
            kl_div.backward()
            d = r.grad.data / self.vat_xi  # Finite difference scaling
            d = d / torch.norm(d.view(x.size(0), -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
        
        # Final perturbation
        r_vadv = self.vat_epsilon * d
        return r_vadv.detach()  # Stop gradients   
     
    def compute_vat_loss(self, x: torch.Tensor, labeled_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute VAT loss for the batch.
        
        Args:
            x: Input images
            labeled_mask: Boolean mask indicating labeled samples
            
        Returns:
            VAT loss
        """
        # Get virtual adversarial perturbation
        r_vadv = self._get_virtual_adversarial_perturbation(x, labeled_mask)
        
        # Apply perturbation
        x_perturbed = x + r_vadv
        
        # Get predictions for original and perturbed inputs
        p_orig = F.softmax(self.model(x), dim=1)
        p_perturbed = F.softmax(self.model(x_perturbed), dim=1)
        
        # Compute KL divergence
        kl_div = self._kl_divergence(p_orig, p_perturbed)
        
        # Return mean KL divergence
        return torch.mean(kl_div)
    
    def train_epoch(self, 
                   train_loader: DataLoader,
                   optimizer: optim.Optimizer,
                   labeled_mask: torch.Tensor) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            labeled_mask: Boolean mask indicating labeled samples
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Get labeled mask for this batch
            batch_labeled_mask = labeled_mask[batch_idx * train_loader.batch_size:
                                            (batch_idx + 1) * train_loader.batch_size]
            
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Supervised loss (only for labeled samples)
            supervised_loss = F.cross_entropy(output[batch_labeled_mask], 
                                           target[batch_labeled_mask]) if torch.any(batch_labeled_mask) else torch.tensor(0.0, device=self.device)
            
            # VAT loss (for all samples)
            vat_loss = self.compute_vat_loss(data, batch_labeled_mask)
            
            # Total loss
            total_loss_batch = supervised_loss + self.vat_alpha * vat_loss
            
            # Backward pass
            total_loss_batch.backward()
            optimizer.step()
            
            # Statistics
            total_loss += total_loss_batch.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              labeled_mask: torch.Tensor,
              num_epochs: int = 100,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-4) -> Dict[str, List[float]]:
        """
        Train the model with VAT.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            labeled_mask: Boolean mask indicating labeled samples
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            
        Returns:
            Dictionary with training history
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        best_val_acc = 0.0
        
        for epoch in tqdm(range(num_epochs), desc="Training"):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, labeled_mask)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_vat_model.pth')
            
            # Log metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
            
            # Print progress
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }

def load_cifar10_subset(labeled_ratio: float = 0.1, 
                        num_samples: int = 5000,
                        batch_size: int = 64,
                        random_state: int = 42) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Load CIFAR-10 subset with partial labeling for semi-supervised learning.
    
    Args:
        labeled_ratio: Ratio of labeled samples
        num_samples: Total number of samples to use
        batch_size: Batch size for data loaders
        random_state: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, labeled_mask)
    """
    # Set random seed
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load CIFAR-10
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    
    # Create subset
    indices = torch.randperm(len(train_dataset))[:num_samples]
    train_subset = Subset(train_dataset, indices)
    
    # Create labeled mask
    n_labeled = int(num_samples * labeled_ratio)
    labeled_indices = torch.randperm(num_samples)[:n_labeled]
    labeled_mask = torch.zeros(num_samples, dtype=torch.bool)
    labeled_mask[labeled_indices] = True
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, labeled_mask

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and validation loss
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Training and validation accuracy
    ax2.plot(history['train_accuracies'], label='Train Accuracy')
    ax2.plot(history['val_accuracies'], label='Val Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Learning curves
    ax3.semilogy(history['train_losses'], label='Train Loss')
    ax3.semilogy(history['val_losses'], label='Val Loss')
    ax3.set_title('Learning Curves (Log Scale)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (log scale)')
    ax3.legend()
    ax3.grid(True)
    
    # Accuracy comparison
    ax4.plot(history['train_accuracies'], label='Train', alpha=0.7)
    ax4.plot(history['val_accuracies'], label='Validation', alpha=0.7)
    ax4.set_title('Accuracy Comparison')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def main():
    """Main function to run VAT experiment on CIFAR-10."""
    print("=== Virtual Adversarial Training on CIFAR-10 ===\n")
    
    # Initialize W&B
    wandb.init(
        project="vat-cifar10",
        config={
            "labeled_ratio": 0.1,
            "num_samples": 5000,
            "batch_size": 64,
            "num_epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "vat_epsilon": 8.0,
            "vat_xi": 1e-6,
            "vat_iterations": 1,
            "vat_alpha": 1.0,
            "random_state": 42
        }
    )
    
    config = wandb.config
    
    # Load dataset
    print("Loading CIFAR-10 subset...")
    train_loader, val_loader, labeled_mask = load_cifar10_subset(
        labeled_ratio=config.labeled_ratio,
        num_samples=config.num_samples,
        batch_size=config.batch_size,
        random_state=config.random_state
    )
    
    print(f"Dataset: {config.num_samples} samples, {torch.sum(labeled_mask)} labeled, "
          f"{torch.sum(~labeled_mask)} unlabeled")
    
    # Initialize model and trainer
    model = CIFAR10CNN(num_classes=10)
    trainer = VATTrainer(
        model=model,
        vat_epsilon=config.vat_epsilon,
        vat_xi=config.vat_xi,
        vat_iterations=config.vat_iterations,
        vat_alpha=config.vat_alpha
    )
    
    # Train model
    print("Training with VAT...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        labeled_mask=labeled_mask,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Plot results
    plot_training_history(history, save_path="vat_training_history.png")
    
    # Final evaluation
    final_val_loss, final_val_acc = trainer.validate(val_loader)
    print(f"\nFinal Results:")
    print(f"Validation Loss: {final_val_loss:.4f}")
    print(f"Validation Accuracy: {final_val_acc:.2f}%")
    
    # Log final metrics
    wandb.log({
        "final_val_loss": final_val_loss,
        "final_val_accuracy": final_val_acc
    })
    
    print("Experiment completed!")

if __name__ == "__main__":
    main() 