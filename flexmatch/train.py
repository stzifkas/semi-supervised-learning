#!/usr/bin/env python3
"""
FlexMatch Training Script for CIFAR-10

This script demonstrates how to train a FlexMatch model on CIFAR-10
with semi-supervised learning using Curriculum Pseudo Labeling (CPL).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import logging
from tqdm import tqdm
import json
from datetime import datetime

# Import our modules
from flexmatch import FlexMatch
from models import create_model
from data_utils import create_data_loaders, CIFAR10_CLASSES

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_flexmatch(
    model_name: str = 'wrn_28_2',
    labeled_samples: int = 4000,
    val_samples: int = 5000,
    epochs: int = 1024,
    labeled_batch_size: int = 64,
    unlabeled_batch_size: int = 128,
    base_threshold: float = 0.95,
    warmup_iterations: int = 40000,
    lambda_u: float = 1.0,
    learning_rate: float = 0.03,
    weight_decay: float = 5e-4,
    save_dir: str = './checkpoints',
    log_interval: int = 100,
    eval_interval: int = 500,
    device: str = None
) -> dict:
    """
    Train FlexMatch model on CIFAR-10
    
    Args:
        model_name: Model architecture to use
        labeled_samples: Number of labeled samples
        val_samples: Number of validation samples
        epochs: Number of training epochs
        labeled_batch_size: Batch size for labeled data
        unlabeled_batch_size: Batch size for unlabeled data
        base_threshold: Base confidence threshold
        warmup_iterations: Number of warm-up iterations
        lambda_u: Weight for unsupervised loss
        learning_rate: Learning rate
        weight_decay: Weight decay
        save_dir: Directory to save checkpoints
        log_interval: Logging interval
        eval_interval: Evaluation interval
        device: Device to use
        
    Returns:
        Dictionary with training results
    """
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    labeled_loader, unlabeled_loader, val_loader, test_loader = create_data_loaders(
        labeled_batch_size=labeled_batch_size,
        unlabeled_batch_size=unlabeled_batch_size,
        val_batch_size=128,
        test_batch_size=128,
        labeled_samples=labeled_samples,
        val_samples=val_samples
    )
    
    # Create model
    logger.info(f"Creating model: {model_name}")
    model = create_model(num_classes=10, model_name=model_name)
    
    # Create FlexMatch trainer
    logger.info("Initializing FlexMatch...")
    flexmatch = FlexMatch(
        model=model,
        num_classes=10,
        base_threshold=base_threshold,
        warmup_iterations=warmup_iterations,
        lambda_u=lambda_u,
        device=device
    )
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'supervised_loss': [],
        'unsupervised_loss': [],
        'val_accuracy': [],
        'test_accuracy': [],
        'mask_ratio': [],
        'adaptive_thresholds': [],
        'learning_effects': []
    }
    
    # Training loop
    logger.info("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        flexmatch.model.train()
        epoch_losses = []
        epoch_supervised_losses = []
        epoch_unsupervised_losses = []
        epoch_mask_ratios = []
        
        # Create iterators for labeled and unlabeled data
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        # Number of steps per epoch (use the smaller of the two)
        steps_per_epoch = min(len(labeled_loader), len(unlabeled_loader))
        
        progress_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
        
        for step in progress_bar:
            try:
                labeled_batch, labeled_targets = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_batch, labeled_targets = next(labeled_iter)
            
            try:
                unlabeled_batch, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch, _ = next(unlabeled_iter)
            
            # Move to device
            labeled_batch = labeled_batch.to(device)
            labeled_targets = labeled_targets.to(device)
            unlabeled_batch = unlabeled_batch.to(device)
            
            # Training step
            loss_dict = flexmatch.train_step(labeled_batch, labeled_targets, unlabeled_batch)
            
            # Record losses
            epoch_losses.append(loss_dict['total_loss'])
            epoch_supervised_losses.append(loss_dict['supervised_loss'])
            epoch_unsupervised_losses.append(loss_dict['unsupervised_loss'])
            epoch_mask_ratios.append(loss_dict['mask_ratio'])
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss_dict['total_loss']:.4f}",
                'Mask': f"{loss_dict['mask_ratio']:.3f}",
                'Iter': loss_dict['current_iteration']
            })
            
            # Logging
            if step % log_interval == 0:
                logger.info(f"Epoch {epoch+1}, Step {step}: "
                          f"Loss={loss_dict['total_loss']:.4f}, "
                          f"Mask={loss_dict['mask_ratio']:.3f}")
        
        # Evaluation
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            logger.info("Evaluating model...")
            
            # Validation accuracy
            val_acc = flexmatch.evaluate(val_loader)
            
            # Test accuracy
            test_acc = flexmatch.evaluate(test_loader)
            
            # Get current adaptive thresholds and learning effects
            adaptive_thresholds = flexmatch.get_adaptive_thresholds().cpu().numpy()
            learning_effects = flexmatch.get_learning_effects().cpu().numpy()
            
            logger.info(f"Epoch {epoch+1}: "
                      f"Val Acc={val_acc:.2f}%, "
                      f"Test Acc={test_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                flexmatch.save_checkpoint(os.path.join(save_dir, 'best_model.pth'))
                logger.info(f"New best model saved! Val Acc: {val_acc:.2f}%")
            
            # Record history
            history['val_accuracy'].append(val_acc)
            history['test_accuracy'].append(test_acc)
            history['adaptive_thresholds'].append(adaptive_thresholds.tolist())
            history['learning_effects'].append(learning_effects.tolist())
        
        # Record epoch statistics
        history['train_loss'].append(np.mean(epoch_losses))
        history['supervised_loss'].append(np.mean(epoch_supervised_losses))
        history['unsupervised_loss'].append(np.mean(epoch_unsupervised_losses))
        history['mask_ratio'].append(np.mean(epoch_mask_ratios))
        
        # Save checkpoint periodically
        if epoch % 100 == 0:
            flexmatch.save_checkpoint(os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # Save final model
    flexmatch.save_checkpoint(os.path.join(save_dir, 'final_model.pth'))
    
    # Save training history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return history

def plot_training_results(history: dict, save_dir: str = './plots'):
    """
    Plot training results
    
    Args:
        history: Training history dictionary
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Total Loss')
    plt.plot(history['supervised_loss'], label='Supervised Loss')
    plt.plot(history['unsupervised_loss'], label='Unsupervised Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(2, 3, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history['test_accuracy'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Mask ratio plot
    plt.subplot(2, 3, 3)
    plt.plot(history['mask_ratio'])
    plt.xlabel('Epoch')
    plt.ylabel('Mask Ratio')
    plt.title('Pseudo-label Selection Ratio')
    plt.grid(True)
    
    # Adaptive thresholds plot
    plt.subplot(2, 3, 4)
    thresholds = np.array(history['adaptive_thresholds'])
    for i in range(10):
        plt.plot(thresholds[:, i], label=f'Class {i}')
    plt.xlabel('Epoch')
    plt.ylabel('Threshold')
    plt.title('Adaptive Thresholds')
    plt.legend()
    plt.grid(True)
    
    # Learning effects plot
    plt.subplot(2, 3, 5)
    learning_effects = np.array(history['learning_effects'])
    for i in range(10):
        plt.plot(learning_effects[:, i], label=f'Class {i}')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Effect')
    plt.title('Class Learning Effects')
    plt.legend()
    plt.grid(True)
    
    # Final thresholds vs class names
    plt.subplot(2, 3, 6)
    final_thresholds = history['adaptive_thresholds'][-1]
    plt.bar(range(10), final_thresholds)
    plt.xlabel('Class')
    plt.ylabel('Final Threshold')
    plt.title('Final Adaptive Thresholds')
    plt.xticks(range(10), CIFAR10_CLASSES, rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train FlexMatch on CIFAR-10')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='wrn_28_2',
                       choices=['wrn_28_2', 'wrn_28_10', 'wrn_40_2', 'wrn_40_10'],
                       help='Model architecture')
    
    # Data parameters
    parser.add_argument('--labeled-samples', type=int, default=4000,
                       help='Number of labeled samples')
    parser.add_argument('--val-samples', type=int, default=5000,
                       help='Number of validation samples')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1024,
                       help='Number of training epochs')
    parser.add_argument('--labeled-batch-size', type=int, default=64,
                       help='Batch size for labeled data')
    parser.add_argument('--unlabeled-batch-size', type=int, default=128,
                       help='Batch size for unlabeled data')
    
    # FlexMatch parameters
    parser.add_argument('--base-threshold', type=float, default=0.95,
                       help='Base confidence threshold')
    parser.add_argument('--warmup-iterations', type=int, default=40000,
                       help='Number of warm-up iterations')
    parser.add_argument('--lambda-u', type=float, default=1.0,
                       help='Weight for unsupervised loss')
    
    # Other parameters
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='Logging interval')
    parser.add_argument('--eval-interval', type=int, default=500,
                       help='Evaluation interval')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--plot', action='store_true',
                       help='Plot training results')
    
    args = parser.parse_args()
    
    # Create timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"flexmatch_{timestamp}")
    
    logger.info("Starting FlexMatch training...")
    logger.info(f"Arguments: {vars(args)}")
    
    # Train model
    history = train_flexmatch(
        model_name=args.model,
        labeled_samples=args.labeled_samples,
        val_samples=args.val_samples,
        epochs=args.epochs,
        labeled_batch_size=args.labeled_batch_size,
        unlabeled_batch_size=args.unlabeled_batch_size,
        base_threshold=args.base_threshold,
        warmup_iterations=args.warmup_iterations,
        lambda_u=args.lambda_u,
        save_dir=save_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        device=args.device
    )
    
    # Plot results if requested
    if args.plot:
        plot_training_results(history, save_dir)
    
    logger.info(f"Training completed! Results saved to: {save_dir}")

if __name__ == '__main__':
    main() 