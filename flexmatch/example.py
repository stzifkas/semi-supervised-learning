#!/usr/bin/env python3
"""
Simple example demonstrating FlexMatch on CIFAR-10

This script shows how to use FlexMatch with a small number of epochs
for quick testing and understanding of the algorithm.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from flexmatch import FlexMatch
from models import create_model
from data_utils import create_data_loaders, CIFAR10_CLASSES
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_flexmatch_example():
    """Run a simple FlexMatch example"""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create data loaders with small dataset for quick testing
    logger.info("Creating data loaders...")
    labeled_loader, unlabeled_loader, val_loader, test_loader = create_data_loaders(
        labeled_samples=1000,  # Small number for quick testing
        val_samples=1000,
        labeled_batch_size=32,
        unlabeled_batch_size=64
    )
    
    # Create model
    logger.info("Creating model...")
    model = create_model(num_classes=10, model_name='wrn_28_2')
    
    # Create FlexMatch trainer
    logger.info("Initializing FlexMatch...")
    flexmatch = FlexMatch(
        model=model,
        num_classes=10,
        base_threshold=0.95,
        warmup_iterations=1000,  # Small warmup for quick testing
        lambda_u=1.0,
        device=device
    )
    
    # Training loop (small number of epochs for demonstration)
    epochs = 10
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        flexmatch.model.train()
        
        # Create iterators
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        # Number of steps (use smaller for demonstration)
        steps = min(10, len(labeled_loader), len(unlabeled_loader))
        
        for step in range(steps):
            # Get batches
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
            
            if step % 5 == 0:
                logger.info(f"  Step {step}: Loss={loss_dict['total_loss']:.4f}, "
                          f"Mask={loss_dict['mask_ratio']:.3f}")
        
        # Evaluation
        if epoch % 2 == 0 or epoch == epochs - 1:
            val_acc = flexmatch.evaluate(val_loader)
            test_acc = flexmatch.evaluate(test_loader)
            
            # Get current adaptive thresholds
            adaptive_thresholds = flexmatch.get_adaptive_thresholds().cpu().numpy()
            learning_effects = flexmatch.get_learning_effects().cpu().numpy()
            
            logger.info(f"  Epoch {epoch+1}: Val Acc={val_acc:.2f}%, Test Acc={test_acc:.2f}%")
            logger.info(f"  Adaptive thresholds: {adaptive_thresholds}")
            logger.info(f"  Learning effects: {learning_effects}")
    
    # Final evaluation
    logger.info("Final evaluation...")
    final_val_acc = flexmatch.evaluate(val_loader)
    final_test_acc = flexmatch.evaluate(test_loader)
    
    logger.info(f"Final results:")
    logger.info(f"  Validation accuracy: {final_val_acc:.2f}%")
    logger.info(f"  Test accuracy: {final_test_acc:.2f}%")
    
    # Plot final adaptive thresholds
    final_thresholds = flexmatch.get_adaptive_thresholds().cpu().numpy()
    final_learning_effects = flexmatch.get_learning_effects().cpu().numpy()
    
    plt.figure(figsize=(12, 5))
    
    # Plot adaptive thresholds
    plt.subplot(1, 2, 1)
    plt.bar(range(10), final_thresholds)
    plt.xlabel('Class')
    plt.ylabel('Adaptive Threshold')
    plt.title('Final Adaptive Thresholds')
    plt.xticks(range(10), CIFAR10_CLASSES, rotation=45)
    plt.grid(True)
    
    # Plot learning effects
    plt.subplot(1, 2, 2)
    plt.bar(range(10), final_learning_effects)
    plt.xlabel('Class')
    plt.ylabel('Learning Effect')
    plt.title('Final Learning Effects')
    plt.xticks(range(10), CIFAR10_CLASSES, rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('flexmatch_example_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("Example completed! Results saved to flexmatch_example_results.png")

def demonstrate_curriculum_pseudo_labeling():
    """Demonstrate how Curriculum Pseudo Labeling works"""
    
    logger.info("Demonstrating Curriculum Pseudo Labeling...")
    
    # Create a simple example
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(num_classes=10, model_name='wrn_28_2')
    flexmatch = FlexMatch(model=model, device=device)
    
    # Simulate some predictions
    logger.info("Simulating predictions for different classes...")
    
    # Example: Class 0 (airplane) is easy, Class 9 (truck) is hard
    predictions = []
    for i in range(100):
        if i < 50:
            # Class 0 predictions (easy class)
            pred = torch.tensor([0.8, 0.1, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005])
        else:
            # Class 9 predictions (hard class)
            pred = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6])
        predictions.append(pred)
    
    predictions = torch.stack(predictions)
    
    # Get pseudo-labels with adaptive thresholds
    logger.info("Computing adaptive thresholds...")
    flexmatch.compute_adaptive_factors()
    adaptive_thresholds = flexmatch.get_adaptive_thresholds()
    
    logger.info(f"Base threshold: {flexmatch.base_threshold}")
    logger.info(f"Adaptive thresholds: {adaptive_thresholds.cpu().numpy()}")
    
    # Show how different classes get different thresholds
    for class_id in range(10):
        threshold = adaptive_thresholds[class_id].item()
        logger.info(f"Class {class_id} ({CIFAR10_CLASSES[class_id]}): threshold = {threshold:.3f}")
    
    logger.info("Curriculum Pseudo Labeling demonstration completed!")

if __name__ == '__main__':
    logger.info("Running FlexMatch example...")
    
    # Run the main example
    run_flexmatch_example()
    
    # Demonstrate CPL
    demonstrate_curriculum_pseudo_labeling()
    
    logger.info("All examples completed!") 