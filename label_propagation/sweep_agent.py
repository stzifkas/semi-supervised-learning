"""
Sweep Agent for Label Propagation Hyperparameter Optimization

This script can be run as a sweep agent to automatically test different
hyperparameter combinations.
"""

import wandb
from label_propagation_moons import LabelPropagation, generate_moons_dataset, plot_results
from sklearn.metrics import accuracy_score
import numpy as np
import os

def train_sweep():
    """
    Training function for sweep agent.
    """
    # Initialize wandb run
    with wandb.init() as run:
        config = wandb.config
        
        # Generate dataset
        X, y, y_true = generate_moons_dataset(
            n_samples=config.n_samples,
            noise=config.noise,
            labeled_ratio=config.labeled_ratio,
            random_state=42
        )
        
        # Create labeled mask
        labeled_mask = y != -1
        
        # Initialize and fit label propagation
        lp = LabelPropagation(
            alpha=config.alpha,
            max_iter=config.max_iter,
            tol=config.tol
        )
        
        lp.fit(X, y)
        
        # Get predictions
        y_pred = lp.labels_
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        labeled_accuracy = accuracy_score(y_true[labeled_mask], y_pred[labeled_mask])
        unlabeled_accuracy = accuracy_score(y_true[~labeled_mask], y_pred[~labeled_mask])
        
        # Log metrics
        wandb.log({
            "overall_accuracy": accuracy,
            "labeled_accuracy": labeled_accuracy,
            "unlabeled_accuracy": unlabeled_accuracy,
            "n_total_samples": len(X),
            "n_labeled_samples": np.sum(labeled_mask),
            "n_unlabeled_samples": np.sum(~labeled_mask)
        })
        
        # Create and log visualization (only for some runs to avoid clutter)
        if run.name.endswith('0'):  # Log every 10th run
            plot_path = f"sweep_results_{run.name}.png"
            plot_results(X, y_true, y_pred, labeled_mask, save_path=plot_path)
            wandb.log({"results_plot": wandb.Image(plot_path)})
            
            # Clean up
            if os.path.exists(plot_path):
                os.remove(plot_path)

if __name__ == "__main__":
    # This will be called by wandb agent
    train_sweep() 