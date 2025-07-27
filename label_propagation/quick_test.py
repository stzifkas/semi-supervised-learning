"""
Quick test script for label propagation without W&B integration.
Useful for testing the algorithm before running full experiments.
"""

import numpy as np
from label_propagation_moons import LabelPropagation, generate_moons_dataset, plot_results
from sklearn.metrics import accuracy_score

def quick_test():
    """
    Run a quick test of the label propagation algorithm.
    """
    print("Running quick test of label propagation on 2D moons...")
    
    # Generate small dataset for quick testing
    X, y, y_true = generate_moons_dataset(
        n_samples=200,
        noise=0.1,
        labeled_ratio=0.2,
        random_state=42
    )
    
    # Create labeled mask
    labeled_mask = y != -1
    
    print(f"Dataset: {len(X)} samples, {np.sum(labeled_mask)} labeled, {np.sum(~labeled_mask)} unlabeled")
    
    # Initialize and fit label propagation
    lp = LabelPropagation(alpha=0.99, max_iter=100, tol=1e-3)
    lp.fit(X, y)
    
    # Get predictions
    y_pred = lp.labels_
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    labeled_accuracy = accuracy_score(y_true[labeled_mask], y_pred[labeled_mask])
    unlabeled_accuracy = accuracy_score(y_true[~labeled_mask], y_pred[~labeled_mask])
    
    print(f"\nResults:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Labeled Samples Accuracy: {labeled_accuracy:.4f}")
    print(f"Unlabeled Samples Accuracy: {unlabeled_accuracy:.4f}")
    
    # Create visualization
    plot_results(X, y_true, y_pred, labeled_mask, save_path="quick_test_results.png")
    print(f"\nVisualization saved as 'quick_test_results.png'")
    
    return accuracy

if __name__ == "__main__":
    quick_test() 