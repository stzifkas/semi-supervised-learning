"""
Label Propagation on 2D Moons Dataset with Weights & Biases Integration

This script implements label propagation algorithm on a 2D moons dataset
and tracks experiments using Weights & Biases.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
import wandb
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class LabelPropagation:
    """
    Label Propagation algorithm implementation for semi-supervised learning.
    """
    
    def __init__(self, alpha: float = 0.99, max_iter: int = 1000, tol: float = 1e-3):
        """
        Initialize Label Propagation algorithm.
        
        Args:
            alpha: Clamping factor (0 < alpha < 1)
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.labels_ = None
        self.probabilities_ = None
        
    def _build_affinity_matrix(self, X: np.ndarray, n_neighbors: int = 7) -> np.ndarray:
        """
        Build affinity matrix using k-nearest neighbors.
        
        Args:
            X: Input features
            n_neighbors: Number of neighbors for k-NN
            
        Returns:
            Affinity matrix
        """
        n_samples = X.shape[0]
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Remove self-connections
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        # Compute Gaussian similarity
        sigma = np.mean(distances)
        similarities = np.exp(-distances**2 / (2 * sigma**2))
        
        # Build sparse affinity matrix
        affinity_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            affinity_matrix[i, indices[i]] = similarities[i]
            affinity_matrix[indices[i], i] = similarities[i]  # Make symmetric
        
        # Normalize
        row_sums = affinity_matrix.sum(axis=1)
        affinity_matrix = affinity_matrix / row_sums[:, np.newaxis]
        
        return affinity_matrix
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LabelPropagation':
        """
        Fit the label propagation model.
        
        Args:
            X: Input features
            y: Labels (use -1 for unlabeled samples)
            
        Returns:
            self
        """
        n_samples = X.shape[0]
        n_classes = len(np.unique(y[y != -1]))
        
        # Build affinity matrix
        affinity_matrix = self._build_affinity_matrix(X)
        
        # Initialize label matrix
        Y = np.zeros((n_samples, n_classes))
        for i, label in enumerate(y):
            if label != -1:
                Y[i, label] = 1
        
        # Label propagation iterations
        Y_prev = Y.copy()
        
        for iteration in range(self.max_iter):
            # Propagate labels
            Y_new = self.alpha * affinity_matrix @ Y_prev + (1 - self.alpha) * Y
            
            # Check convergence
            if np.linalg.norm(Y_new - Y_prev) < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
                
            Y_prev = Y_new.copy()
        
        # Store results
        self.labels_ = np.argmax(Y_new, axis=1)
        self.probabilities_ = Y_new
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for new samples.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        # For simplicity, we'll use nearest neighbor prediction
        nbrs = NearestNeighbors(n_neighbors=1).fit(X)
        distances, indices = nbrs.kneighbors(X)
        return self.labels_[indices.flatten()]

def generate_moons_dataset(n_samples: int = 1000, noise: float = 0.1, 
                          labeled_ratio: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D moons dataset with partial labeling.
    
    Args:
        n_samples: Total number of samples
        noise: Noise level for the moons
        labeled_ratio: Ratio of labeled samples
        random_state: Random seed
        
    Returns:
        X: Features, y: Labels (with -1 for unlabeled)
    """
    # Generate moons dataset
    X, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    # Create partially labeled dataset
    n_labeled = int(n_samples * labeled_ratio)
    labeled_indices = np.random.choice(n_samples, n_labeled, replace=False)
    
    y = np.full(n_samples, -1)  # -1 indicates unlabeled
    y[labeled_indices] = y_true[labeled_indices]
    
    return X, y, y_true

def plot_results(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, 
                labeled_mask: np.ndarray, iteration: int = 0, save_path: Optional[str] = None):
    """
    Plot the results of label propagation.
    
    Args:
        X: Input features
        y_true: True labels
        y_pred: Predicted labels
        labeled_mask: Boolean mask for labeled samples
        iteration: Current iteration number
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Plot 1: True labels
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
    plt.title('True Labels')
    plt.colorbar(scatter)
    
    # Plot 2: Labeled vs Unlabeled
    plt.subplot(1, 3, 2)
    colors = ['red' if labeled else 'lightgray' for labeled in labeled_mask]
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6)
    plt.title('Labeled (Red) vs Unlabeled (Gray)')
    
    # Plot 3: Predicted labels
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    plt.title(f'Predicted Labels (Iteration {iteration})')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    """
    Main function to run label propagation experiment with W&B logging.
    """
    # Initialize W&B
    wandb.init(
        project="label-propagation-moons",
        config={
            "n_samples": 1000,
            "noise": 0.1,
            "labeled_ratio": 0.1,
            "alpha": 0.99,
            "max_iter": 1000,
            "tol": 1e-3,
            "n_neighbors": 7,
            "random_state": 42
        }
    )
    
    config = wandb.config
    
    # Generate dataset
    print("Generating 2D moons dataset...")
    X, y, y_true = generate_moons_dataset(
        n_samples=config.n_samples,
        noise=config.noise,
        labeled_ratio=config.labeled_ratio,
        random_state=config.random_state
    )
    
    # Create labeled mask
    labeled_mask = y != -1
    
    # Log dataset statistics
    wandb.log({
        "n_total_samples": len(X),
        "n_labeled_samples": np.sum(labeled_mask),
        "n_unlabeled_samples": np.sum(~labeled_mask),
        "labeled_ratio_actual": np.sum(labeled_mask) / len(X)
    })
    
    # Initialize and fit label propagation
    print("Fitting label propagation model...")
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
        "unlabeled_accuracy": unlabeled_accuracy
    })
    
    # Print results
    print(f"\nResults:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Labeled Samples Accuracy: {labeled_accuracy:.4f}")
    print(f"Unlabeled Samples Accuracy: {unlabeled_accuracy:.4f}")
    
    # Create visualization
    print("\nCreating visualization...")
    plot_results(X, y_true, y_pred, labeled_mask, save_path="label_propagation_results.png")
    
    # Log the plot to W&B
    wandb.log({"results_plot": wandb.Image("label_propagation_results.png")})
    
    # Create detailed classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    wandb.log({"classification_report": report})
    
    # Log confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        probs=None, 
        y_true=y_true, 
        preds=y_pred,
        class_names=["Class 0", "Class 1"]
    )})
    
    # Log feature importance (if applicable)
    if hasattr(lp, 'probabilities_'):
        # Log probability distributions
        prob_df = pd.DataFrame(lp.probabilities_, columns=['prob_class_0', 'prob_class_1'])
        prob_df['true_label'] = y_true
        prob_df['predicted_label'] = y_pred
        prob_df['is_labeled'] = labeled_mask
        
        wandb.log({"probability_distributions": wandb.Table(dataframe=prob_df)})
    
    print("\nExperiment completed! Check W&B dashboard for detailed results.")
    wandb.finish()

if __name__ == "__main__":
    main() 