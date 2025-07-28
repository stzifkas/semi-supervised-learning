"""
Maximum Likelihood Predictor with Entropy Loss - FIXED VERSION
Following the paper: "Semi-supervised Learning by Entropy Minimization"
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict
import warnings
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

class MaximumLikelihoodPredictor:
    """
    Maximum Likelihood Predictor with Entropy Loss for semi-supervised learning.
    
    This implementation follows the paper "Semi-supervised Learning by Entropy Minimization"
    by Grandvalet and Bengio.
    """
    
    def __init__(self, 
                 entropy_weight: float = 0.1,
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 regularization: float = 0.01,
                 random_state: int = 42):
        """
        Initialize Maximum Likelihood Predictor with entropy loss.
        
        Args:
            entropy_weight: Lambda parameter - weight for entropy regularization
            max_iter: Maximum iterations for optimization
            tol: Convergence tolerance
            regularization: L2 regularization for smoothness
            random_state: Random seed
        """
        self.entropy_weight = entropy_weight
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization
        self.random_state = random_state
        
        # Model parameters
        self.weights_ = None
        self.bias_ = None
        self.scaler_ = StandardScaler()
        
        # Training data
        self.X_train_ = None
        self.y_train_ = None
        self.labeled_mask_ = None
        self.probabilities_ = None
        self.labels_ = None
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Stable sigmoid function."""
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def _predict_proba_logistic(self, X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
        """
        Predict probabilities using logistic regression.
        
        Args:
            X: Input features [n_samples, n_features]
            weights: Weight vector [n_features]
            bias: Bias term
            
        Returns:
            Probabilities [n_samples, 2]
        """
        # Compute logits
        logits = X @ weights + bias
        
        # Convert to probabilities for binary classification
        p1 = self._sigmoid(logits)
        p0 = 1 - p1
        
        return np.column_stack([p0, p1])
    
    def _compute_g_k(self, probabilities: np.ndarray, y: np.ndarray, labeled_mask: np.ndarray) -> np.ndarray:
        """
        Compute g_k as defined in the paper.
        
        For labeled data: g_k = z_k (one-hot encoding)
        For unlabeled data: g_k = f_k (model probabilities)
        
        Args:
            probabilities: Model probabilities [n_samples, n_classes]
            y: Labels (with -1 for unlabeled)
            labeled_mask: Boolean mask for labeled samples
            
        Returns:
            g_k values [n_samples, n_classes]
        """
        n_samples, n_classes = probabilities.shape
        g_k = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            if labeled_mask[i]:
                # Labeled: g_k = z_k (one-hot)
                g_k[i, y[i]] = 1.0
            else:
                # Unlabeled: g_k = f_k (model probabilities)
                g_k[i] = probabilities[i]
        
        return g_k
    
    def _compute_criterion(self, params: np.ndarray, X: np.ndarray, y: np.ndarray, 
                          labeled_mask: np.ndarray) -> float:
        """
        Compute the criterion C(θ, λ) = L(θ) - λ * H_emp(θ) from the paper.
        
        Args:
            params: Parameters [weights..., bias]
            X: Input features
            y: Labels (with -1 for unlabeled)
            labeled_mask: Boolean mask for labeled samples
            
        Returns:
            Negative criterion (for minimization)
        """
        # Extract parameters
        n_features = X.shape[1]
        weights = params[:n_features]
        bias = params[n_features]
        
        # Get probabilities
        probabilities = self._predict_proba_logistic(X, weights, bias)
        
        # Compute likelihood term L(θ) - only on labeled data
        likelihood = 0.0
        n_labeled = np.sum(labeled_mask)
        
        if n_labeled > 0:
            labeled_probs = probabilities[labeled_mask]
            labeled_y = y[labeled_mask]
            
            # Log-likelihood of labeled samples
            log_probs = np.log(np.clip(labeled_probs[range(n_labeled), labeled_y], 1e-10, 1.0))
            likelihood = np.mean(log_probs)
        
        # Compute entropy term H_emp(θ) - on all data
        g_k = self._compute_g_k(probabilities, y, labeled_mask)
        
        # Empirical entropy: -1/n * sum(g_k * log(g_k))
        n_samples = len(X)
        g_k_clipped = np.clip(g_k, 1e-10, 1.0)
        entropy = -(1/n_samples) * np.sum(g_k * np.log(g_k_clipped))
        
        # Regularization term (smoothness constraint from paper)
        l2_reg = 0.5 * self.regularization * np.sum(weights**2)
        
        # Paper's criterion: C = L - λ*H - regularization
        criterion = likelihood - self.entropy_weight * entropy - l2_reg
        
        # Return negative for minimization
        return -criterion
    
    def fit(self, X: np.ndarray, y: np.ndarray, labeled_mask: np.ndarray) -> 'MaximumLikelihoodPredictor':
        """
        Fit the maximum likelihood predictor with entropy regularization.
        
        Args:
            X: Input features
            y: Labels (use -1 for unlabeled samples)
            labeled_mask: Boolean mask for labeled samples
            
        Returns:
            self
        """
        # Store training data
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.labeled_mask_ = labeled_mask.copy()
        
        # Standardize features
        X_scaled = self.scaler_.fit_transform(X)
        
        print(f"Fitting Maximum Likelihood Predictor with entropy weight λ={self.entropy_weight}")
        print(f"Labeled samples: {np.sum(labeled_mask)}, Unlabeled samples: {np.sum(~labeled_mask)}")
        
        # Initialize parameters with supervised logistic regression on labeled data
        if np.sum(labeled_mask) > 0:
            X_labeled = X_scaled[labeled_mask]
            y_labeled = y[labeled_mask]
            
            # Initial fit with sklearn
            initial_lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
            initial_lr.fit(X_labeled, y_labeled)
            
            # Extract parameters
            initial_weights = initial_lr.coef_[0]
            initial_bias = initial_lr.intercept_[0]
        else:
            # Random initialization if no labeled data
            n_features = X_scaled.shape[1]
            initial_weights = np.random.normal(0, 0.1, n_features)
            initial_bias = 0.0
        
        # Combine parameters for optimization
        initial_params = np.concatenate([initial_weights, [initial_bias]])
        
        print(f"Initial parameters: weights shape {initial_weights.shape}, bias {initial_bias:.4f}")
        
        # Optimize the criterion using scipy.minimize
        print("Optimizing criterion...")
        result = minimize(
            fun=self._compute_criterion,
            x0=initial_params,
            args=(X_scaled, y, labeled_mask),
            method='L-BFGS-B',
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge. Success: {result.success}")
            print(f"Message: {result.message}")
        else:
            print("Optimization converged successfully!")
        
        # Extract final parameters
        n_features = X_scaled.shape[1]
        self.weights_ = result.x[:n_features]
        self.bias_ = result.x[n_features]
        
        print(f"Final parameters: weights norm {np.linalg.norm(self.weights_):.4f}, bias {self.bias_:.4f}")
        
        # Get final predictions and probabilities
        self.probabilities_ = self._predict_proba_logistic(X_scaled, self.weights_, self.bias_)
        self.labels_ = np.argmax(self.probabilities_, axis=1)
        
        # Compute final statistics
        final_criterion = -self._compute_criterion(result.x, X_scaled, y, labeled_mask)
        print(f"Final criterion value: {final_criterion:.4f}")
        
        if np.sum(~labeled_mask) > 0:
            unlabeled_entropy = self._compute_entropy(self.probabilities_[~labeled_mask])
            print(f"Final entropy on unlabeled samples: {unlabeled_entropy:.4f}")
        
        return self
    
    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict on new data."""
        if self.weights_ is None:
            raise ValueError("Model must be fitted first")
        
        X_scaled = self.scaler_.transform(X_new)
        probabilities = self._predict_proba_logistic(X_scaled, self.weights_, self.bias_)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X_new: np.ndarray) -> np.ndarray:
        """Predict probabilities on new data."""
        if self.weights_ is None:
            raise ValueError("Model must be fitted first")
        
        X_scaled = self.scaler_.transform(X_new)
        return self._predict_proba_logistic(X_scaled, self.weights_, self.bias_)
    
    def _compute_entropy(self, probabilities: np.ndarray) -> float:
        """Compute entropy of probability distributions."""
        eps = 1e-10
        probs = np.clip(probabilities, eps, 1 - eps)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        return np.mean(entropy)
    
    def get_entropy(self) -> Dict[str, float]:
        """Get entropy statistics for the fitted model."""
        if self.probabilities_ is None:
            raise ValueError("Model must be fitted first")
        
        labeled_entropy = self._compute_entropy(self.probabilities_[self.labeled_mask_])
        unlabeled_entropy = self._compute_entropy(self.probabilities_[~self.labeled_mask_])
        total_entropy = self._compute_entropy(self.probabilities_)
        
        return {
            'labeled_entropy': labeled_entropy,
            'unlabeled_entropy': unlabeled_entropy,
            'total_entropy': total_entropy
        }

def generate_moons_dataset(n_samples: int = 1000, noise: float = 0.1, 
                          labeled_ratio: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate 2D moons dataset with partial labeling."""
    np.random.seed(random_state)
    
    # Generate moons dataset
    X, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    # Create partially labeled dataset
    n_labeled = int(n_samples * labeled_ratio)
    labeled_indices = np.random.choice(n_samples, n_labeled, replace=False)
    
    y = np.full(n_samples, -1)  # -1 indicates unlabeled
    y[labeled_indices] = y_true[labeled_indices]
    
    return X, y, y_true

def plot_results(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, 
                labeled_mask: np.ndarray, probabilities: np.ndarray = None,
                save_path: Optional[str] = None):
    """Plot the results of maximum likelihood prediction with entropy."""
    n_plots = 4 if probabilities is not None else 3
    plt.figure(figsize=(5 * n_plots, 5))
    
    # Plot 1: True labels
    plt.subplot(1, n_plots, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
    plt.title('True Labels')
    plt.colorbar(scatter)
    
    # Plot 2: Labeled vs Unlabeled
    plt.subplot(1, n_plots, 2)
    colors = ['red' if labeled else 'lightgray' for labeled in labeled_mask]
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6)
    plt.title('Labeled (Red) vs Unlabeled (Gray)')
    
    # Plot 3: Predicted labels
    plt.subplot(1, n_plots, 3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    plt.title('Predicted Labels')
    plt.colorbar(scatter)
    
    # Plot 4: Prediction confidence
    if probabilities is not None:
        plt.subplot(1, n_plots, 4)
        confidence = np.max(probabilities, axis=1)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=confidence, cmap='plasma', alpha=0.6)
        plt.title('Prediction Confidence')
        plt.colorbar(scatter)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def compare_with_supervised(X: np.ndarray, y: np.ndarray, y_true: np.ndarray, 
                           labeled_mask: np.ndarray, entropy_weight: float = 0.1):
    """Compare semi-supervised method with supervised baseline."""
    
    # Fit semi-supervised model
    mlp = MaximumLikelihoodPredictor(entropy_weight=entropy_weight, random_state=42)
    mlp.fit(X, y, labeled_mask)
    
    # Fit supervised baseline (only labeled data)
    X_labeled = X[labeled_mask]
    y_labeled = y[labeled_mask]
    
    scaler = StandardScaler()
    X_labeled_scaled = scaler.fit_transform(X_labeled)
    X_scaled = scaler.transform(X)
    
    supervised_model = LogisticRegression(random_state=42, max_iter=1000)
    supervised_model.fit(X_labeled_scaled, y_labeled)
    
    # Get predictions
    semi_pred = mlp.predict(X)
    supervised_pred = supervised_model.predict(X_scaled)
    
    # Calculate accuracies
    semi_acc = accuracy_score(y_true, semi_pred)
    supervised_acc = accuracy_score(y_true, supervised_pred)
    
    # Calculate accuracies on unlabeled data only
    semi_unlabeled_acc = accuracy_score(y_true[~labeled_mask], semi_pred[~labeled_mask])
    supervised_unlabeled_acc = accuracy_score(y_true[~labeled_mask], supervised_pred[~labeled_mask])
    
    print(f"\n=== COMPARISON ===")
    print(f"Semi-supervised (λ={entropy_weight}):")
    print(f"  Overall accuracy: {semi_acc:.4f}")
    print(f"  Unlabeled accuracy: {semi_unlabeled_acc:.4f}")
    print(f"\nSupervised baseline:")
    print(f"  Overall accuracy: {supervised_acc:.4f}")
    print(f"  Unlabeled accuracy: {supervised_unlabeled_acc:.4f}")
    print(f"\nImprovement: {semi_acc - supervised_acc:.4f} overall, {semi_unlabeled_acc - supervised_unlabeled_acc:.4f} unlabeled")
    
    return mlp, supervised_model

def main():
    """Main function to run the experiment."""
    print("=== Semi-supervised Learning by Entropy Minimization ===\n")
    
    # Generate dataset
    print("Generating 2D moons dataset...")
    X, y, y_true = generate_moons_dataset(
        n_samples=1000,
        noise=0.1,
        labeled_ratio=0.003,
        random_state=42
    )
    
    labeled_mask = y != -1
    print(f"Dataset: {len(X)} samples, {np.sum(labeled_mask)} labeled, {np.sum(~labeled_mask)} unlabeled")
    
    # Compare different entropy weights
    entropy_weights = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    for ew in entropy_weights:
        print(f"\n--- Testing entropy weight λ={ew} ---")
        mlp, _ = compare_with_supervised(X, y, y_true, labeled_mask, entropy_weight=ew)
        
        # Show entropy statistics
        entropy_stats = mlp.get_entropy()
        print(f"Entropy statistics:")
        for key, value in entropy_stats.items():
            print(f"  {key}: {value:.4f}")
    
    # Final visualization with optimal entropy weight
    print(f"\n--- Final Results with λ=0.1 ---")
    mlp = MaximumLikelihoodPredictor(entropy_weight=0.1, random_state=42)
    mlp.fit(X, y, labeled_mask)
    
    y_pred = mlp.labels_
    probabilities = mlp.probabilities_
    
    plot_results(X, y_true, y_pred, labeled_mask, probabilities, 
                save_path="entropy_minimization_results.png")
    
    print("Experiment completed!")

if __name__ == "__main__":
    main()