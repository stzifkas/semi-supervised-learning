# Virtual Adversarial Training (VAT) on CIFAR-10

This project implements Virtual Adversarial Training (VAT) for semi-supervised learning on CIFAR-10 dataset. VAT is a regularization method that makes neural networks robust to small adversarial perturbations, improving generalization especially when labeled data is scarce.

## Features

- **VAT Algorithm**: Complete implementation of Virtual Adversarial Training following Miyato et al.
- **CIFAR-10 Dataset**: Semi-supervised learning on CIFAR-10 with configurable labeled/unlabeled ratios
- **PyTorch Implementation**: Modern PyTorch-based implementation with CNN architecture
- **Weights & Biases Integration**: Complete experiment tracking with metrics and visualizations
- **Quick Testing**: Standalone test script for rapid algorithm validation
- **Training Visualization**: Comprehensive plotting of training curves and results

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd vat_perturbation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Weights & Biases (optional):
```bash
wandb login
```

## Usage

### Full Experiment

Run the complete experiment with W&B tracking:

```bash
python vat_cifar.py
```

This will:
- Load CIFAR-10 subset with configurable labeled ratio
- Train with VAT for 100 epochs
- Log all metrics to W&B
- Save training history plots
- Display final results

## VAT Algorithm

Virtual Adversarial Training works as follows:

### 1. Forward Pass
- Take input images x and get predictions p = softmax(model(x))

### 2. Find Adversarial Direction
- Initialize random perturbation d
- Use power iteration to find direction that maximizes KL divergence:
  - Add small perturbation: x_perturbed = x + ξ * d
  - Get predictions: p_perturbed = softmax(model(x_perturbed))
  - Compute KL divergence: KL(p || p_perturbed)
  - Update d using gradient of KL divergence
  - Normalize d to unit length

### 3. Apply Virtual Adversarial Perturbation
- Scale perturbation: r_vadv = ε * d / ||d||
- Apply to input: x_adversarial = x + r_vadv

### 4. Compute VAT Loss
- Get predictions for adversarial input: p_adversarial = softmax(model(x_adversarial))
- Compute KL divergence: L_vat = KL(p || p_adversarial)

### 5. Total Loss
- Combine supervised and VAT loss: L_total = L_supervised + α * L_vat
- Backpropagate and update weights

## Configuration

### Main Experiment Parameters

The main experiment can be configured by modifying the `wandb.init()` call in `vat_cifar.py`:

```python
wandb.init(
    project="vat-cifar10",
    config={
        "labeled_ratio": 0.1,        # Ratio of labeled samples
        "num_samples": 5000,         # Total number of samples
        "batch_size": 64,            # Batch size
        "num_epochs": 100,           # Number of training epochs
        "learning_rate": 0.001,      # Learning rate
        "weight_decay": 1e-4,        # Weight decay
        "vat_epsilon": 8.0,          # Maximum perturbation magnitude
        "vat_xi": 1e-6,             # Small constant for numerical stability
        "vat_iterations": 1,         # Number of power iterations
        "vat_alpha": 1.0,           # Weight for VAT loss
        "random_state": 42           # Random seed
    }
)
```

### VAT Parameters

- **vat_epsilon**: Maximum magnitude of adversarial perturbation (typically 8.0)
- **vat_xi**: Small constant for numerical stability in power iteration (typically 1e-6)
- **vat_iterations**: Number of power iterations to find adversarial direction (typically 1-2)
- **vat_alpha**: Weight for VAT loss in total loss function (typically 1.0)

## Model Architecture

The implementation uses a simple CNN architecture for CIFAR-10:

```
CIFAR10CNN:
├── Conv2d(3, 32, 3x3, padding=1) + ReLU + MaxPool2d(2x2)
├── Conv2d(32, 64, 3x3, padding=1) + ReLU + MaxPool2d(2x2)
├── Conv2d(64, 128, 3x3, padding=1) + ReLU + MaxPool2d(2x2)
├── Flatten
├── Linear(128*4*4, 512) + ReLU + Dropout(0.25)
└── Linear(512, 10)
```

## Dataset

The implementation uses CIFAR-10 dataset with:

- **Training Set**: 50,000 images (subset used for semi-supervised learning)
- **Test Set**: 10,000 images (used for validation)
- **Classes**: 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Image Size**: 32x32 RGB images
- **Data Augmentation**: Random crop, horizontal flip, normalization

## Semi-Supervised Setup

The dataset is split into labeled and unlabeled samples:

- **Labeled Samples**: Small subset with known labels (e.g., 10% of training data)
- **Unlabeled Samples**: Remaining samples without labels
- **Supervised Loss**: Cross-entropy loss on labeled samples only
- **VAT Loss**: Applied to all samples (both labeled and unlabeled)

## W&B Integration Features

### Metrics Tracked
- Training and validation loss
- Training and validation accuracy
- Learning rate scheduling
- VAT loss components
- Dataset statistics

### Visualizations
- Training curves (loss and accuracy)
- Learning curves in log scale
- Accuracy comparison plots
- Model performance over time

### Hyperparameter Optimization
- Automated experiment tracking
- Best run identification
- Parallel execution support

## Output Files

- `best_vat_model.pth`: Best model weights based on validation accuracy
- `vat_training_history.png`: Training history visualization
- `quick_test_vat_results.png`: Quick test results visualization
- W&B dashboard: Comprehensive experiment tracking

## Example Results

Typical results for CIFAR-10 with 10% labeled data:
- **Supervised Baseline**: ~60-65% accuracy
- **VAT (10% labeled)**: ~70-75% accuracy
- **Improvement**: +5-10% accuracy improvement

## Key Advantages of VAT

1. **Label-Free Regularization**: VAT loss works on unlabeled data
2. **Adversarial Robustness**: Makes model robust to small perturbations
3. **Smoothness**: Encourages smooth predictions in input space
4. **Efficiency**: Only 1-2 extra forward/backward passes per batch
5. **Generality**: Works with any differentiable model

## Customization

### Adding New Models

To use different architectures, modify the `CIFAR10CNN` class or create new model classes:

```python
class YourModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Your architecture here
    
    def forward(self, x):
        # Your forward pass here
        return x
```

### Modifying VAT Parameters

Adjust VAT parameters in the `VATTrainer` initialization:

```python
trainer = VATTrainer(
    model=model,
    vat_epsilon=8.0,      # Increase for stronger regularization
    vat_xi=1e-6,          # Decrease for more precise adversarial direction
    vat_iterations=2,      # Increase for better adversarial direction
    vat_alpha=1.0         # Adjust weight of VAT loss
)
```

### Adding New Datasets

To use different datasets, modify the `load_cifar10_subset` function or create new data loading functions following the same interface.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or number of samples
2. **Slow Training**: Reduce VAT iterations or use smaller epsilon
3. **Poor Convergence**: Adjust learning rate or VAT alpha
4. **W&B Login**: Make sure you're logged in with `wandb login`

### Performance Tips

- Use smaller batch sizes for memory efficiency
- Reduce VAT iterations for faster training
- Use data augmentation for better generalization
- Monitor validation accuracy for early stopping

## Theoretical Background

VAT is based on the principle of local distributional smoothness (LDS). The algorithm:

1. **Finds Adversarial Direction**: Uses power iteration to find the direction that maximizes KL divergence
2. **Applies Virtual Perturbation**: Applies the found perturbation to create adversarial examples
3. **Regularizes Predictions**: Minimizes the difference between original and perturbed predictions

This encourages the model to make smooth predictions in the neighborhood of each training point, improving generalization.

## References

- Miyato, T., Maeda, S., Koyama, M., & Ishii, S. (2018). Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning. IEEE Transactions on Pattern Analysis and Machine Intelligence.
- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and Harnessing Adversarial Examples. arXiv preprint arXiv:1412.6572.

## Contributing

Feel free to contribute by:
- Adding new model architectures
- Improving the VAT algorithm
- Adding more datasets
- Enhancing visualizations
- Optimizing performance

## License

This project is open source and available under the MIT License. 