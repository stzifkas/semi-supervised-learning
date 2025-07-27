# Label Propagation on 2D Moons with Weights & Biases

This project implements label propagation algorithm on a 2D moons dataset with comprehensive Weights & Biases (W&B) integration for experiment tracking and hyperparameter optimization.

## Features

- **Label Propagation Algorithm**: Custom implementation of semi-supervised learning using label propagation
- **2D Moons Dataset**: Synthetic dataset generation with configurable noise and labeling ratios
- **Weights & Biases Integration**: Complete experiment tracking with metrics, visualizations, and hyperparameter optimization
- **Hyperparameter Sweeping**: Automated hyperparameter optimization using W&B sweeps
- **Visualization**: Comprehensive plotting of results including true labels, labeled vs unlabeled samples, and predictions
- **Metrics Tracking**: Accuracy metrics for overall, labeled, and unlabeled samples

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd label-propagation-moons
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Weights & Biases:
```bash
wandb login
```

## Usage

### Basic Usage

Run the main experiment:

```bash
python label_propagation_moons.py
```

This will:
- Generate a 2D moons dataset with 1000 samples
- Apply label propagation with default parameters
- Log results to W&B
- Create visualizations
- Display accuracy metrics

### Hyperparameter Optimization

1. Initialize a sweep:
```bash
python sweep_config.py
```

2. Run the sweep agent (in a separate terminal):
```bash
wandb agent <sweep-id>
```

Or run multiple agents for parallel optimization:
```bash
wandb agent <sweep-id> --count 4
```

## Configuration

### Main Experiment Parameters

The main experiment can be configured by modifying the `wandb.init()` call in `label_propagation_moons.py`:

```python
wandb.init(
    project="label-propagation-moons",
    config={
        "n_samples": 1000,        # Total number of samples
        "noise": 0.1,             # Noise level in moons dataset
        "labeled_ratio": 0.1,     # Ratio of labeled samples
        "alpha": 0.99,            # Clamping factor
        "max_iter": 1000,         # Maximum iterations
        "tol": 1e-3,              # Convergence tolerance
        "n_neighbors": 7,         # Number of neighbors for k-NN
        "random_state": 42        # Random seed
    }
)
```

### Sweep Parameters

The sweep configuration in `sweep_config.py` defines the hyperparameter search space:

- `n_samples`: [500, 1000, 2000]
- `noise`: 0.05 to 0.3
- `labeled_ratio`: 0.05 to 0.3
- `alpha`: 0.8 to 0.999
- `max_iter`: [500, 1000, 2000]
- `tol`: [1e-4, 1e-3, 1e-2]
- `n_neighbors`: [5, 7, 10, 15]

## Algorithm Details

### Label Propagation

The label propagation algorithm works as follows:

1. **Affinity Matrix Construction**: Build a similarity matrix using k-nearest neighbors and Gaussian similarity
2. **Label Initialization**: Initialize label matrix with known labels (use -1 for unlabeled)
3. **Iterative Propagation**: Propagate labels through the graph using the formula:
   ```
   Y_new = α * A @ Y_prev + (1 - α) * Y_initial
   ```
   where A is the affinity matrix and α is the clamping factor
4. **Convergence**: Stop when the change in labels is below tolerance

### Dataset Generation

The 2D moons dataset is generated using scikit-learn's `make_moons` function with:
- Configurable noise level
- Partial labeling (only a fraction of samples are labeled)
- Two classes forming crescent shapes

## W&B Integration Features

### Metrics Tracked
- Overall accuracy
- Labeled samples accuracy
- Unlabeled samples accuracy
- Dataset statistics (total, labeled, unlabeled samples)

### Visualizations
- True labels plot
- Labeled vs unlabeled samples plot
- Predicted labels plot
- Confusion matrix
- Probability distributions

### Hyperparameter Optimization
- Random search across parameter space
- Automated experiment tracking
- Best run identification
- Parallel execution support

## Output Files

- `label_propagation_results.png`: Visualization of results
- W&B dashboard: Comprehensive experiment tracking and visualization

## Example Results

Typical results for the default configuration:
- Overall Accuracy: ~0.95-0.98
- Labeled Samples Accuracy: ~1.0 (perfect on training data)
- Unlabeled Samples Accuracy: ~0.94-0.97

## Customization

### Adding New Metrics

To add custom metrics, modify the `wandb.log()` calls in the main script:

```python
wandb.log({
    "custom_metric": your_calculation,
    "another_metric": another_calculation
})
```

### Modifying the Algorithm

The `LabelPropagation` class can be extended with:
- Different affinity matrix construction methods
- Alternative propagation schemes
- Additional regularization terms

### Adding New Datasets

To use different datasets, modify the `generate_moons_dataset` function or create new dataset generators following the same interface.

## Troubleshooting

### Common Issues

1. **W&B Login**: Make sure you're logged in with `wandb login`
2. **Dependencies**: Install all requirements with `pip install -r requirements.txt`
3. **Memory Issues**: Reduce `n_samples` for large datasets
4. **Convergence**: Adjust `max_iter` and `tol` parameters

### Performance Tips

- Use smaller `n_neighbors` for faster computation
- Reduce `max_iter` if convergence is quick
- Use parallel sweep agents for faster hyperparameter optimization

## Contributing

Feel free to contribute by:
- Adding new algorithms
- Improving visualizations
- Adding more datasets
- Enhancing W&B integration

## License

This project is open source and available under the MIT License. 