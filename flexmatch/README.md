# FlexMatch Implementation for CIFAR-10

This directory contains a complete implementation of FlexMatch for semi-supervised learning on CIFAR-10, featuring Curriculum Pseudo Labeling (CPL) with adaptive class-specific thresholds.

## Overview

FlexMatch addresses a key limitation of FixMatch: the "constant threshold problem." While FixMatch uses a single confidence threshold (e.g., τ = 0.95) for all classes, FlexMatch introduces adaptive class-specific thresholds that automatically adjust based on how well each class is learning.

### Key Features

- **Curriculum Pseudo Labeling (CPL)**: Adaptive thresholds that adjust per class
- **Warm-up Phase**: Lower thresholds during initial training to accept more samples
- **Weak/Strong Augmentation**: Same consistency regularization as FixMatch
- **Wide ResNet Architecture**: As used in the original FlexMatch paper
- **Comprehensive Evaluation**: Training history, adaptive thresholds, and learning effects

## Files Structure

```
flexmatch/
├── flexmatch.py          # Main FlexMatch implementation
├── models.py             # Wide ResNet model architectures
├── data_utils.py         # CIFAR-10 data loading and splitting
├── train.py              # Full training script
├── example.py            # Simple example for quick testing
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The CIFAR-10 dataset will be automatically downloaded on first run.

## Quick Start

### Simple Example

Run a quick demonstration with a small number of epochs:

```bash
python example.py
```

This will:
- Train FlexMatch for 10 epochs with 1000 labeled samples
- Demonstrate Curriculum Pseudo Labeling
- Show adaptive thresholds and learning effects
- Generate visualization plots

### Full Training

Train FlexMatch with the full configuration:

```bash
python train.py --epochs 1024 --labeled-samples 4000 --plot
```

### Command Line Options

```bash
python train.py [OPTIONS]

Options:
  --model                    Model architecture (wrn_28_2, wrn_28_10, wrn_40_2, wrn_40_10)
  --labeled-samples         Number of labeled samples (default: 4000)
  --val-samples            Number of validation samples (default: 5000)
  --epochs                 Number of training epochs (default: 1024)
  --labeled-batch-size     Batch size for labeled data (default: 64)
  --unlabeled-batch-size   Batch size for unlabeled data (default: 128)
  --base-threshold         Base confidence threshold (default: 0.95)
  --warmup-iterations      Warm-up iterations (default: 40000)
  --lambda-u               Weight for unsupervised loss (default: 1.0)
  --save-dir               Directory to save checkpoints
  --plot                   Plot training results
```

## How FlexMatch Works

### The Problem FixMatch Solves

FixMatch uses a single threshold for all classes:
- If max(p(y|α(u))) > 0.95 → use pseudo-label
- If max(p(y|α(u))) ≤ 0.95 → ignore sample

**Problem**: Different classes learn at different rates!

### FlexMatch's Solution: Curriculum Pseudo Labeling

Instead of one threshold, FlexMatch uses adaptive class-specific thresholds:

```
T(c) = τ × β(c)
```

Where:
- τ = 0.95 (base threshold)
- β(c) = class-specific adjustment factor
- T(c) = final threshold for class c

### The Math Behind β(c)

1. **Compute learning effects** for each class:
   ```
   σ(c) = Σ 1(ûₙ = c)
   ```

2. **Normalize** to get relative learning effect:
   ```
   β(c) = σ(c) / maxⱼ σ(j)
   ```

### Example

After processing 1000 unlabeled images:
- σ(airplane) = 150 → β(airplane) = 150/200 = 0.75
- σ(ship) = 50 → β(ship) = 50/200 = 0.25 (struggling class!)
- σ(cat) = 200 → β(cat) = 200/200 = 1.0 (dominant class)

Final thresholds:
- T(airplane) = 0.95 × 0.75 = 0.71
- T(ship) = 0.95 × 0.25 = 0.24 (much lower!)
- T(cat) = 0.95 × 1.0 = 0.95

### Warm-up Phase

During initial training when most data is unused:
```
β(c) = 1 / √(t/t₀)
```

This lowers all thresholds to accept more samples initially.

## Algorithm Flow

1. **Weak Augmentation**: Apply weak augmentation to unlabeled samples
2. **Adaptive Thresholding**: Use class-specific thresholds T(c)
3. **Pseudo-label Selection**: Select samples that pass threshold
4. **Strong Augmentation**: Apply strong augmentation to selected samples
5. **Consistency Loss**: Compute loss between weak pseudo-labels and strong predictions
6. **Update Learning Effects**: Track how many samples per class are selected
7. **Update Adaptive Factors**: Recompute β(c) based on learning effects

## Expected Performance

Based on the FlexMatch paper:

| Dataset | Labeled Samples | FixMatch | FlexMatch | Improvement |
|---------|----------------|----------|-----------|-------------|
| CIFAR-10 | 4000 | ~94.9% | ~95.7% | +0.8% |
| CIFAR-100 | 4000 | ~71.7% | ~75.2% | +3.5% |
| STL-10 | 4000 | ~86.5% | ~88.6% | +2.1% |

## Key Implementation Details

### Model Architecture
- **Wide ResNet**: Used as in the original paper
- **Variants**: wrn_28_2, wrn_28_10, wrn_40_2, wrn_40_10
- **Default**: wrn_28_2 (good balance of performance and speed)

### Data Augmentation
- **Weak**: Random horizontal flip + random crop
- **Strong**: Weak + RandAugment + CutOut

### Training Configuration
- **Optimizer**: SGD with momentum 0.9
- **Learning Rate**: 0.03 with cosine decay
- **Weight Decay**: 5e-4
- **Batch Sizes**: 64 (labeled), 128 (unlabeled)

### FlexMatch Parameters
- **Base Threshold**: 0.95
- **Warm-up Iterations**: 40000
- **Lambda_u**: 1.0 (unsupervised loss weight)

## Monitoring Training

The implementation provides comprehensive monitoring:

1. **Loss Tracking**: Total, supervised, and unsupervised losses
2. **Mask Ratio**: Percentage of unlabeled samples selected
3. **Adaptive Thresholds**: Current thresholds for each class
4. **Learning Effects**: How many samples per class are being selected
5. **Accuracy**: Validation and test accuracy

## Visualization

The training script generates plots showing:
- Training losses over time
- Accuracy progression
- Mask ratio (pseudo-label selection rate)
- Adaptive thresholds evolution
- Learning effects per class
- Final thresholds vs class names

## Advanced Usage

### Custom Model
```python
from models import create_model
from flexmatch import FlexMatch

# Create custom model
model = create_model(num_classes=10, model_name='wrn_40_10')

# Initialize FlexMatch
flexmatch = FlexMatch(model=model, num_classes=10)
```

### Custom Data Loaders
```python
from data_utils import create_data_loaders

# Create loaders with custom parameters
labeled_loader, unlabeled_loader, val_loader, test_loader = create_data_loaders(
    labeled_samples=2000,
    val_samples=3000,
    labeled_batch_size=32,
    unlabeled_batch_size=64
)
```

### Monitoring Adaptive Thresholds
```python
# Get current adaptive thresholds
thresholds = flexmatch.get_adaptive_thresholds()
print(f"Current thresholds: {thresholds}")

# Get learning effects
learning_effects = flexmatch.get_learning_effects()
print(f"Learning effects: {learning_effects}")
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch sizes or use smaller model
2. **Slow Training**: Use fewer workers or smaller model
3. **Poor Performance**: Check data augmentation and hyperparameters

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## References

- **FlexMatch Paper**: "FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling"
- **FixMatch Paper**: "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"
- **Wide ResNet Paper**: "Wide Residual Networks"

## License

This implementation is provided for educational and research purposes. 