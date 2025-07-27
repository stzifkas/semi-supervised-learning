"""
Weights & Biases Sweep Configuration for Label Propagation Hyperparameter Optimization
"""

import wandb

def get_sweep_config():
    """
    Define sweep configuration for hyperparameter optimization.
    """
    sweep_config = {
        'method': 'random',  # 'grid', 'random', or 'bayes'
        'name': 'label-propagation-optimization',
        'metric': {
            'name': 'overall_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'n_samples': {
                'values': [500, 1000, 2000]
            },
            'noise': {
                'min': 0.05,
                'max': 0.3
            },
            'labeled_ratio': {
                'min': 0.05,
                'max': 0.3
            },
            'alpha': {
                'min': 0.8,
                'max': 0.999
            },
            'max_iter': {
                'values': [500, 1000, 2000]
            },
            'tol': {
                'values': [1e-4, 1e-3, 1e-2]
            },
            'n_neighbors': {
                'values': [5, 7, 10, 15]
            }
        }
    }
    
    return sweep_config

def run_sweep():
    """
    Initialize and run the sweep.
    """
    sweep_config = get_sweep_config()
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="label-propagation-moons")
    
    print(f"Sweep ID: {sweep_id}")
    print("You can now run the sweep agent with:")
    print(f"wandb agent {sweep_id}")
    
    return sweep_id

if __name__ == "__main__":
    run_sweep() 