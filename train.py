#!/usr/bin/env python
"""
Training script for MNIST classification models.

This script trains and evaluates machine learning models on the MNIST dataset.

Usage:
    python train.py                     # Train all models
    python train.py --model decision-tree  # Train only decision tree
    python train.py --model logistic-regression  # Train only logistic regression
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    load_mnist_raw,
    DecisionTreeModel,
    LogisticRegressionModel,
    set_seed,
    print_model_comparison,
    save_results,
)


def train_decision_tree(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Decision Tree model."""
    print("\n" + "="*70)
    print("TRAINING DECISION TREE CLASSIFIER")
    print("="*70)
    
    dt = DecisionTreeModel(max_depth=30, min_samples_split=10, min_samples_leaf=5)
    dt.train(X_train, y_train)
    
    # Evaluate on all sets
    train_results = dt.evaluate(X_train, y_train, "Training")
    val_results = dt.evaluate(X_val, y_val, "Validation")
    test_results = dt.evaluate(X_test, y_test, "Test")
    
    return dt, {
        'train': train_results,
        'val': val_results,
        'test': test_results,
    }


def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Logistic Regression model."""
    print("\n" + "="*70)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*70)
    
    lr = LogisticRegressionModel(max_iter=1000)
    lr.train(X_train, y_train)
    
    # Evaluate on all sets
    train_results = lr.evaluate(X_train, y_train, "Training")
    val_results = lr.evaluate(X_val, y_val, "Validation")
    test_results = lr.evaluate(X_test, y_test, "Test")
    
    return lr, {
        'train': train_results,
        'val': val_results,
        'test': test_results,
    }


def plot_comparison(models_results, output_dir="."):
    """Create comparison visualizations."""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Accuracy comparison across datasets
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = ['train', 'val', 'test']
    x = np.arange(len(datasets))
    width = 0.35
    
    for i, (model_name, results) in enumerate(models_results.items()):
        accuracies = [results[ds]['accuracy'] for ds in datasets]
        ax.bar(x + i*width, accuracies, width, label=model_name, alpha=0.8)
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(['Training', 'Validation', 'Test'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    filepath = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filepath}")
    plt.close()


def main(args):
    """Main training function."""
    # Set random seeds
    set_seed(42)
    
    # Load data
    print("\n" + "="*70)
    print("LOADING MNIST DATASET")
    print("="*70)
    print("\nLoading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_raw()
    
    print(f"\n✅ Data loaded successfully!")
    print(f"  Training set:   {X_train.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set:       {X_test.shape}")
    
    models_results = {}
    
    # Train models based on arguments
    if args.model in ['all', 'decision-tree']:
        print("\n" + "-"*70)
        dt, dt_results = train_decision_tree(X_train, y_train, X_val, y_val, X_test, y_test)
        models_results['Decision Tree'] = dt_results['test']
        
        # Save results
        save_results(dt_results['test'], 'decision_tree', args.output_dir)
    
    if args.model in ['all', 'logistic-regression']:
        print("\n" + "-"*70)
        lr, lr_results = train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test)
        models_results['Logistic Regression'] = lr_results['test']
        
        # Save results
        save_results(lr_results['test'], 'logistic_regression', args.output_dir)
    
    # Print comparison
    if models_results:
        print_model_comparison(models_results)
        
        # Generate visualizations
        if args.plot:
            plot_comparison(models_results, args.output_dir)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\n✅ Training finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train machine learning models for MNIST classification"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=['all', 'decision-tree', 'logistic-regression'],
        default='all',
        help="Which model(s) to train"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save results and visualizations"
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        default=True,
        help="Skip generating comparison plots"
    )
    
    args = parser.parse_args()
    
    main(args)
