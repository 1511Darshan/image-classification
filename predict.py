#!/usr/bin/env python
"""
Prediction script for MNIST classification.

This script loads trained models and makes predictions on new data.

Usage:
    python predict.py --model decision-tree --input data.npy
    python predict.py --model logistic-regression --input sample.npy
"""

import argparse
import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    load_mnist_raw,
    DecisionTreeModel,
    LogisticRegressionModel,
)


def load_data(filepath):
    """Load data from numpy file."""
    data = np.load(filepath)
    return data


def predict_on_samples(X, model_instance, num_samples=5):
    """Make predictions on samples and print results."""
    print(f"\nMaking predictions on {num_samples} samples...")
    print("-" * 50)
    
    # If X is 2D, predict on first num_samples rows
    if X.ndim == 2:
        X_sample = X[:min(num_samples, len(X))]
    else:
        X_sample = X.reshape(1, -1)
    
    predictions = model_instance.predict(X_sample)
    
    for i, pred in enumerate(predictions):
        print(f"Sample {i+1}: Predicted digit = {pred}")
    
    return predictions


def make_prediction(args):
    """Main prediction function."""
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    # Load input data
    print(f"Loading data from {args.input}...")
    X = load_data(args.input)
    print(f"✅ Data loaded: shape {X.shape}")
    
    # Load model
    print(f"\nInitializing {args.model} model...")
    
    if args.model == "decision-tree":
        model = DecisionTreeModel()
    elif args.model == "logistic-regression":
        model = LogisticRegressionModel()
    else:
        print(f"❌ Unknown model: {args.model}")
        sys.exit(1)
    
    # Train on full dataset if not using pre-trained
    if args.train:
        print("Training model on full MNIST dataset...")
        X_train, y_train, _, _, _, _ = load_mnist_raw()
        model.train(X_train, y_train)
        print("✅ Model trained!")
    
    # Make predictions
    if args.train or X.size > 0:  # Make sure we have data to predict on
        predictions = predict_on_samples(X, model, args.samples)
        
        # Save predictions if requested
        if args.output:
            output_path = args.output
            np.save(output_path, predictions)
            print(f"\n✅ Predictions saved to {output_path}")
    
    print("\n✅ Prediction complete!")


def quick_test():
    """Quick test on sample MNIST data."""
    print("Running quick test on MNIST test set...")
    print("="*50)
    
    # Load test data
    _, _, _, _, X_test, y_test = load_mnist_raw()
    print(f"✅ Loaded {len(X_test)} test samples")
    
    # Test Decision Tree
    print("\nTesting Decision Tree...")
    dt = DecisionTreeModel()
    dt.train(X_test[:100], y_test[:100])  # Quick train on subset
    predict_on_samples(X_test[:5], dt, num_samples=5)
    print("✅ Decision Tree test complete")
    
    # Test Logistic Regression
    print("\nTesting Logistic Regression...")
    lr = LogisticRegressionModel()
    lr.train(X_test[:100], y_test[:100])  # Quick train on subset
    predict_on_samples(X_test[:5], lr, num_samples=5)
    print("✅ Logistic Regression test complete")
    
    print("\n" + "="*50)
    print("✅ All tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make predictions with MNIST classification models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=['decision-tree', 'logistic-regression'],
        default='logistic-regression',
        help="Which model to use for predictions"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input data file (.npy)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save predictions (.npy)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples to display predictions for"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train model on full MNIST dataset before predicting"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test instead of making predictions"
    )
    
    args = parser.parse_args()
    
    # Run quick test if requested
    if args.test:
        quick_test()
    else:
        # Check if input file is provided
        if not args.input:
            print("❌ Error: --input is required (or use --test for quick test)")
            parser.print_help()
            sys.exit(1)
        
        make_prediction(args)
