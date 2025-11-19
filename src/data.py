"""
Data loading and preprocessing utilities for MNIST dataset.
"""

import os
import numpy as np


def load_mnist_raw():
    """
    Load raw MNIST dataset from preprocessed numpy arrays.
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load preprocessed data
    X_train = np.load(os.path.join(base_path, 'X_train_scaled.npy'))
    y_train = np.load(os.path.join(base_path, 'y_train.npy'))
    X_val = np.load(os.path.join(base_path, 'X_val.npy'))
    y_val = np.load(os.path.join(base_path, 'y_val.npy'))
    X_test = np.load(os.path.join(base_path, 'X_test_scaled.npy'))
    y_test = np.load(os.path.join(base_path, 'y_test.npy'))
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_data_shapes():
    """
    Get shapes of all datasets.
    
    Returns:
        dict: Dictionary with shapes of all datasets
    """
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_raw()
    
    return {
        'X_train': X_train.shape,
        'y_train': y_train.shape,
        'X_val': X_val.shape,
        'y_val': y_val.shape,
        'X_test': X_test.shape,
        'y_test': y_test.shape,
    }


def get_data_statistics(X):
    """
    Get statistics about the data.
    
    Args:
        X: Feature matrix
        
    Returns:
        dict: Statistics including mean, std, min, max
    """
    return {
        'mean': float(X.mean()),
        'std': float(X.std()),
        'min': float(X.min()),
        'max': float(X.max()),
        'shape': X.shape,
    }


def get_class_distribution(y):
    """
    Get distribution of classes in labels.
    
    Args:
        y: Label array
        
    Returns:
        dict: Class counts
    """
    unique, counts = np.unique(y, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


if __name__ == "__main__":
    # Example usage
    print("Loading MNIST data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_raw()
    
    print("\nDataset Shapes:")
    for name, shape in get_data_shapes().items():
        print(f"  {name}: {shape}")
    
    print("\nTraining Data Statistics:")
    stats = get_data_statistics(X_train)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nClass Distribution (Training):")
    dist = get_class_distribution(y_train)
    for digit, count in sorted(dist.items()):
        print(f"  Digit {digit}: {count} samples")
