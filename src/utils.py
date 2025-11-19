"""
Utility functions for model training and evaluation.
"""

import os
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report


def save_results(results, model_name, output_dir="."):
    """
    Save model evaluation results to JSON.
    
    Args:
        results: Dictionary of evaluation metrics
        model_name: Name of the model
        output_dir: Directory to save results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert numpy types to Python types for JSON serialization
    serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, (np.floating, float)):
            serializable[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            serializable[key] = int(value)
        else:
            serializable[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"✅ Results saved to {filepath}")
    return filepath


def get_classification_report(y_true, y_pred):
    """
    Get detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        str: Classification report
    """
    return classification_report(y_true, y_pred)


def print_model_comparison(models_results):
    """
    Print comparison table for multiple models.
    
    Args:
        models_results: Dict mapping model names to their evaluation results
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON - TEST SET PERFORMANCE")
    print("="*70)
    
    # Create comparison table
    print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    
    best_accuracy = 0
    best_model = None
    
    for model_name, results in models_results.items():
        accuracy = results['accuracy']
        precision = results['precision']
        recall = results['recall']
        f1 = results['f1']
        
        print(f"{model_name:<25} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name
    
    print("-" * 70)
    print(f"\n🏆 Best Model: {best_model} ({best_accuracy*100:.2f}% accuracy)")


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


if __name__ == "__main__":
    # Example usage
    set_seed(42)
    print("✅ Random seeds set for reproducibility")
