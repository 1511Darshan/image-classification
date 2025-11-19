"""
Model definitions for MNIST classification.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)


class MNISTClassifier:
    """Base class for MNIST classifiers."""
    
    def __init__(self, model=None):
        """
        Initialize classifier.
        
        Args:
            model: sklearn model instance
        """
        self.model = model
        self.trained = False
    
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
        self.trained = True
        print(f"✅ {self.__class__.__name__} training complete!")
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, dataset_name="Test"):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            dataset_name: Name of the dataset (for printing)
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n{dataset_name} Set Metrics ({self.__class__.__name__}):")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
        }


class DecisionTreeModel(MNISTClassifier):
    """Decision Tree classifier for MNIST."""
    
    def __init__(self, max_depth=30, min_samples_split=10, min_samples_leaf=5, random_state=42):
        """
        Initialize Decision Tree classifier.
        
        Args:
            max_depth: Maximum depth of tree
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf
            random_state: Random state for reproducibility
        """
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        super().__init__(model)


class LogisticRegressionModel(MNISTClassifier):
    """Logistic Regression classifier for MNIST."""
    
    def __init__(self, max_iter=1000, solver='lbfgs', random_state=42):
        """
        Initialize Logistic Regression classifier.
        
        Args:
            max_iter: Maximum iterations
            solver: Optimization algorithm ('lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga')
            random_state: Random state for reproducibility
        """
        model = LogisticRegression(
            max_iter=max_iter,
            multi_class='multinomial',
            random_state=random_state,
            solver=solver
        )
        super().__init__(model)


if __name__ == "__main__":
    # Example usage
    from data import load_mnist_raw
    
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_raw()
    
    print("\nTraining Decision Tree...")
    dt = DecisionTreeModel()
    dt.train(X_train, y_train)
    dt.evaluate(X_test, y_test)
    
    print("\nTraining Logistic Regression...")
    lr = LogisticRegressionModel()
    lr.train(X_train, y_train)
    lr.evaluate(X_test, y_test)
