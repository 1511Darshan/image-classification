"""
Unit tests for MNIST classification models.
"""

import pytest
import numpy as np
from src import (
    load_mnist_raw,
    DecisionTreeModel,
    LogisticRegressionModel,
    get_data_shapes,
    get_data_statistics,
    get_class_distribution,
)


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_mnist_raw(self):
        """Test loading MNIST data."""
        X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_raw()
        
        # Check shapes
        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        
        # Check feature dimensions
        assert X_train.shape[1] == 784  # 28*28
        
        # Check data types
        assert X_train.dtype in [np.float32, np.float64]
        assert y_train.dtype in [np.int32, np.int64]
    
    def test_get_data_shapes(self):
        """Test getting data shapes."""
        shapes = get_data_shapes()
        
        assert 'X_train' in shapes
        assert 'y_train' in shapes
        assert shapes['X_train'][1] == 784
    
    def test_get_data_statistics(self):
        """Test data statistics calculation."""
        X_train, _, _, _, _, _ = load_mnist_raw()
        stats = get_data_statistics(X_train)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert isinstance(stats['mean'], (int, float))
    
    def test_get_class_distribution(self):
        """Test class distribution calculation."""
        _, y_train, _, _, _, _ = load_mnist_raw()
        dist = get_class_distribution(y_train)
        
        # Check all 10 digits are present
        assert len(dist) == 10
        assert all(isinstance(count, int) for count in dist.values())


class TestDecisionTreeModel:
    """Test Decision Tree model."""
    
    @pytest.fixture
    def small_dataset(self):
        """Create small dataset for testing."""
        X = np.random.randn(100, 784)
        y = np.random.randint(0, 10, 100)
        return X, y
    
    def test_initialization(self):
        """Test model initialization."""
        model = DecisionTreeModel()
        assert model.model is not None
        assert not model.trained
    
    def test_training(self, small_dataset):
        """Test model training."""
        X, y = small_dataset
        model = DecisionTreeModel()
        model.train(X, y)
        
        assert model.trained
    
    def test_prediction(self, small_dataset):
        """Test making predictions."""
        X_train, y_train = small_dataset
        X_test = X_train[:10]
        
        model = DecisionTreeModel()
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (10,)
        assert all(0 <= pred <= 9 for pred in predictions)
    
    def test_evaluation(self, small_dataset):
        """Test model evaluation."""
        X, y = small_dataset
        model = DecisionTreeModel()
        model.train(X, y)
        
        results = model.evaluate(X, y)
        
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert 'confusion_matrix' in results
        assert 0 <= results['accuracy'] <= 1


class TestLogisticRegressionModel:
    """Test Logistic Regression model."""
    
    @pytest.fixture
    def small_dataset(self):
        """Create small dataset for testing."""
        X = np.random.randn(100, 784)
        y = np.random.randint(0, 10, 100)
        return X, y
    
    def test_initialization(self):
        """Test model initialization."""
        model = LogisticRegressionModel()
        assert model.model is not None
        assert not model.trained
    
    def test_training(self, small_dataset):
        """Test model training."""
        X, y = small_dataset
        model = LogisticRegressionModel()
        model.train(X, y)
        
        assert model.trained
    
    def test_prediction(self, small_dataset):
        """Test making predictions."""
        X_train, y_train = small_dataset
        X_test = X_train[:10]
        
        model = LogisticRegressionModel()
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (10,)
        assert all(0 <= pred <= 9 for pred in predictions)
    
    def test_evaluation(self, small_dataset):
        """Test model evaluation."""
        X, y = small_dataset
        model = LogisticRegressionModel()
        model.train(X, y)
        
        results = model.evaluate(X, y)
        
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert 'confusion_matrix' in results
        assert 0 <= results['accuracy'] <= 1


class TestModelComparison:
    """Test comparing models."""
    
    def test_models_have_similar_interface(self):
        """Test that models implement the same interface."""
        models = [
            DecisionTreeModel(),
            LogisticRegressionModel(),
        ]
        
        for model in models:
            assert hasattr(model, 'train')
            assert hasattr(model, 'predict')
            assert hasattr(model, 'evaluate')
            assert hasattr(model, 'trained')


class TestSmokeTests:
    """Smoke tests for basic functionality."""
    
    def test_import_all_modules(self):
        """Test that all modules can be imported."""
        from src import data, model, utils
        assert data is not None
        assert model is not None
        assert utils is not None
    
    def test_train_simple_model(self):
        """Test training a simple model with small data."""
        X = np.random.randn(50, 784).astype(np.float32)
        y = np.random.randint(0, 10, 50)
        
        model = DecisionTreeModel()
        model.train(X, y)
        
        predictions = model.predict(X[:5])
        assert len(predictions) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
