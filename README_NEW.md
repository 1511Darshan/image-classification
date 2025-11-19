# MNIST Image Classification Project

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive machine learning and deep learning project for handwritten digit classification using the MNIST dataset. This repository implements end-to-end pipelines including data preprocessing, traditional ML models, NLP analysis, and CNN-based deep learning approaches.

## 📊 Project Overview

This project demonstrates a complete ML/DL workflow:

1. **Data Preprocessing** - MNIST dataset loading, normalization, and splitting
2. **Traditional ML Models** - Decision Tree and Logistic Regression classifiers
3. **NLP Analysis** - Sentiment analysis using VADER and TextBlob (Task 3)
4. **Deep Learning** - Convolutional Neural Network for high-accuracy classification

### 🎯 Key Results

| Model | Accuracy | Framework |
|-------|----------|-----------|
| Decision Tree | 86.90% | Scikit-learn |
| Logistic Regression | 92.01% | Scikit-learn |
| CNN | >95.00% | TensorFlow/Keras |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/1511Darshan/image-classification.git
   cd image-classification
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # OR using conda
   conda create -n mnist python=3.10
   conda activate mnist
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage Guide

### Training Models

#### Train all models:
```bash
python train.py
```

#### Train specific model:
```bash
python train.py --model decision-tree
python train.py --model logistic-regression
```

#### Train with custom output directory:
```bash
python train.py --output-dir ./results
```

### Making Predictions

#### Quick test (uses MNIST test set):
```bash
python predict.py --test
```

#### Predict on custom data:
```bash
python predict.py --model logistic-regression --input data.npy --output predictions.npy
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### Using in Jupyter Notebooks

See the `notebooks/` directory for interactive exploration:

```bash
jupyter notebook notebooks/
```

## 📁 Project Structure

```
image-classification/
├── README.md                      # This file
├── LICENSE                        # MIT License
├── CONTRIBUTING.md               # Contribution guidelines
├── requirements.txt              # Python dependencies
├── pytest.ini                    # Pytest configuration
│
├── src/                          # Reusable Python modules
│   ├── __init__.py
│   ├── data.py                  # Data loading & preprocessing
│   ├── model.py                 # Model definitions & training
│   └── utils.py                 # Utility functions
│
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── task-1.ipynb             # Data preprocessing
│   ├── task-2.ipynb             # Traditional ML models
│   ├── task-3.ipynb             # NLP sentiment analysis
│   └── task-4.ipynb             # Deep learning CNN
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   └── test_models.py           # Model tests
│
├── data/                         # Data directory
│   └── README.md                # Dataset documentation
│
├── models/                       # Saved model weights
│   └── .gitkeep
│
├── train.py                      # Training script
├── predict.py                    # Prediction script
│
├── X_train_scaled.npy           # Preprocessed training features
├── y_train.npy                  # Training labels
├── X_val.npy                    # Validation features
├── y_val.npy                    # Validation labels
├── X_test_scaled.npy            # Test features
└── y_test.npy                   # Test labels
```

## 📚 Detailed Task Documentation

### Task 1: Data Preprocessing ✅

**File:** `notebooks/task-1.ipynb`

- Load MNIST dataset from binary IDX files
- Normalize pixel values to [0, 1]
- Standardize features (mean=0, std=1)
- Check for data quality and missing values
- Split into train/validation/test (80/20)
- Save as NumPy arrays

**Output:** `X_train_scaled.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test_scaled.npy`, `y_test.npy`

### Task 2: Machine Learning Models ✅

**File:** `notebooks/task-2.ipynb` | **Reusable code:** `src/`

**Models Implemented:**

1. **Decision Tree Classifier**
   - Parameters: max_depth=30, min_samples_split=10, min_samples_leaf=5
   - Test Accuracy: 86.90%
   - Good for interpretability

2. **Logistic Regression**
   - Solver: lbfgs, Max iterations: 1000
   - Test Accuracy: 92.01%
   - Best traditional ML performance

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- Per-class performance analysis

### Task 3: Natural Language Processing ✅

**File:** `notebooks/task-3.ipynb`

- Sentiment analysis using VADER (Rule-based, optimized for social media)
- Sentiment analysis using TextBlob (Lexicon-based)
- Comparison of methods and subjectivity analysis
- Analysis of 20 sample texts

### Task 4: Deep Learning CNN ✅

**File:** `notebooks/task-4.ipynb`

**CNN Architecture:**
```
Input (28×28×1)
↓
Conv2D(32) → Conv2D(64) → MaxPool → Dropout(0.25)
↓
Conv2D(128) → MaxPool → Dropout(0.25)
↓
Dense(256) + BatchNorm + Dropout(0.5)
↓
Dense(128) + BatchNorm + Dropout(0.3)
↓
Dense(10) + Softmax
```

- **Optimizer:** Adam (learning_rate=0.001)
- **Loss:** Categorical Crossentropy
- **Training:** 30 epochs with Early Stopping
- **Test Accuracy:** >95.00%
- **Improvement over traditional ML:** +3-8%

## 🔬 Dataset Information

### MNIST Overview
- **Source:** [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- **Samples:** 70,000 (60,000 training + 10,000 test)
- **Image Size:** 28×28 pixels (784 features)
- **Classes:** 10 (digits 0-9)
- **License:** Public Domain

See `data/README.md` for detailed dataset documentation and download instructions.

## 💻 Development

### Code Style

This project follows Python best practices:

```bash
# Format code with black
black .

# Check with flake8
flake8 src/ tests/

# Run tests
pytest

# Check imports with isort
isort .
```

### Pre-commit Hooks

Set up pre-commit to automatically format/lint code:

```bash
pre-commit install
```

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## 📊 Model Performance Details

### Confusion Matrix Analysis

- **Decision Tree:** Performs well on digits 0, 1, 6, 8 but struggles with 4, 9
- **Logistic Regression:** More balanced performance across all digits
- **CNN:** Consistent >95% accuracy on all digit classes

### Learning Curves

Training history visualization shows:
- CNN: Smooth convergence, no overfitting
- Traditional ML: Immediate convergence

## 🔧 Configuration

### Python Dependencies

Core packages:
- `numpy>=1.24` - Numerical computing
- `pandas>=2.0` - Data manipulation
- `scikit-learn>=1.3` - Traditional ML
- `torch>=2.0` + `torchvision>=0.15` - Deep learning (PyTorch)
- `tensorflow>=2.14` - Alternative (TensorFlow/Keras)
- `matplotlib>=3.7`, `seaborn>=0.13` - Visualization

Development:
- `pytest>=7.4` - Testing
- `black>=23.7`, `flake8>=6.1` - Code quality
- `jupyter>=1.0` - Notebooks

See `requirements.txt` for complete list.

## 📈 Results Summary

### Performance Metrics (Test Set)

| Metric | Decision Tree | Logistic Regression | CNN |
|--------|---------------|-------------------|-----|
| Accuracy | 86.90% | 92.01% | >95.00% |
| Precision | 0.8703 | 0.9205 | >0.95 |
| Recall | 0.8690 | 0.9201 | >0.95 |
| F1-Score | 0.8689 | 0.9200 | >0.95 |

### Key Insights

1. **Deep learning outperforms traditional ML** by 3-8% on image classification
2. **Logistic Regression** provides best speed/accuracy trade-off for traditional methods
3. **CNN architecture** with dropout and batch norm effectively prevents overfitting
4. **Data preprocessing** (standardization) crucial for all models

## 🚨 Troubleshooting

### Common Issues

**Issue:** Module not found error when running train.py
```bash
# Solution: Ensure you're in the project root directory
cd image-classification
python train.py
```

**Issue:** Data files not found
```bash
# Solution: Run task-1.ipynb first to generate preprocessed arrays
jupyter notebook notebooks/task-1.ipynb
```

**Issue:** CUDA/GPU errors with deep learning
```bash
# Solution: CPU-only mode is default. For GPU, reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Dataset License

The MNIST dataset is in the public domain. See [data/README.md](data/README.md) for citation details.

## 👥 Author & Contact

**Darshan** ([@1511Darshan](https://github.com/1511Darshan))

For questions, issues, or suggestions:
- Open an [Issue](https://github.com/1511Darshan/image-classification/issues)
- Submit a [Pull Request](https://github.com/1511Darshan/image-classification/pulls)
- See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines

## 🔗 References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [PyTorch Documentation](https://pytorch.org/docs)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/guide)
- [Deep Learning for Computer Vision](https://cs231n.stanford.edu/)

## 🎓 Learning Resources

This project is suitable for:
- Learning ML fundamentals with real data
- Understanding data preprocessing pipelines
- Comparing traditional ML vs deep learning
- Best practices for code organization
- Testing and CI/CD in ML projects

---

## 📌 Project Status

- ✅ Task 1 (Data Preprocessing): Complete
- ✅ Task 2 (Traditional ML Models): Complete  
- ✅ Task 3 (NLP Analysis): Complete
- ✅ Task 4 (Deep Learning CNN): Complete
- ✅ Code Refactoring & Modularization: Complete
- ✅ Testing & CI/CD: Implemented
- ✅ Documentation: Complete

**Last Updated:** November 2025

---

<p align="center">
Made with ❤️ for the ML community
</p>
