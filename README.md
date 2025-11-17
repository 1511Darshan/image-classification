# MNIST Image Classification Project

## Overview
This project implements an end-to-end machine learning pipeline for handwritten digit classification using the MNIST dataset.

---

## Tasks

### Task 1: Data Preprocessing ✅ COMPLETED
**File:** `task-1.ipynb`

**Objectives:**
- Load and explore the MNIST dataset
- Clean and normalize pixel values (0-1 range)
- Check for missing values and data quality
- Standardize features (mean=0, std=1)
- Split data into train/validation/test sets (80/20 split)
- Save preprocessed data as numpy arrays

**Dataset Summary:**
- **Training set:** 48,000 samples
- **Validation set:** 12,000 samples
- **Test set:** 10,000 samples
- **Features:** 784 (28×28 pixel images)
- **Classes:** 10 (digits 0-9)

**Output Files:**
- `X_train_scaled.npy` - Training features (48000, 784)
- `y_train.npy` - Training labels (48000,)
- `X_val.npy` - Validation features (12000, 784)
- `y_val.npy` - Validation labels (12000,)
- `X_test_scaled.npy` - Test features (10000, 784)
- `y_test.npy` - Test labels (10000,)

---

### Task 2: Machine Learning Model ✅ COMPLETED
**File:** `task-2.ipynb`

**Objectives:**
- Implement Decision Tree Classifier
- Implement Logistic Regression model
- Evaluate both models using multiple metrics
- Compare model performance
- Create visualizations for model evaluation

**Models Implemented:**
1. **Decision Tree Classifier**
   - Max depth: 30, Min samples split: 10, Min samples leaf: 5
   - Test Accuracy: ~86.90%
   - Good for interpretability

2. **Logistic Regression**
   - Solver: lbfgs, Max iterations: 1000
   - Test Accuracy: ~92.01%
   - Better generalization

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices for each model
- Per-class performance analysis
- Detailed classification reports

**Output Visualizations:**
- `confusion_matrices.png` - Confusion matrices for both models
- `accuracy_comparison.png` - Model accuracy comparison
- `metrics_comparison.png` - All metrics across models

---

### Task 3: Natural Language Processing ✅ COMPLETED
**File:** `task-3.ipynb`

**Objectives:**
- Perform sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Perform sentiment analysis using TextBlob
- Compare sentiment analysis methods
- Analyze subjectivity vs objectivity
- Create visualizations for sentiment distribution

**Methods Implemented:**
1. **VADER Sentiment Analysis**
   - Rule-based approach optimized for social media
   - Provides: positive, negative, neutral, compound scores
   - Compound score: -1 (most negative) to +1 (most positive)

2. **TextBlob Sentiment Analysis**
   - Lexicon-based approach for general text
   - Provides: polarity and subjectivity scores
   - Polarity: -1 (negative) to +1 (positive)
   - Subjectivity: 0 (objective) to 1 (subjective)

**Analysis Results:**
- 20 sample texts analyzed
- Positive, Negative, and Neutral sentiment classification
- Agreement rate between methods
- Correlation analysis of scores
- Subjectivity analysis

**Output Visualizations:**
- `sentiment_distribution.png` - Sentiment count and score histograms
- `sentiment_comparison.png` - Scatter plots of method comparison
- Detailed analysis of 5 representative texts

---

### Task 4: Image Classification with CNN ✅ COMPLETED
**File:** `task-4.ipynb`

**Objectives:**
- Build and train a Convolutional Neural Network (CNN)
- Classify handwritten digits with high accuracy
- Evaluate model performance with comprehensive metrics
- Compare deep learning vs traditional machine learning
- Visualize training progress and predictions

**CNN Architecture:**
- **Input Layer:** 28×28×1 grayscale images
- **Block 1:** Conv2D(32) → Conv2D(64) → MaxPool(2×2) → Dropout(0.25)
- **Block 2:** Conv2D(128) → MaxPool(2×2) → Dropout(0.25)
- **Dense Block:** Dense(256) + BatchNorm + Dropout(0.5) → Dense(128) + BatchNorm + Dropout(0.3)
- **Output Layer:** Dense(10) + Softmax (10-class classification)
- **Optimizer:** Adam (learning rate: 0.001)
- **Loss Function:** Categorical Crossentropy
- **Training:** 30 epochs with Early Stopping

**Performance Metrics:**
- Test Accuracy: >95%
- Precision, Recall, F1-Score for all classes
- Confusion matrix showing per-class accuracy
- Significant improvement over traditional ML models

**Performance Comparison:**
- Decision Tree: 86.90%
- Logistic Regression: 92.01%
- CNN: >95.00% ✨
- **Deep Learning advantage: +3-8% improvement**

**Output Visualizations:**
- `training_history.png` - Loss and accuracy over epochs
- `cnn_confusion_matrix.png` - Per-class accuracy breakdown
- `cnn_sample_predictions.png` - Random prediction examples
- `cnn_misclassified.png` - Failed classifications analysis
- `model_comparison.png` - All models side-by-side comparison

---



## Technologies Used
- Python 3.13
- NumPy - Numerical computing
- Pandas - Data manipulation
- Scikit-learn - Machine learning utilities
- NLTK - Natural Language Toolkit
- TextBlob - Simplified NLP
- Matplotlib & Seaborn - Data visualization
- TensorFlow/Keras - Deep learning (for future tasks)
- Jupyter Notebook - Interactive development

---

## How to Run

Run any task notebook in Jupyter or VS Code:

```bash
jupyter notebook task-1.ipynb  # Data Preprocessing
jupyter notebook task-2.ipynb  # Traditional ML Models
jupyter notebook task-3.ipynb  # NLP Sentiment Analysis
jupyter notebook task-4.ipynb  # Deep Learning CNN
```

Or simply open the notebooks in VS Code and execute the cells.

---

## Dataset Information
- **Source:** MNIST (Modified National Institute of Standards and Technology)
- **Format:** Binary idx files (train-images.idx3-ubyte, train-labels.idx1-ubyte, etc.)
- **Total Samples:** 70,000 (60,000 training + 10,000 test)
- **Image Size:** 28×28 pixels (784 features when flattened)
- **Classes:** 10 (digits 0-9)

---

## Project Structure
```
image-classification/
├── task-1.ipynb                 # Data preprocessing
├── task-2.ipynb                 # Machine learning (Decision Tree, Logistic Regression)
├── task-3.ipynb                 # NLP sentiment analysis (VADER, TextBlob)
├── task-4.ipynb                 # Deep learning CNN
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore rules
├── .gitattributes              # Git large file storage config
├── X_train_scaled.npy          # Preprocessed training features
├── y_train.npy                 # Training labels
├── X_val.npy                   # Validation features
├── y_val.npy                   # Validation labels
├── X_test_scaled.npy           # Test features
├── y_test.npy                  # Test labels
├── confusion_matrices.png       # Task 2: Model confusion matrices
├── accuracy_comparison.png      # Task 2: Accuracy comparison
├── metrics_comparison.png       # Task 2: Metrics comparison
├── sentiment_distribution.png   # Task 3: Sentiment analysis distribution
├── sentiment_comparison.png     # Task 3: VADER vs TextBlob comparison
├── training_history.png         # Task 4: CNN training history
├── cnn_confusion_matrix.png     # Task 4: CNN confusion matrix
├── cnn_sample_predictions.png   # Task 4: Sample predictions
├── cnn_misclassified.png        # Task 4: Misclassified examples
└── model_comparison.png         # Task 4: All models comparison
```

---

## Results Summary

### Task 1: Data Preprocessing ✅
- Successfully loaded 60,000 training images and 10,000 test images
- Normalized pixel values from [0, 255] to [0, 1]
- Standardized features to mean ≈ 0 and std ≈ 1
- Split training data into 80% train and 20% validation
- All preprocessed data saved as numpy arrays
- Balanced class distribution across all 10 digits

### Task 2: Machine Learning Models ✅
**Decision Tree Classifier:**
- Test Accuracy: ~86.90%
- Good interpretability but prone to overfitting

**Logistic Regression:**
- Test Accuracy: ~92.01%
- Better generalization and more reliable predictions

### Task 3: Sentiment Analysis ✅
- 20 sample texts analyzed with VADER and TextBlob
- VADER: Rule-based, optimized for social media (compound score)
- TextBlob: Lexicon-based, includes subjectivity analysis
- Moderate agreement rate between methods
- Both effective for sentiment classification

### Task 4: CNN Image Classification ✅
- **Test Accuracy: >95%** 🎯
- Outperforms traditional ML models by 3-8%
- Well-balanced performance across all digit classes
- Comprehensive visualizations of training and predictions
- Demonstrates deep learning advantage for image tasks

---

## Performance Comparison

| Model | Accuracy |
|-------|----------|
| Decision Tree | 86.90% |
| Logistic Regression | 92.01% |
| CNN | >95.00% |

**Key Insight:** Deep learning (CNN) significantly outperforms traditional machine learning models on image classification tasks. The convolutional architecture is specifically designed to extract spatial features from images, resulting in superior performance.

---

## Author
Darshan (1511Darshan)

---

## License
MIT License
