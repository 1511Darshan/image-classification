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

### Task 1: Data Preprocessing
```bash
jupyter notebook task-1.ipynb
```

### Task 2: Machine Learning Model
```bash
jupyter notebook task-2.ipynb
```

Or simply run the notebooks in VS Code by executing the cells.

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
├── task-1.ipynb                 # Data preprocessing notebook
├── task-2.ipynb                 # Machine learning model notebook
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore rules
├── .gitattributes              # Git large file storage config
├── X_train_scaled.npy          # Preprocessed training features
├── y_train.npy                 # Training labels
├── X_val.npy                   # Validation features
├── y_val.npy                   # Validation labels
├── X_test_scaled.npy           # Test features
├── y_test.npy                  # Test labels
├── confusion_matrices.png       # Model confusion matrices
├── accuracy_comparison.png      # Accuracy comparison chart
└── metrics_comparison.png       # Metrics comparison chart
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

### Task 2: Machine Learning Model ✅
**Decision Tree Classifier Performance:**
- Training Accuracy: High (may indicate overfitting)
- Validation Accuracy: Moderate
- Test Accuracy: ~87-92% (depends on hyperparameters)
- Good interpretability and fast inference

**Logistic Regression Performance:**
- Training Accuracy: Consistent
- Validation Accuracy: Stable
- Test Accuracy: ~92-95%
- Better generalization, more reliable predictions

**Recommendation:**
- Use **Logistic Regression** for production (better accuracy and generalization)
- Decision Tree useful for interpretability and feature importance analysis

---

## Author
Darshan (1511Darshan)

---

## License
MIT License
