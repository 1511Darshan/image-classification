# Data Directory

## MNIST Dataset

This project uses the **MNIST (Modified National Institute of Standards and Technology)** dataset for handwritten digit classification.

### Dataset Information

| Property | Value |
|----------|-------|
| **Name** | MNIST Handwritten Digits |
| **Source** | [MNIST Database](http://yann.lecun.com/exdb/mnist/) |
| **License** | Public Domain |
| **Total Samples** | 70,000 (60,000 training + 10,000 testing) |
| **Image Size** | 28×28 pixels (784 features when flattened) |
| **Classes** | 10 (digits 0-9) |
| **Format** | Binary IDX files |

### Dataset Files

The raw MNIST dataset consists of four binary files:

| File | Purpose | Size |
|------|---------|------|
| `train-images-idx3-ubyte` | Training images (60,000) | ~47 MB |
| `train-labels-idx1-ubyte` | Training labels | ~60 KB |
| `t10k-images-idx3-ubyte` | Test images (10,000) | ~7.8 MB |
| `t10k-labels-idx1-ubyte` | Test labels | ~10 KB |

### Downloading the Dataset

#### Option 1: Automatic Download (Recommended)

The dataset will be automatically downloaded when you run `task-1.ipynb`:

```python
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

#### Option 2: Manual Download

Download the binary files from the [MNIST website](http://yann.lecun.com/exdb/mnist/):

```bash
# Linux/macOS
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# Extract
gunzip *.gz
```

#### Option 3: Using Python

```python
import urllib.request
import gzip
import os

base_url = "http://yann.lecun.com/exdb/mnist/"
files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

for file in files:
    url = base_url + file
    print(f"Downloading {file}...")
    urllib.request.urlretrieve(url, file)
    
    # Extract
    with gzip.open(file, 'rb') as f_in:
        with open(file[:-3], 'wb') as f_out:
            f_out.write(f_in.read())
    os.remove(file)
    print(f"✅ {file} downloaded and extracted")
```

### Preprocessed Data

After running `task-1.ipynb`, the following preprocessed numpy arrays are generated:

| File | Shape | Purpose |
|------|-------|---------|
| `X_train_scaled.npy` | (48000, 784) | Training features (scaled) |
| `y_train.npy` | (48000,) | Training labels |
| `X_val.npy` | (12000, 784) | Validation features |
| `y_val.npy` | (12000,) | Validation labels |
| `X_test_scaled.npy` | (10000, 784) | Test features (scaled) |
| `y_test.npy` | (10000,) | Test labels |

#### Data Processing Steps

1. **Normalization:** Pixel values scaled from [0, 255] to [0, 1]
2. **Standardization:** Features standardized to mean ≈ 0, std ≈ 1
3. **Train/Val Split:** Original 60,000 training samples split 80/20 (48,000/12,000)
4. **Format:** Saved as NumPy arrays for efficient loading

### Data Statistics

```
Training Set:
  - Samples: 48,000
  - Features: 784 (28×28 pixels flattened)
  - Classes: 10 (balanced distribution)
  - Feature range: [-2.5, 2.5] (after standardization)
  
Validation Set:
  - Samples: 12,000
  - Features: 784
  - Classes: 10

Test Set:
  - Samples: 10,000
  - Features: 784
  - Classes: 10
```

### Loading Preprocessed Data

```python
import numpy as np
import os

base_path = os.getcwd()

# Load arrays
X_train = np.load(os.path.join(base_path, 'X_train_scaled.npy'))
y_train = np.load(os.path.join(base_path, 'y_train.npy'))
X_val = np.load(os.path.join(base_path, 'X_val.npy'))
y_val = np.load(os.path.join(base_path, 'y_val.npy'))
X_test = np.load(os.path.join(base_path, 'X_test_scaled.npy'))
y_test = np.load(os.path.join(base_path, 'y_test.npy'))

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
```

### License

The MNIST dataset is in the public domain. You are free to use it for any purpose, including commercial applications.

### Citation

If you use MNIST in a publication, please cite:

```bibtex
@article{lecun1998gradient,
  title={Gradient-based learning applied to document recognition},
  author={LeCun, Y. and Bottou, L. and Bengio, Y. and Haffner, P.},
  journal={Proceedings of the IEEE},
  volume={86},
  number={11},
  pages={2278--2324},
  year={1998},
  publisher={IEEE}
}
```

### References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [MNIST on Kaggle](https://www.kaggle.com/datasets/oddratool/mnist-dataset)
- [Wikipedia: MNIST](https://en.wikipedia.org/wiki/MNIST_database)

---

**Note:** If you use a different dataset, please update this file with the appropriate information.
