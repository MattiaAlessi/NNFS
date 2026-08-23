# Neural Network from Scratch

A minimal neural network framework built entirely from scratch in Python using only NumPy, then applied to the **Fashion MNIST** image classification task.

## Project Structure

| File | Description |
|------|-------------|
| `p1.py` – `p7.py` | Progressive exercises covering layers, activations, and loss functions |
| `EXTRACT_DATASET.py` | Downloads and extracts the Fashion MNIST dataset |
| `Train.py` | Full framework (layers, optimizers, losses, accuracy) + training pipeline |

## Features

- **Dense layers** with L1/L2 regularization
- **Activations:** ReLU, Softmax, Sigmoid, Linear
- **Optimizers:** SGD, Adagrad, RMSprop, Adam (with learning rate decay)
- **Losses:** Categorical Cross-Entropy, Binary Cross-Entropy, MSE, MAE
- **Dropout** regularization
- Model **save/load** via pickle

## Usage

1. **Extract the dataset:**
   ```bash
   python EXTRACT_DATASET.py
   ```

2. **Train the model:**
   ```bash
   python Train.py
   ```

## Fashion MNIST Classes

| Label | Class |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Requirements

```
numpy
nnfs
opencv-python
```

## References

Built following the [Neural Networks from Zero to Hero](https://nnfs.io/) course.
