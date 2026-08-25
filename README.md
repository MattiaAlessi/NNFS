# Neural Network from Scratch

A minimal neural network framework built entirely from scratch in Python using only NumPy, then applied to the **Fashion MNIST** image classification task.

## Project Structure

| File | Description |
|------|-------------|
| `EXTRACT_DATASET.py` | Downloads and extracts the Fashion MNIST dataset |
| `Train.py` | Full framework and training pipeline |
| `build_exe.bat` | Builds the Windows executable locally |
| `.github/workflows/build-windows.yml` | Builds and publishes tagged Windows releases |

## Requirements

For Python development:

```text
numpy
nnfs
opencv-python
pyinstaller
```

Install them with:

```bash
python -m pip install -r requirements.txt
```

## Run from source

```bash
python Train.py --epochs 10 --batch-size 128
```

The dataset is downloaded automatically the first time. You can also download it explicitly:

```bash
python EXTRACT_DATASET.py
```

## Run inference with a trained model

After training, the model is saved as a pickle file (default: `fashion_mnist.model`). Because the pickle stores the full model object, you must import the framework from `Train.py` before loading it:

```python
from Train import Model
import cv2
import numpy as np

model = Model.load('fashion_mnist.model')

# Load a 28x28 grayscale image and preprocess it exactly like during training
image_data = cv2.imread('my_image.png', cv2.IMREAD_GRAYSCALE)
X = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

predictions = model.predict(X)
print('Predicted class:', predictions.argmax(axis=-1)[0])
```

The predicted integer maps to a class in the table below.

To evaluate accuracy on the test set instead:

```python
from Train import Model, ensure_dataset, create_data_mnist

model = Model.load('fashion_mnist.model')
dataset_dir = ensure_dataset('.')
_, _, X_test, y_test = create_data_mnist(str(dataset_dir))
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5
model.evaluate(X_test, y_test, batch_size=128)
```

## Build the Windows `.exe`

On Windows, after installing the requirements:

```bat
build_exe.bat
```

The result is `dist\fashion-mnist-trainer.exe`. The executable downloads Fashion-MNIST automatically when started, so the dataset does not need to be bundled into the binary.

Options:

```text
fashion-mnist-trainer.exe --epochs 10 --batch-size 128 --output fashion_mnist.model
```

The training process may take a long time and requires an internet connection on the first run.


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

## References

Built following the [Neural Networks from Zero to Hero](https://nnfs.io/) course.
