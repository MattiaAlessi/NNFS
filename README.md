# Neural Network from Scratch

A neural network framework that has been built entirely from scratch in python (inspired by:[NNFS](https://nnfs.io/) course).
This project also include scripts that downloads a dataset and permits to try it yourself using the **Fashion MNIST** image classification.

## Project Structure:
```text
NeuralNetworkFromScratch
├── Train.py                    # main script
├── EXTRACT_DATASET.py          # download and extract dataset
├── build_exe.bat               # build the .exe (windows only)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
└── .gitignore                  # Ignore List
└── .github
    └── workflows
        └── build-windows.yml   # Create the github release
```

## Clone Repo:

```bash
git clone https://github.com/MattiaAlessi/NNFS.git
```

## Requirements

- No requirement is needed if you're using the .exe in the [release page](https://github.com/MattiaAlessi/NNFS/releases)
- If you wanna try building a neural network with this framework (source code) install the dependencis with:

```bash
pip install -r requirements.txt
```

# How to use:
## Run from source code

```bash
python Train.py --epochs 10 --batch-size 128
```

**The dataset is downloaded automatically the first time you run the program. You can also download it manually by running this file with the command:**

```bash
python EXTRACT_DATASET.py
```

## How to run a trained model?

All models are saved as a pickle file by default as `fashion_mnist.model`
You can import the Model class from `Train.py` to use the NN.

**EXAMPLE:**

```python
from Train import Model
import cv2
import numpy as np

model = Model.load('fashion_mnist.model') #default model

# Load a 28x28 grayscale image and preprocess it exactly like during training
image_data = cv2.imread('my_image.png', cv2.IMREAD_GRAYSCALE)
X = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

predictions = model.predict(X)
print('Predicted class:', predictions.argmax(axis=-1)[0])
```

To evaluate accuracy:

```python
from Train import Model, ensure_dataset, create_data_mnist

model = Model.load('fashion_mnist.model')
dataset_dir = ensure_dataset('.')
_, _, X_test, y_test = create_data_mnist(str(dataset_dir))
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5
model.evaluate(X_test, y_test, batch_size=128)
```

If you wanna build the `.exe` just run:

```bash
.\build_exe.bat
```

Options:

```text
fashion-mnist-trainer.exe --epochs 10 --batch-size 128 --output fashion_mnist.model
```

The training process may take a long time and requires an internet connection on the first run to install from internet the images.


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
