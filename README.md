# Neural Network from Scratch


This project is a tiny neural network framework written in pure Python (no black boxes). You can train it to recognise clothes from the Fashion MNIST dataset or with any other dataset!!!

(P.S. this has been inspired by the: [NNFS](https://nnfs.io/) course).   
(P.P.S. there are scripts that downloads automatically the dataset used in the framework)



# How to use

## 1. Clone the Repo:

```bash
git clone https://github.com/MattiaAlessi/NNFS.git
```

## 2. Install requirements

```bash
pip install -r requirements.txt
```


## 3. Run from source code

```bash
python Train.py --epochs 10 --batch-size 128
```

### What Do Those Options Mean?

- --epochs : how many times the network sees the whole dataset. More = smarter, but slower.

- --batch-size : how many images it looks at before updating its “brain”. Bigger values = more stable learning, but need more memory.

IN SHORT:  
`+` epochs = slower but smarter  
`+` batch-size = learning more stable but more memory needed

**IF THE DATASET ISN'T DOWNLOADED USE:**
```bash
python EXTRACT_DATASET.py
```


## How it really works?

1) The program takes each **28×28 pixel image** (grayscaled by the code) and flattens it into a long list of numbers.

2) These numbers pass through several “**layers**” that learn to spot patterns (like edges, shapes, etc.).

3) At the end, the network gives a score for each of the 10 clothing types and picks the highest one.



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

## What Can I Do With the Trained Model?

Just have fun and use your creativity...

- Test it with your own images (take a photo of a piece of clothing, resize it to 28×28, and see if the model guesses right)

- Tweak the code (add an extra layer, change the activation function, and see how the results change)

- Compare (see if normalising the data really helps)

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

This project was built while following the awesome [Neural Networks from Zero to Hero](https://nnfs.io/) course on which I've added my own twiks, if you like drop a star on github, it really means a lot to me.
