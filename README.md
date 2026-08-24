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

## Publish a GitHub Release automatically

The workflow in `.github/workflows/build-windows.yml` runs when you push a tag beginning with `v`, builds the `.exe` on a Windows runner, and attaches it to a GitHub Release.

```bash
git add .
git commit -m "Package Windows trainer release"
git push origin master
git tag v1.0.0
git push origin v1.0.0
```

After the tag is pushed, open the repository's **Releases** page. GitHub Actions will publish `fashion-mnist-trainer.exe` as a release asset.

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
