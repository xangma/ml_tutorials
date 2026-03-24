# ML Tutorials

Small, notebook-first PyTorch tutorials that explain what the models are doing, not just how to run them.

## Contents

- `iris.ipynb`: a fully connected network for the Iris dataset, with manual walkthroughs of standardization, logits, cross-entropy, backpropagation, and decision boundaries.
- `mnist.ipynb`: a convolutional neural network for MNIST, with shape tracing, filter intuition, gradient inspection, training, and evaluation analysis.

## Quick Start

1. Create and activate a virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Launch Jupyter:

```bash
jupyter lab
```

4. Open either notebook and run the cells from top to bottom.

## Notebook Workflow

For this repo, the public GitHub version is meant to be readable in-browser, so committed notebooks can include curated outputs.

Useful commands:

```bash
make execute-notebooks
make clean-notebooks
```

- `make execute-notebooks` reruns both notebooks in place and saves the outputs.
- `make clean-notebooks` removes outputs if you want a lighter commit or need to reset notebook state.

## Notes

- `iris.ipynb` downloads the Iris CSV directly from the UCI repository at runtime.
- `mnist.ipynb` downloads MNIST into `MNIST_data/` on first run. That directory is intentionally ignored by git.
- Commit executed outputs when they improve the GitHub reading experience, but avoid noisy or accidental output changes.

## Dependencies

The notebooks use:

- PyTorch
- Torchvision
- Matplotlib
- JupyterLab

If your platform needs a custom PyTorch install command, install `torch` and `torchvision` first, then install the remaining packages from `requirements.txt`.
