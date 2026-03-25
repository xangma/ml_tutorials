# ML Tutorials

Small, notebook-first PyTorch tutorials that focus on what the models are computing, not just how to run them.

## Notebooks

- `iris.ipynb`: a fully connected network for the Iris dataset, with walkthroughs of standardization, logits, cross-entropy, backpropagation, and decision boundaries.
- `mnist.ipynb`: a convolutional neural network for MNIST, with shape tracing, filter intuition, gradient inspection, training, and evaluation analysis.

## Quick Start

From the repository root:

```bash
conda create -n ml_tutorials python jupyter ipykernel
conda activate ml_tutorials
pip install -r requirements.txt
```

If your platform needs a custom PyTorch install command, install `torch` and `torchvision` first, then install the remaining packages from `requirements.txt`.

## Notebook Workflow

Committed notebooks can include curated outputs so the GitHub view stays readable in-browser.

```bash
make execute-notebooks
make clean-notebooks
```

- `make execute-notebooks` reruns both notebooks in place and saves their outputs.
- `make clean-notebooks` removes outputs and execution counts if you want a lighter diff or need to reset notebook state.

## Notes

- `iris.ipynb` downloads the Iris CSV directly from the UCI repository at runtime.
- `mnist.ipynb` downloads MNIST into `MNIST_data/` on first run. That directory is intentionally ignored by git.
- Commit executed outputs when they improve the reading experience, but avoid noisy or accidental notebook changes.

## Dependencies

The notebooks use `torch`, `torchvision`, `matplotlib`, `pandas`, `jupyter`, `ipykernel`, and `jupyterlab`.
