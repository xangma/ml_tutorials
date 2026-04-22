# Iris Notebook Training Explainability Design

## Context

The repository contains tutorial notebooks for small machine learning examples, including `iris.ipynb`. The current Iris notebook already explains dataset preparation, tensor shapes, one forward pass, one backward pass, training, and evaluation. The next revision should preserve that beginner-friendly path while making training itself much more observable.

The revised notebook should answer questions such as:

- What changes inside the network from epoch to epoch?
- How do individual weights, biases, and gradients move during SGD?
- How do hidden activations evolve as training progresses?
- How can a reader connect scalar parameter updates to improving predictions?

This work also includes fixing notebook quality issues discovered in review:

- stale saved outputs currently include an old `NameError`
- the install note omits `pandas`
- DataLoader shuffling is sensitive to earlier notebook cell execution order
- the PCA hidden-state visualization currently fits on combined train and test activations

## Goals

- Keep the existing notebook readable for beginners.
- Add explicit instrumentation for parameter, gradient, and activation changes over training.
- Show both macro training behavior and micro SGD updates.
- Keep the training path explicit and easy to follow without relying on opaque hooks.
- Leave the notebook in a clean executed state with consistent outputs.

## Non-Goals

- Turning the notebook into a benchmark or model-selection workflow.
- Adding advanced optimizer variants, schedulers, or regularization techniques.
- Replacing the current simple MLP architecture.
- Building a reusable training framework across notebooks.

## Recommended Approach

Instrument the existing training loop directly and keep the mechanics visible in notebook cells.

This approach is preferred over hooks or a separate observer wrapper because it:

- keeps the notebook’s teaching path aligned with the code that actually runs
- avoids duplicating optimizer logic
- makes each recorded metric easy to explain in plain language
- keeps debugging straightforward when notebook outputs do not match expectations

## Proposed Notebook Structure

### 1. Core Path

Retain the current high-level flow:

1. dataset loading and exploration
2. train/test split and standardization
3. model definition and tensor-shape walkthrough
4. one flower through the network
5. loss, backpropagation, and one SGD update
6. training
7. evaluation and interpretation

This path remains the default reading route for beginners.

### 2. Training Observatory

Extend the training section so each epoch records a compact set of diagnostics.

Per-epoch metrics:

- `train_loss`
- `train_accuracy`
- `test_loss`
- `test_accuracy`
- `fc1_weight_norm`
- `fc1_bias_norm`
- `fc2_weight_norm`
- `fc2_bias_norm`
- `fc1_grad_norm`
- `fc2_grad_norm`
- `hidden_pre_mean`
- `hidden_mean`
- `relu_active_fraction`
- tracked scalar parameters:
  - `fc1.weight[0,0]`
  - `fc1.bias[0]`
  - `fc2.weight[0,0]`
  - `fc2.bias[0]`

Presentation:

- one compact per-epoch history table for selected epochs
- one multi-panel figure covering loss, accuracy, gradient norms, parameter norms, tracked scalar parameters, and activation summaries
- short markdown interpretation after the plots

### 3. SGD Microscope

Add a dedicated section that uses one fixed mini-batch and traces a short sequence of updates, for example five optimizer steps.

For each step, record:

- step index
- loss
- tracked parameter values before the step
- tracked gradients after `loss.backward()`
- expected SGD update for tracked parameters
- actual parameter values after `optimizer.step()`
- predicted probabilities for one selected example in the fixed batch

Presentation:

- a tabular view of parameter and gradient changes across the short run
- one plot of loss vs. step
- one plot of tracked parameter values vs. step
- brief interpretation explaining how gradient sign determines update direction

This section should be intentionally narrow so it acts as a microscope, not the main narrative.

## Detailed Design

### Data Loading And Setup

Keep the current UCI fetch for now, but the notebook text should acknowledge the dependency on internet access. The install guidance will be updated to include `pandas`, matching the imported packages.

The top setup cell will continue to set:

- `torch.manual_seed(42)`
- print formatting for tensors
- DataFrame float formatting

### Split, Standardization, And Reproducibility

The split logic remains stratified and train-only statistics remain the source of standardization parameters.

To make training reproducible regardless of whether earlier cells sample demonstration batches:

- create a dedicated `train_loader_generator = torch.Generator().manual_seed(42)`
- pass that generator to the training DataLoader
- use separate demonstration access patterns that do not accidentally become the implicit source of future training randomness

The notebook may still sample example batches for explanation, but those samples should not silently perturb the later training run.

### Model And Forward-Pass Explanations

Keep the current `IrisNet` and forward-feature helper because it supports both teaching and instrumentation.

No hooks will be introduced. Instead, the training loop will explicitly call `forward_features` when diagnostics need access to hidden activations.

### Training Instrumentation

Add small helper functions for explicit metric extraction, for example:

- `parameter_norms(model)`
- `gradient_norms(model)`
- `tracked_parameter_values(model)`
- `activation_summary(hidden_pre, hidden)`

The main `fit()` function will accept configuration such as:

- whether diagnostics should be collected
- which tracked parameters to record

The function will still train in the same visible order:

1. zero gradients
2. forward pass
3. compute loss
4. backward pass
5. record diagnostics for the current batch or epoch
6. optimizer step

At epoch end, the notebook will aggregate values into a history object that can be converted to a DataFrame for plotting and inspection.

### History Representation

Use a plain Python dictionary of lists or a list of dictionaries, then build a pandas DataFrame after training.

Rationale:

- easy for beginners to inspect
- easy to print and plot
- avoids introducing extra abstractions

### SGD Microscope Helper

Create a dedicated helper that receives:

- a model
- one fixed batch of features and labels
- a learning rate
- number of steps
- tracked parameter specification

For each step, the helper will:

1. compute logits and loss on the same batch
2. backpropagate
3. store tracked parameter values and gradients
4. compute expected updates for tracked parameters
5. apply `optimizer.step()`
6. store actual updated values

This helper will not replace normal training. It exists solely to support the microscope section.

## Visualization Plan

### Epoch-Level Observatory Figure

Use a multi-panel layout such as `3 x 2`:

- panel 1: train and test loss
- panel 2: train and test accuracy
- panel 3: `fc1` and `fc2` gradient norms
- panel 4: `fc1` and `fc2` parameter norms
- panel 5: tracked scalar weights and biases
- panel 6: activation summaries, including ReLU active fraction

The plotting code should use descriptive titles and axis labels so the figures remain presentation-ready.

### Epoch Summary Table

Display a compact DataFrame for selected epochs such as:

- first epoch
- every 25th epoch
- final epoch

This lets readers inspect actual numbers instead of relying only on charts.

### SGD Microscope Plots And Tables

Show:

- a table with one row per optimizer step
- a line plot for loss across steps
- a line plot for tracked parameters across steps

Keep the tracked set small so the table remains readable in the notebook.

## Evaluation And Interpretation Changes

Keep the current confusion-matrix and example-probability evaluation flow.

For the hidden-space PCA visualization:

- fit PCA directions using train hidden activations only
- project both train and test hidden activations into that basis

This keeps the explanatory visualization aligned with the notebook’s earlier anti-leakage message.

## Error Handling And Failure Modes

Potential issues and handling:

- Missing package imports:
  - update the install note to mention `pandas`
- Notebook state drift:
  - re-execute from a clean kernel before saving
- Reproducibility drift from DataLoader shuffling:
  - use a dedicated generator for the training loader
- Visual clutter from too many tracked values:
  - keep tracked parameters to a small named subset
- Misleading interpretation of gradient norms:
  - explain that norms show sensitivity magnitude, not direct model quality

## Testing And Validation Plan

Validation will be notebook-oriented rather than unit-test oriented.

Required checks:

1. execute the notebook end to end with `jupyter nbconvert --execute`
2. confirm the notebook saves without stale error outputs
3. verify the updated install note matches actual imports
4. verify training history contains the expected diagnostics columns
5. verify the SGD microscope section shows exact `expected` and `actual` SGD updates that match numerically for tracked parameters
6. verify PCA is fitted on train hidden activations and applied consistently to both splits

## Implementation Steps

1. clean the notebook’s saved state issues and setup note
2. make DataLoader behavior reproducible independent of earlier demo cells
3. extend the training loop with explicit diagnostic collection
4. add the epoch-level observatory plots and summary table
5. add the SGD microscope helper, table, and plots
6. update the hidden-state PCA flow to fit on train only
7. execute the notebook from a clean kernel and save the final outputs

## Success Criteria

The design is successful if:

- the notebook still reads cleanly from top to bottom for a first-time learner
- a reader can see concrete numeric evidence of how weights, biases, gradients, and activations evolve
- the SGD microscope section makes the parameter update rule tangible rather than abstract
- the notebook executes cleanly and reproducibly from a fresh start
- the final saved notebook is free of stale errors and inconsistent execution state
