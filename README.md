# Neural Network Tutorials

Four tutorials on the same problem: Recognizing handwritten MNIST digits. The best is to read them in order.

| # | Notebook | What changes | Parameters | Validation accuracy |
|---|---|---|---|---|
| 1 | [`basic_neural_net.ipynb`](basic_neural_net.ipynb) | dense network in NumPy, every gradient by hand | 7,960 | 90.8% |
| 2 | [`basic_neural_net_pytorch.ipynb`](basic_neural_net_pytorch.ipynb) | **the framework** — same network, PyTorch | 7,960 | 92.7% |
| 3 | [`cnn_pytorch.ipynb`](cnn_pytorch.ipynb) | **the architecture** — convolution | 9,098 | 98.1% |
| 4 | [`vit_pytorch.ipynb`](vit_pytorch.ipynb) | **the architecture again** — self-attention | 105,098 | ~97.8% |

## The networks

### 1. From scratch, in NumPy

```
x (784×1) → linear W¹,b¹ → ReLU → linear W²,b² → softmax → ŷ (10×1)
```

One hidden layer of 10 ReLU units and a 10-way softmax output, trained by batch
gradient descent on the cross-entropy loss for 1,000 iterations at a learning
rate of 0.5. No PyTorch, no autograd: every matrix product, derivative, and
parameter update is written out explicitly.

### 2. The same network, in PyTorch

The architecture, data, learning rate, and 1,000 full-batch iterations are
unchanged — only the framework differs, so anything that differs in the result
is attributable to PyTorch and nothing else. The notebook closes by checking the
gradients derived by hand in notebook 1 against what autograd computes.

### 3. Convolutional network

```
x (1×28×28) → [conv 3×3, 1→8 → ReLU → maxpool 2] → [conv 3×3, 8→16 → ReLU → maxpool 2] → flatten (784) → linear → ŷ (10)
```

Trained with Adam for 10 epochs at a learning rate of 10⁻³. After two pooling
stages the tensor holds 16 × 7 × 7 = 784 values, so the final linear layer has
the same shape as the one in notebook 1 — it simply sees learned features
instead of raw pixels.

### 4. Vision Transformer

```
image → 16 patches of 7×7 → linear embed → [CLS] + positions → 2 transformer blocks → head → ŷ (10)
```

Width 64, 4 heads, 2 pre-norm blocks, trained with AdamW and a one-cycle
schedule for 20 epochs. Attention is implemented from scratch rather than with
`nn.MultiheadAttention`, and the notebook then rebuilds the identical model from
PyTorch's built-ins and shows the two agree.

## The data

Nothing to download. All four notebooks read `data/train.csv.gz`, which ships
with this repository — 42,000 labelled 28×28 handwritten digits, gzipped to
8.5 MB. `pandas` reads it compressed, so cloning is the whole setup.

The loading cell checks a few other locations too: an uncompressed
`data/train.csv`, the notebook's own folder, and Google Drive when running
inside Colab. An existing copy will be found if you have one.

The images come from the MNIST database of handwritten digits, in the CSV
layout Kaggle's Digit Recognizer uses — one row per image, a `label` column
followed by `pixel0` through `pixel783`.

## Running the notebooks

```bash
git clone https://github.com/RibeiroResearch/neural-nets-tutorials.git
cd neural-nets-tutorials
```

```bash
pip install numpy pandas matplotlib jupyter
jupyter notebook basic_neural_net.ipynb
```

Notebooks 2 to 4 additionally need PyTorch:

```bash
pip install torch
jupyter notebook basic_neural_net_pytorch.ipynb
```

All four run cleanly from top to bottom. The first three train in about a minute
on a laptop CPU; `vit_pytorch.ipynb` takes a few minutes, since it runs 20
epochs. No GPU is required, though the PyTorch notebooks will use CUDA or Apple
silicon (MPS) automatically when one is available.
