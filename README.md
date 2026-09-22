# Neural Network Tutorials

Four tutorials, the same problem: Recognizing handwritten MNIST digits. The best is to read them in order.

| # | Notebook | What changes | Parameters | Validation accuracy |
|---|---|---|---|---|
| 1 | [`basic_neural_net.ipynb`](basic_neural_net.ipynb) | dense network in NumPy, every gradient by hand | 7,960 | 90.8% |
| 2 | [`basic_neural_net_pytorch.ipynb`](basic_neural_net_pytorch.ipynb) | **the framework** — same network, PyTorch | 7,960 | 92.7% |
| 3 | [`cnn_pytorch.ipynb`](cnn_pytorch.ipynb) | **the architecture** — convolution | 9,098 | 98.1% |
| 4 | [`vit_pytorch.ipynb`](vit_pytorch.ipynb) | **the architecture again** — self-attention | 105,098 | ~97.8% |

## The networks

### 1. From scratch, in NumPy

```math
\begin{array}{cl}
784 \times 1 & x \\
\downarrow & W^{[1]},\, b^{[1]} \\
10 \times 1 & \\
\downarrow & \text{ReLU} \\
10 \times 1 & \\
\downarrow & W^{[2]},\, b^{[2]} \\
10 \times 1 & \\
\downarrow & \text{softmax} \\
10 \times 1 & \hat{y}
\end{array}
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

```math
\begin{array}{cl}
1 \times 28 \times 28 & x \\
\downarrow & \text{conv } 3\times3,\ 1 \to 8 \\
8 \times 28 \times 28 & \\
\downarrow & \text{ReLU},\ \text{maxpool } 2 \\
8 \times 14 \times 14 & \\
\downarrow & \text{conv } 3\times3,\ 8 \to 16 \\
16 \times 14 \times 14 & \\
\downarrow & \text{ReLU},\ \text{maxpool } 2 \\
16 \times 7 \times 7 & \\
\downarrow & \text{flatten} \\
784 & \\
\downarrow & \text{linear} \\
10 & \hat{y}
\end{array}
```

Trained with Adam for 10 epochs at a learning rate of 10⁻³. After two pooling
stages the tensor holds 16 × 7 × 7 = 784 values, so the final linear layer has
the same shape as the one in notebook 1 — it simply sees learned features
instead of raw pixels.

### 4. Vision Transformer

```math
\begin{array}{cl}
1 \times 28 \times 28 & \text{image} \\
\downarrow & \text{patchify into } 16 \text{ patches of } 7\times7 \\
16 \times 49 & \\
\downarrow & \text{linear embed} \\
16 \times 64 & \\
\downarrow & \text{prepend } [\text{CLS}], \text{ add positions} \\
17 \times 64 & \\
\downarrow & 2 \times \text{transformer block} \\
17 \times 64 & \\
\downarrow & \text{head, on the } [\text{CLS}] \text{ row} \\
10 & \hat{y}
\end{array}
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
