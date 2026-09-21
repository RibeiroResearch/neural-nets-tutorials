# Neural Network Tutorials

Four tutorials on the same problem: Recognizing handwritten MNIST digits. The best is to read them in order.

| # | Notebook | What changes | Parameters | Validation accuracy |
|---|---|---|---|---|
| 1 | [`basic_neural_net.ipynb`](basic_neural_net.ipynb) | dense network in NumPy, every gradient by hand | 7,960 | 90.8% |
| 2 | [`basic_neural_net_pytorch.ipynb`](basic_neural_net_pytorch.ipynb) | **the framework** — same network, PyTorch | 7,960 | 92.7% |
| 3 | [`cnn_pytorch.ipynb`](cnn_pytorch.ipynb) | **the architecture** — convolution | 9,098 | 98.1% |
| 4 | [`vit_pytorch.ipynb`](vit_pytorch.ipynb) | **the architecture again** — self-attention | 105,098 | ~97.8% |

All four use the same data, the same 80/20 split, and the same seed, so the
accuracies are directly comparable — and the comparison is the point of reading
them in order.

Notebook 1 to 2 changes only the framework and gains about two points, almost
all of it from `nn.Linear`'s scale-aware initialization rather than from PyTorch
being faster or cleverer. Notebook 2 to 3 changes only the architecture and
gains five. Isolating the variables is what makes that attribution possible.

Convolution beats the dense networks decisively at essentially the same
parameter budget, cutting the error rate from 7.3% to 1.9%: architecture, not
capacity. The Vision Transformer then spends **eleven times** the CNN's
parameters to land in the same place. The remaining accuracy difference is
within run-to-run noise, so the honest reading is parity at 11.6× the cost, not
a ranking — and that is the most useful lesson in the series. A convolution's
built-in assumptions about images are correct, and at this data scale a model
that must learn those assumptions from 33,600 examples cannot buy an advantage
with parameters alone. When data is scarce a correct inductive bias beats
flexibility; when data is abundant, flexibility wins.

## 1. Handwritten Digit Recognition from Scratch

[`basic_neural_net.ipynb`](basic_neural_net.ipynb) builds a two-layer neural
network using nothing but NumPy. There is no PyTorch, no TensorFlow, and no
autograd: every matrix product, every derivative, and every parameter update is
written out explicitly, so the correspondence between the mathematics and the
code stays visible throughout.

Each step is derived before it is implemented. The notebook is meant to be read
top to bottom and run as you go.

### The network

```
x (784×1) → linear W¹,b¹ → ReLU → linear W²,b² → softmax → ŷ (10×1)
```

One hidden layer of 10 ReLU units, a 10-way softmax output, 7,960 trainable
parameters, trained by batch gradient descent on the cross-entropy loss. It
reaches **90.8% validation accuracy** after 1,000 iterations at a learning rate
of 0.5.

### Contents

| Section | Topic |
|---|---|
| 1–3 | Setup, the data, and its matrix representation (examples as columns) |
| 4 | Architecture, parameter count, and why random initialization breaks symmetry |
| 5 | Forward propagation: broadcasting, why ReLU prevents affine collapse, softmax shift-invariance |
| 6 | One-hot encoding and cross-entropy as maximum likelihood |
| 7 | Backpropagation, derived in full |
| 8–9 | The update rule and the training loop |
| 10–11 | Evaluation, confusion matrix, and what the hidden layer learned |
| 12 | Appendix: verifying the gradients numerically |
| 13 | Summary |

The centrepiece is Section 7, which chains the softmax Jacobian with the
derivative of the logarithm to show why the two combine into a subtraction:

$$dZ^{[2]} = A^{[2]} - Y$$

Section 12 then checks the whole backward pass against centred finite
differences, which agrees to about $10^{-8}$.

### Prerequisites

Matrix multiplication, partial derivatives, and the chain rule. Everything else
is developed in the notebook.

## 2. The Same Network, in PyTorch

[`basic_neural_net_pytorch.ipynb`](basic_neural_net_pytorch.ipynb) rebuilds
notebook 1 with **only the framework changed**. Same architecture, same 7,960
parameters, same data and split, same learning rate, and the same 1,000
full-batch iterations — not mini-batches, not a different optimizer. Anything
that differs is therefore attributable to PyTorch and nothing else.

Every PyTorch facility is introduced by naming what it replaces:

| In the NumPy notebook | Here |
|---|---|
| `initialize_parameters` | `nn.Linear` constructs and initializes |
| `forward_propagation` | `nn.Module.forward` |
| `backward_propagation` — six hand-derived equations | `loss.backward()` (autograd) |
| `update_parameters` | `optimizer.step()` |
| `cross_entropy_loss` + `softmax` | `nn.CrossEntropyLoss` |
| `one_hot` | handled internally by the loss |

### The centrepiece

Section 12 takes the six equations derived by hand in notebook 1, implements
them in PyTorch, and checks them against what autograd computes:

```
parameter       max abs difference      relative   verdict
fc1.weight               3.725e-09     9.415e-08   MATCH
fc1.bias                 4.657e-09     1.795e-07   MATCH
fc2.weight               3.725e-09     1.890e-07   MATCH
fc2.bias                 1.304e-08     2.898e-07   MATCH
```

Agreement at floating-point precision establishes two things at once: the
derivation in notebook 1 was correct, and autograd applies the same chain rule
rather than approximating anything. That is a sharper confirmation than the
finite-difference check, which only shows a gradient is plausible to
$O(\varepsilon^2)$.

It also explains the accuracy difference. `nn.Linear` initializes weights from
$\mathcal{U}(-1/\sqrt{n_{\text{in}}},\, 1/\sqrt{n_{\text{in}}})$ where notebook 1
used a fixed $\mathcal{U}(-0.5, 0.5)$. Set `MATCH_NUMPY_INIT = True` to remove
even that difference.

### Prerequisites

Notebook 1. No prior PyTorch is assumed.

## 3. Convolutional Neural Networks in PyTorch

[`cnn_pytorch.ipynb`](cnn_pytorch.ipynb) keeps PyTorch fixed and changes **the
architecture**, swapping the fully connected hidden layer for convolutions.
Because notebook 2 already covered `nn.Module`, autograd, `nn.CrossEntropyLoss`
and optimizers, this one adds only what convolution requires: 4-D `(N,C,H,W)`
tensors, `nn.Conv2d` and pooling, output-size arithmetic, and `DataLoader`
mini-batches — notebook 2 having stayed full-batch deliberately.

### The network

```
x (1×28×28) → [conv 3×3, 1→8 → ReLU → maxpool 2] → [conv 3×3, 8→16 → ReLU → maxpool 2] → flatten (784) → linear → ŷ (10)
```

9,098 parameters, trained with Adam for 10 epochs at a learning rate of
$10^{-3}$, reaching **98.1% validation accuracy**.

After two pooling stages the tensor is $16 \times 7 \times 7 = 784$ values —
exactly the input dimension of the first notebook. The final linear layer is
therefore the same shape as the one written by hand there; it simply sees 784
learned features instead of 784 raw pixels.

### Contents

| Section | Topic |
|---|---|
| 1–2 | Setup, device selection (CUDA/MPS/CPU), and the data as NCHW tensors |
| 3 | Why convolution: local receptive fields, weight sharing, translation equivariance |
| 4 | Output-size arithmetic and pooling |
| 5 | The architecture, layer by layer |
| 6 | Autograd: what `loss.backward()` actually does, and why gradients accumulate |
| 7 | `CrossEntropyLoss` on logits, log-sum-exp stability, and the Adam update rule |
| 8 | Mini-batch training |
| 9–10 | Evaluation and inspection of the learned filters |
| 11 | Comparison across all three networks |

Section 3 is the conceptual core: it derives the convolution as a local, weight-
shared inner product,

$$S(i,j) = \sum_{u}\sum_{v} K(u,v)\, I(i+u,\, j+v) + b$$

and shows a hand-built Sobel kernel acting on a real digit before any training,
so that the filters recovered by gradient descent in Section 10 can be compared
against it.

### Prerequisites

Notebooks 1 and 2, or equivalent familiarity with backpropagation and PyTorch.

## 4. Vision Transformers

[`vit_pytorch.ipynb`](vit_pytorch.ipynb) replaces convolution with
self-attention, keeping the dataset and split identical. Attention is
implemented from scratch — no `nn.MultiheadAttention`, no
`nn.TransformerEncoderLayer` — so the tensor shapes and the softmax stay
visible. Section 12 then rebuilds the identical model from those built-ins and
proves the two agree by copying the trained weights across: the logits match
bit for bit, and the predictions agree on all 8,400 validation images.

### The network

```
image → 16 patches of 7×7 → linear embed → [CLS] + positions → 2 transformer blocks → head → ŷ (10)
```

$d_{\text{model}} = 64$, 4 heads, 2 pre-norm blocks, 105,098 parameters, trained
with AdamW and a one-cycle schedule for 20 epochs to **~97.8% validation
accuracy**.

### Contents

| Section | Topic |
|---|---|
| 1–2 | Setup and the shared data pipeline |
| 3 | Images as sequences of patches, and the $O(N^2)$ cost that forces patching |
| 4 | The `[CLS]` token, positional embeddings, and why attention needs them |
| 5 | Scaled dot-product attention, including why the $\sqrt{d_k}$ is there |
| 6 | Multi-head attention |
| 7 | Residual connections, LayerNorm vs BatchNorm, the position-wise MLP, pre-norm |
| 8–9 | Assembling the ViT, AdamW, and one-cycle scheduling |
| 10–11 | Evaluation and reading the attention maps |
| 12 | The same model from `torch.nn` built-ins, proved numerically identical |
| 13 | Inductive bias: why the extra parameters buy nothing here |

The core result is Section 5's

$$\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

with the $\sqrt{d_k}$ justified by a variance argument the notebook then
verifies numerically, and permutation equivariance demonstrated in code to show
why positional embeddings are not optional.

Section 11 plots where the `[CLS]` token attends in the final block, per head,
and quantifies how sharply each head concentrates using the entropy of its
attention distribution.

### Prerequisites

Notebook 3, or comfort with PyTorch modules and training loops. The attention
mathematics is developed from scratch.

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
