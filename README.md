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

## Beyond the sequence

The four above all train from random weights on the same digits, which is what
makes their accuracies comparable. Two further notebooks do neither, so they sit
outside that table.

### Transfer learning

[`vit_transfer_learning.ipynb`](vit_transfer_learning.ipynb) loads an
ImageNet-pretrained `vit_b_16` — the notebook 4 architecture at 824 times the
size — replaces its classification head, freezes the backbone, and classifies 37
cat and dog breeds from **20 training images per class**. The comparison it runs
is a pretrained backbone against a randomly initialized one, holding the
architecture, data, head, optimizer and epochs fixed, so the gap between them
measures what the ImageNet training bought rather than asserting it.

| | Test accuracy | Parameters trained |
|---|---|---|
| random backbone + linear head | 0.0886 | 28,453 |
| pretrained backbone + linear head | **0.9092** | 28,453 |
| pretrained, attention layers fine-tuned | 0.8930 | 28,376,869 |

The third row is the one worth sitting with: a thousand times more trained
parameters, and it finished behind the linear probe. Section 13 takes that
apart.

It downloads about 1.1 GB and takes roughly twenty minutes end to end.

### Attention by hand

[`attention_by_hand.ipynb`](attention_by_hand.ipynb) is a companion to notebook
4, filling the gap between its Section 5, which derives the attention formula,
and its Section 11, which plots a trained model's maps. **Nothing in it is
trained.** Every projection is set by hand on tokens whose seven dimensions all
have names, so each attention map is as interpretable as the choice that
produced it.

It answers the question the formula hides — why there are three projection
matrices, and why $\mathbf{W}_q$ and $\mathbf{W}_k$ are two rather than one:

- two different $(\mathbf{W}_q, \mathbf{W}_k)$ pairs with the same product give **bit-identical** attention, so only $\mathbf{W}_q^{\mathsf{T}}\mathbf{W}_k$ is meaningful
- a spatial locality kernel needs an $\mathbf{S}$ with a negative eigenvalue, which no tied $\mathbf{W}^{\mathsf{T}}\mathbf{W}$ can produce — and building it yields a convolution, assembled out of attention
- the same attention matrix carries different payloads through $\mathbf{W}_v$, and reading a constant feature returns exactly 1.0 for every query: attention averages, and cannot count

Pure NumPy, no GPU, seconds to run.

[`attention_explorer.html`](attention_explorer.html) is the interactive version
of the same construction — click any patch to pick the query, switch what the
query and key projections read, and drag the softmax scale to watch attention
slide between a uniform average and a one-hot lookup. Open it in a browser; it
needs no server and no Python.

## The data

**Notebooks 1 to 4: nothing to download.** They read `data/train.csv.gz`, which
ships with this repository — 42,000 labelled 28×28 handwritten digits, gzipped
to 8.5 MB. `pandas` reads it compressed, so cloning is the whole setup.

The loading cell checks a few other locations too: an uncompressed
`data/train.csv`, the notebook's own folder, and Google Drive when running
inside Colab. An existing copy will be found if you have one.

The images come from the MNIST database of handwritten digits, in the CSV
layout Kaggle's Digit Recognizer uses — one row per image, a `label` column
followed by `pixel0` through `pixel783`.

**The transfer-learning notebook is the exception**, necessarily: reusing
weights you did not train means fetching them. It downloads an ImageNet
checkpoint (~330 MB, cached by `torch.hub`) and the Oxford-IIIT Pet dataset
(~800 MB, into `data/oxford-iiit-pet/`, which is gitignored). The dataset comes
from the fast.ai S3 mirror, which served it about twenty times faster than
Oxford's own host when this was written; `torchvision`'s downloader stays on as
the fallback.

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

The transfer-learning notebook also needs `torchvision`, for the pretrained
model and the dataset:

```bash
pip install torch torchvision
jupyter notebook vit_transfer_learning.ipynb
```

All four MNIST notebooks run cleanly from top to bottom. The first three train
in about a minute on a laptop CPU; `vit_pytorch.ipynb` takes a few minutes,
since it runs 20 epochs. No GPU is required, though the PyTorch notebooks will
use CUDA or Apple silicon (MPS) automatically when one is available.

`vit_transfer_learning.ipynb` is the slow one — roughly twenty minutes, most of
it downloading. Its fine-tuning section runs only when a GPU is present; set
`RUN_FINETUNE = False` to skip it and keep the central comparison, which trains
on cached features in seconds.

`attention_by_hand.ipynb` needs only NumPy and matplotlib and runs in seconds.
`attention_explorer.html` needs nothing at all — open it in a browser.
