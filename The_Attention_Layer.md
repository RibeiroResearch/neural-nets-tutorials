# The Attention Layer

This tutorial covers the basic components of the **attention layer** in transformers. The transformers architecture implements the concept of attention, and it was originally proposed to solve the language-translation problem (e.g., translate from English to German). Figure 1 shows the flow diagram of the standard transformer model.

The transformer architecture consists of two main components: an encoder and a decoder. The encoder's goal is to "learn" a representation of the input language (e.g., English). The decoder has the combined goals of "learning" the output language and learning how the output language relates to the input language to achieve translation. 

![https://en.wikipedia.org/wiki/Transformer_(deep_learning)#/media/File:Transformer,_full_architecture.png](https://upload.wikimedia.org/wikipedia/commons/3/34/Transformer%2C_full_architecture.png?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=original)

**Figure** 1: Standard encoder-decoder transformer architecture (with pre-layer normalization). Figure from Wikipedia: https://en.wikipedia.org/wiki/Transformer_(deep_learning).  



This tutorial focuses on the basic (multi-head) self-attention layer of the encoder component. The self-attention layer is represented by the blue box in Figure 2.

<img src="https://upload.wikimedia.org/wikipedia/commons/9/92/Transformer%2C_one_encoder_block.png?utm_source=en.wikipedia.org&utm_campaign=imageinfo&utm_content=original" alt="https://upload.wikimedia.org/wikipedia/commons/9/92/Transformer%2C_one_encoder_block.png?utm_source=en.wikipedia.org&utm_campaign=imageinfo&utm_content=original" style="zoom:50%;" />

**Figure** 2: The encoder layer. This layer's main components are the **attention layer** and the **feed-forward network**. Figure from Wikipedia: https://en.wikipedia.org/wiki/Transformer_(deep_learning).  

In particular, these notes focus on the Vision Transformer (ViT) (Dosovitskiy et al.), which uses only the encoder, treating image patches as the tokens of a sentence.

## The sequence of steps 

**1. Extract the set of non-overlapping image patches.** Cut an image $I \in \mathbb{R}^{C \times H \times W}$ into a set of $N$ non-overlapping square patches $\mathcal{P} \;=\; \big\{ \text{patch}_0,\dots,\text{patch}_{N-1}\big\}$ with $\text{patch}_i \in \mathbb{R}^{C \times P \times P}$. 

- $N = HW/P^2$, assuming $P$ divides $H$ and $W$.

**2. Convert the patches into a set of tokens.** Each patch is vectorized and mapped to the model width, so the token set is the image of $\mathcal{P}$ under one shared map:
$$
\begin{align}
\mathcal{T} \;=\; \big\{\, \mathbf{t}_i \;=\; \mathbf{W}_{\texttt{tokenize}}\,\mathrm{vec}(\text{patch}_i) + \mathbf{b}
\;:\; \text{patch}_i \in \mathcal{P} \,\big\},
\qquad \mathbf{t}_i \in \mathbb{R}^{d \times 1}
\end{align}
$$

- $\mathbf{W}_{\texttt{tokenize}} \in \mathbb{R}^{d \times CP^2}$.
- $\mathbf{b} \in \mathbb{R}^{d \times 1}$.

**3. Assemble the input matrix.** Each row of the matrix is a token with the added positional embeddings:
$$
\begin{align}
\mathbf{T}_{\texttt{in}} \;=\;
\begin{bmatrix}
(\mathbf{t}_0 + \mathbf{p}_0)^{\mathsf{T}} \\ \vdots \\ (\mathbf{t}_{N-1} + \mathbf{p}_{N-1})^{\mathsf{T}}
\end{bmatrix}
\;\in\; \mathbb{R}^{N \times d}
\end{align}
$$

where $\mathbf{p}_i \in \mathbb{R}^{d \times 1}$ is a learned positional embedding for patch position $i$. 

**4. Layer normalization.** LN acts on a single token's vector $\mathbf{t} \in \mathbb{R}^{d \times 1}$, normalizing over the $d$ components $t_c$ of $\mathbf{t}$. The normalization layer is given by:
$$
\begin{align}
\operatorname{LN}(\mathbf{t}) = \boldsymbol{\gamma} \odot \frac{\mathbf{t} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \boldsymbol{\beta},
\end{align}
$$

where $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^{d \times 1}$ are a learned per-component scale and shift, shared across tokens. For details on the parameters $\boldsymbol{\gamma}, \boldsymbol{\beta}$ see notes at the end of this document. The mean $\mu$ and variance $\sigma^2$ are given by: 
$$
\begin{align}
\mu = \frac{1}{d}\sum_{c=1}^{d} t_c,
\quad\text{and}\quad
\sigma^2 = \frac{1}{d}\sum_{c=1}^{d}(t_c - \mu)^2.
\end{align}
$$
The operator $\odot$ is the **Hadamard product**, which is element-wise multiplication of two vectors or matrices of the same shape. 

LN is applied to each row of $\mathbf{T}_{\texttt{in}}$ independently. Here, we will follow the diagrams in Figures 1 and 2 and apply LN to the tokens prior to passing them to the attention layer (i.e., pre-LN, as in ViT):
$$
\begin{align}
\mathbf{X} \;=\; \operatorname{LN}(\mathbf{T}_{\texttt{in}}) \;=\; \begin{bmatrix} \operatorname{LN}(\mathbf{t}_0 + \mathbf{p}_0)^{\mathsf{T}} \\ \vdots \\ \operatorname{LN}(\mathbf{t}_{N-1} + \mathbf{p}_{N-1})^{\mathsf{T}} \end{bmatrix} \;\in\; \mathbb{R}^{N \times d}.
\end{align}
$$
We denote the matrix of normalized tokens by $\mathbf{X}$. 

**5. Multi-head self-attention.** The attention sub-layer (LN, MHSA, and residual) takes the $N \times d$ token matrix and returns a matrix of the same shape. Each output row is a new representation of one token, mixed from all tokens, including itself.

**Queries, keys, values**. In this step, each row of $\mathbf{X}$ is *linearly mapped* into the **query**, **key**, and **value** spaces by learned matrices $\mathbf{W}^{Q}_h$, $\mathbf{W}^{K}_h$, and $\mathbf{W}^{V}_h$. There is one set of matrices per attention head $h$, shared across all $N$ tokens. See notes at the end of this document for a clarification on the use of the term *linearly mapped* instead of *projection*.

Given the (learned) matrices of the linear maps, the corresponding mapped tokens for each head $h = 1,\dots,H$, with head width $d_k = d/H$, are:
$$
\begin{align}
\mathbf{Q}_h = \mathbf{X}\mathbf{W}^{Q}_h, \qquad \mathbf{K}_h = \mathbf{X}\mathbf{W}^{K}_h, \qquad \mathbf{V}_h = \mathbf{X}\mathbf{W}^{V}_h,
\end{align}
$$
with $\mathbf{W}^{Q}_h, \mathbf{W}^{K}_h, \mathbf{W}^{V}_h \in \mathbb{R}^{d \times d_k}$, and consequently, $\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h \in \mathbb{R}^{N \times d_k}$. Their $i$-th rows, $\mathbf{q}^{(h)\mathsf{T}}_i$, $\mathbf{k}^{(h)\mathsf{T}}_i$ and $\mathbf{v}^{(h)\mathsf{T}}_i$, are the query, key and value of token $i$, respectively.

**Attention weights**. The *softmax* function is applied to each row:
$$
\begin{align}
\mathbf{A}_h \;=\; \operatorname{softmax}\!\left( \frac{\mathbf{Q}_h \mathbf{K}_h^{\mathsf{T}}}{\sqrt{d_k}} \right) \in \mathbb{R}^{N \times N}, \qquad a^{(h)}_{ij} = \frac{\exp\!\big(\mathbf{q}^{(h)\mathsf{T}}_i\mathbf{k}^{(h)}_j/\sqrt{d_k}\big)}{\sum_{l=0}^{N-1}\exp\!\big(\mathbf{q}^{(h)\mathsf{T}}_i\mathbf{k}^{(h)}_l/\sqrt{d_k}\big)}
\end{align}
$$
Each row of $\mathbf{A}_h$ is non-negative and sums to 1. Entry $a^{(h)}_{ij}$ is how much token $i$ attends to token $j$.

**Head output**. Each head returns a weighted average of the values:
$$
\begin{align}
\mathbf{Z}_h \;=\; \mathbf{A}_h \mathbf{V}_h \in \mathbb{R}^{N \times d_k}, \qquad \text{with}\quad \mathbf{z}^{(h)}_i = \sum_{j=0}^{N-1} a^{(h)}_{ij}\, \mathbf{v}^{(h)}_j.
\end{align}
$$
**Output of the attention layer**. Finally, the heads are concatenated and mapped back to width $d$, then added to the input through a residual connection:
$$
\begin{align}
\operatorname{MHSA}(\mathbf{X}) = \big[\,\mathbf{Z}_1 \;\cdots\; \mathbf{Z}_H\,\big]\,\mathbf{W}^{O}, \quad \mathbf{W}^{O} \in \mathbb{R}^{d \times d},
\end{align}
$$
where $\mathbf{W}^{O}$ is the learned output matrix, which combines the heads (see notes at the end of this document for details on $\mathbf{W}^{O}$). The concatenation has $H d_k = d$ columns. The output tokens are:
$$
\begin{align}
\mathbf{T}_{\texttt{attn}} = \mathbf{T}_{\texttt{in}} + \operatorname{MHSA}\!\big(\operatorname{LN}(\mathbf{T}_{\texttt{in}})\big) \in \mathbb{R}^{N \times d}
\end{align}
$$
The input and output of the attention layer are $N$ tokens of width $d$. After passing through the attention layer, each output token becomes a mixture of all tokens, weighted by content similarity.

## Additional notes

### Map vs. projection

In the machine learning literature, the maps $\mathbf{W}^{Q}_h, \mathbf{W}^{K}_h, \mathbf{W}^{V}_h \in \mathbb{R}^{d \times d_k}$ are commonly called *projections* (e.g., "query, key, and value projections"), following Vaswani et al. (2017). The term is used loosely, to mean any learned linear map, typically one that changes dimension. It is not a projection in the linear-algebraic sense of an idempotent operator ($P^2 = P$). The three matrices are rectangular and unconstrained, so that sense does not apply. Each such map can, however, be factored (via the SVD) into an orthogonal projection onto a $d_k$-dimensional subspace of $\mathbb{R}^d$ followed by an invertible change of coordinates. 

The SVD factorization is as follows. Let $\mathbf{W}$ denote any of $\mathbf{W}^{Q}_h, \mathbf{W}^{K}_h, \mathbf{W}^{V}_h$. Take the thin SVD $\mathbf{W} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{R}^{\mathsf{T}}$, with $\mathbf{U} \in \mathbb{R}^{d \times d_k}$ having orthonormal columns and $\boldsymbol{\Sigma}, \mathbf{R} \in \mathbb{R}^{d_k \times d_k}$. Then for a token $\mathbf{t}$:
$$
\begin{align}
\mathbf{t}^{\mathsf{T}}\mathbf{W} \;=\; \underbrace{(\mathbf{t}^{\mathsf{T}}\mathbf{U})}_{\text{projection coordinates}}\;\underbrace{\boldsymbol{\Sigma}\mathbf{R}^{\mathsf{T}}}_{\text{scale, rotate}}
\end{align}
$$

- $\mathbf{t}^{\mathsf{T}}\mathbf{U}$ gives the coordinates, in the basis $\mathbf{U}$, of the orthogonal projection $\mathbf{U}\mathbf{U}^{\mathsf{T}}\mathbf{t}$ onto the $d_k$-dimensional subspace $\operatorname{col}(\mathbf{U}) \subset \mathbb{R}^d$. This part is a genuine projection. The components of $\mathbf{t}$ orthogonal to that subspace are discarded.
- $\boldsymbol{\Sigma}\mathbf{R}^{\mathsf{T}}$ then rescales and rotates those coordinates. If $\mathbf{W}$ has full rank $d_k$, this step is invertible.

Formally, we have: 
$$
\text{orthogonal projections} \;\subset\; \text{projections} \;\subset\; \text{linear maps} \;\subset\; \text{maps}\notag
$$

- **Map:** any function $f: A \to B$. With no qualifier, it need not even be linear.
- **Linear map:** $f(\alpha\mathbf{x} + \beta\mathbf{y}) = \alpha f(\mathbf{x}) + \beta f(\mathbf{y})$. Between $\mathbb{R}^d$ and $\mathbb{R}^{d_k}$, every linear map is given by a matrix.
- **Projection:** a linear map from a space to itself with $P^2 = P$ (idempotent operator).
- **Orthogonal projection:** a projection with $P = P^{\mathsf{T}}$.

Every projection is therefore a linear map, but not conversely. 

### Layer normalization

Vectors $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$ are **learned parameters**: a per-component **scale** and **shift**, both in $\mathbb{R}^{d \times 1}$.
$$
\begin{align}
\operatorname{LN}(\mathbf{t})_c = \gamma_c \, \hat{t}_c + \beta_c, \qquad \hat{t}_c = \frac{t_c - \mu}{\sqrt{\sigma^2 + \epsilon}}
\end{align}
$$
Normalization alone forces every token to have mean 0 and variance 1 across its components. That is a hard constraint the next layer might not want. The parameters $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$ let the network choose a per-component scale and offset other than 1 and 0. Because they are shared across tokens, while $\mu$ and $\sigma^2$ are computed per token, they cannot restore each token's original statistics. What LN removes, by design, is the per-token mean and spread. 

**Practical details:**

- They are usually initialized as $\boldsymbol{\gamma} = \mathbf{1}$ and $\boldsymbol{\beta} = \mathbf{0}$, so LN starts as pure normalization.
- They are **shared across all $N$ tokens**. The same $\boldsymbol{\gamma}, \boldsymbol{\beta}$ apply to every row of $\mathbf{T}_{\texttt{in}}$.
- Each LN layer has **its own pair**. In a ViT, every block has two of them: one before attention and one before the MLP.

### The output matrix $\mathbf{W}^{O}$

Like $\mathbf{W}^{Q}_h$, $\mathbf{W}^{K}_h$, and $\mathbf{W}^{V}_h$, the matrix $\mathbf{W}^{O}$ is a **learned parameter**: it is initialized randomly and updated by backpropagation during training. In code, it is a linear layer from $d$ to $d$, often with a bias $\mathbf{b}^{O} \in \mathbb{R}^{d \times 1}$, which we omit here.

$\mathbf{W}^{O}$ has two roles. First, it maps the concatenated head outputs back to the model width $d$, so that the result can be added to $\mathbf{T}_{\texttt{in}}$ through the residual connection. Second, it is the only place where the outputs of the different heads are combined.

The second role becomes explicit if we partition the rows of $\mathbf{W}^{O}$ into $H$ blocks, one per head:
$$
\begin{align}
\mathbf{W}^{O} = \begin{bmatrix} \mathbf{W}^{O}_1 \\ \vdots \\ \mathbf{W}^{O}_H \end{bmatrix}, \qquad \mathbf{W}^{O}_h \in \mathbb{R}^{d_k \times d}.
\end{align}
$$
By block multiplication, the multi-head output becomes:
$$
\begin{align}
\operatorname{MHSA}(\mathbf{X}) \;=\; \big[\,\mathbf{Z}_1 \;\cdots\; \mathbf{Z}_H\,\big]\,\mathbf{W}^{O} \;=\; \sum_{h=1}^{H} \mathbf{Z}_h \mathbf{W}^{O}_h.
\end{align}
$$
Concatenating the heads and multiplying by $\mathbf{W}^{O}$ is therefore equivalent to letting each head map its own output back to width $d$ and summing the results. Each head contributes to all $d$ output components, not only to its own $d_k$-wide slice.

## References

- Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
- Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *International Conference on Learning Representations (ICLR)*.

