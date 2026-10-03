# On the design of encoding (positional encoding)

Attention computes $\mathbf{q}_i \cdot \mathbf{k}_j$, a bilinear form. So the test for any hand-designed encoding $\phi(p)$ is:

> Can the position relationship I care about be written as $\phi(p_i)^{\mathsf{T}}\mathbf{S}\,\phi(p_j)$ for some $\mathbf{S}$?

Distance is the common case, but it isn't the only relationship worth having. Others: relative offset (which direction $j$ lies from $i$), same row or column, ordering. Each is a different function of the pair, and each needs the encoding to make it bilinear.

Three encodings and what each one admits:

| $\phi(p)$                               | $\phi(p_i)\cdot\phi(p_j)$                   | Description                                              |
| --------------------------------------- | ------------------------------------------- | -------------------------------------------------------- |
| $(x, y)$                                | $x_ix_j + y_iy_j$                           | alignment measured from the origin (rarely what we want) |
| $(x, y, \lVert p \rVert^2, 1)$          | $-\lVert p_i - p_j \rVert^2$                | locality                                                 |
| $(\sin \omega_k p,\ \cos \omega_k p)_k$ | $\sum_k \cos\big(\omega_k (p_i - p_j)\big)$ | relative position                                        |

The third row is why sinusoids are the standard choice, and it's one line of trigonometry:

$$\sin(\omega p_i)\sin(\omega p_j) + \cos(\omega p_i)\cos(\omega p_j) = \cos\big(\omega(p_i - p_j)\big)$$

The dot product depends only on the **difference** between positions, never on the absolute values. So sinusoidal encoding hands you translation-invariant attention for free — a patch 3 to the left is compared the same way wherever in the image the pair sits. With enough frequencies you can approximate any shift-invariant kernel this way.

Row two is what §4 of the notebook does, and it's the standard lift for making squared distance linear — the same $\lVert a - b \rVert^2 = \lVert a \rVert^2 - 2a\cdot b + \lVert b \rVert^2$ expansion that appears in kernel methods.

So the general framing: **designing a non-learned positional encoding is designing a feature map, chosen so the comparison you want becomes an inner product.** That's the kernel trick, applied to position. And a learned embedding gives the model free parameters and let training find a feature map that makes its own useful comparisons linear.



