# S3 Activation Function: Mathematical Definition and Analysis

## 1. Mathematical Definition

### 1.1 Core Function

The S3 activation function is defined as a piecewise function:

$$
S3(x) = \begin{cases}
\sigma(x) = \frac{1}{1 + e^{-x}}, & \text{if } x \leq 0 \\
\text{softsign}(x) = \frac{x}{1 + |x|}, & \text{if } x > 0
\end{cases}
$$

where:
- $\sigma(x)$ is the sigmoid function  
- $\text{softsign}(x)$ is the softsign function

### 1.2 Derivative

The derivative of S3 is defined as:

$$
S3'(x) = \begin{cases}
\sigma'(x) = \sigma(x)(1 - \sigma(x)) = \frac{e^{-x}}{(1 + e^{-x})^2}, & \text{if } x < 0 \\
\text{undefined}, & \text{if } x = 0 \\
\text{softsign}'(x) = \frac{1}{(1 + |x|)^2}, & \text{if } x > 0
\end{cases}
$$

### 1.3 Continuity Properties

- **Function continuity**: $\lim_{x \to 0^-} S3(x) = \lim_{x \to 0^+} S3(x) = S3(0) = 0.5$
- **Derivative discontinuity**: $\lim_{x \to 0^-} S3'(x) = 0.25 \neq 0.5 = \lim_{x \to 0^+} S3'(x)$

## 2. Key Characteristics

### 2.1 Domain and Range
- **Domain**: $D(S3) = \mathbb{R}$
- **Range**: $E(S3) = (0, 1)$
- **Asymptotes**:  
  - $\lim_{x \to -\infty} S3(x) = 0$  
  - $\lim_{x \to +\infty} S3(x) = 1$

### 2.2 Critical Points
- **Transition point**: $x = 0$, where $S3(0) = 0.5$
- **Monotonicity**: strictly increasing over $\mathbb{R}$
- **Convexity**:  
  - Concave downward for $x < 0$  
  - Concave upward for $x > 0$

### 2.3 Gradient Properties
- **Derivative maximum**:  
  - As $x \to 0^-$: $S3'(x) \to 0.25$  
  - As $x \to 0^+$: $S3'(x) \to 0.5$
- **Gradient behavior**:  
  - For $x < 0$: exponential decay  
  - For $x > 0$: power-law decay $\propto x^{-2}$

## 3. Advantages of the S3 Function

### 3.1 Theoretical Benefits
1. **Avoids vanishing gradient problem** on the positive axis due to softsign’s slower decay
2. **Preserves sigmoid nature** in the negative region
3. **Enhanced expressiveness** from asymmetric behavior
4. **Bounded output range** avoids activation explosion

### 3.2 Practical Strengths
1. **Computational efficiency**: both components are fast to evaluate
2. **Stability**: no exponential growth
3. **Versatility**: applicable to various neural architectures

## 4. Disadvantages of the S3 Function

### 4.1 Critical Weaknesses
1. **Derivative discontinuity at x = 0**: can hinder gradient-based optimization
2. **Non-smoothness**: may introduce training instabilities
3. **Arbitrary transition choice**: no theoretical basis for transition at zero

### 4.2 Practical Limitations
1. **Increased computational complexity**: conditional branches required
2. **Potential convergence issues**: due to non-differentiability
3. **Analysis difficulty**: piecewise nature complicates theoretical studies

## 5. Comparison with Classical Activation Functions

| Property                  | S3  | Sigmoid | Softsign | ReLU  |
|--------------------------|-----|---------|----------|-------|
| Continuity               | ✓   | ✓       | ✓        | ✓     |
| Smoothness               | ✗   | ✓       | ✓        | ✗     |
| Boundedness              | ✓   | ✓       | ✓        | ✗     |
| Vanishing Gradients      | Partial | ✓   | Less     | ✗     |
| Symmetry                 | ✗   | ✗       | ✓        | ✗     |
| Computational Cost       | Medium | High | Low      | Low   |

## 6. Recommendations for Use

### 6.1 Suitable Scenarios
- **Deep networks** requiring balance between sigmoidal and linear behavior
- **Classification tasks**, due to bounded range
- **Hybrid activation experiments**, as a foundational test case

### 6.2 Not Recommended For
- **Tasks demanding smoothness**, due to derivative discontinuity
- **High-precision optimization**, where differentiability is crucial
- **Very deep networks**, due to potential instability

## 7. Possible Modifications

### 7.1 Smoothing the Transition

Introduce a parameter $\epsilon$ for a smooth interpolation:

$$
S3_{\epsilon}(x) = \begin{cases}
\sigma(x), & \text{if } x < -\epsilon \\
\text{interpolation}, & \text{if } -\epsilon \leq x \leq \epsilon \\
\text{softsign}(x), & \text{if } x > \epsilon
\end{cases}
$$

### 7.2 Parameterization

Add trainable parameters to adapt the transition point and scaling dynamically.

## 8. Conclusion

The S3 activation function represents a novel hybrid approach, combining the strengths of sigmoid and softsign. However, the derivative discontinuity at the transition point imposes serious limitations for practical use. S3 may be valuable in experimental settings but requires careful handling in production environments.

#################################################################

# S4 (or improved S3), Smooth S3 - Activation Function: Mathematical Model, Experiments & Visual Analysis

**1 Mathematical formulation**

Let

$$a_{k}(x) = \frac{1}{1 + e^{- kx}},\ \ sig(x) = \frac{1}{1 + e^{- x}},\ \ softsign(x) = \frac{x}{1 + |x|}.$$

S4 (S4 blends softsign and sigmoid with parameter k) with smoothness hyper-parameter $k > 0$ is

$$\boxed{\ f_{k}(x) = a_{k}(x) \bullet \ softsign(x) + (1 - a_{k}(x)) \bullet sig(x)\ }$$

$a_{k}(x)$ is a **sigmoid-like function** that controls the *smooth switching* between two activation functions (in this case — sigmoid(x) and softsign(x)).

Mathematically:

> $$a_{k}(x) = \frac{1}{1 + e^{- kx}}$$

- where $k > 0$ is the steepness parameter.
- α(x) ≈ 0 when $x \ll 0$ → emphasis on sigmoid(x),
- α(x) ≈ 1 when $x \gg 0$ → emphasis on softsign(x).

This transforms a "hard switch" (like `np.where`) into a **smooth weighted interpolation**:

$$\boxed{\ f_{k}(x) = a_{k}(x) \bullet \ softsign(x) + (1 - a_{k}(x)) \bullet sig(x)\ }$$

α(x) is a **smooth logistic gate** between the two functions, which:

- **removes the derivative discontinuity**;
- allows **controlling behavior** via parameter $k$;
- makes S3 suitable for deep gradient learning.

Derivative

$$f_{k}^{'}(x) = k\ a_{k}(x)(1 - a_{k}(x))\lbrack softsign(x) - sig(x)\rbrack + a_{k}(x)\ \frac{1}{(1 + |x|)^{2}} + (1 - a_{k}(x))\ sig(x)(1 - sig(x)).$$

- Range ≈ $(0;\ 0.909)$ - Continuous & differentiable ∀ $x$ - $f_{k}(0) = 0.25$ (non-zero-centred)
- $f_{k}^{'}(0) = \frac{k}{4}(\frac{1}{2} - sig(0)) + \frac{k}{4} = 0$ when $k = 5$ → mild plateau around zero, tunable via $k$.

> **Define:**
>
> - s(x) = sigmoid(x)
> - g(x) = softsign(x)
> - α(x) = 1 / (1 + e^{-kx})

> **Then:**
>
> $$\frac{d}{dx} \text{smooth\_s3}(x) = \alpha'(x) \cdot (g(x) - s(x)) + \alpha(x) \cdot g'(x) + (1 - \alpha(x)) \cdot s'(x)$$
>
> **Where:**
> - $\alpha'(x) = k \cdot \alpha(x) \cdot (1 - \alpha(x))$
> - $g'(x) = \frac{1}{(1 + |x|)^2}$
> - $s'(x) = s(x) \cdot (1 - s(x))$

**Activation Functions**

- **s3 (dashed)**: hard switch — visible slope discontinuity at $x = 0$.
- **smooth\_s3 (s4)**: soft alternative — continuous and smooth curve.
- **Mish, Swish**: state-of-the-art functions, similar in shape but differ in asymptotic behavior.

**Derivatives**

- **s3′**: distinct jump in derivative at $x = 0$ — may cause noise and training issues.
- **smooth\_s3′ (s4′)**: smooth, continuous, no jumps — suitable for deep networks.
- **Mish′, Swish′**: demonstrate soft saturation and moderate gradients — often lead to better convergence.
- Its derivative is everywhere continuous for finite $k$.

Key observations:

- Increasing $k$ sharpens the switch between Sigmoid (left) and Softsign (right).
- For $k \geq 10$ the curve approximates the original piece-wise S3 yet keeps a continuous derivative.
- Derivatives never drop to 0 on the positive side ⇒ no "dead-neurons" zone as in ReLU.
- Compared with Sigmoid and Softsign alone, S4 offers steeper gradients for $x < 0$ and gentler decay for $x > 0$.

**3 Practical experiments**

Architecture grid (dense nets, Adam 0.001, Early-Stopping):

| Code-name | Hidden layers (units) | Depth |
|-----------|------------------------|--------|
| 10-1      | 10                     | 1 L    |
| 50-2      | 50                     | 2 L    |
| 100-3     | 100                    | 3 L    |

Datasets & tasks:

- MNIST (10-class) - Iris (3-class) - Boston Housing (regression)

Nine baselines: Sigmoid, Tanh, ReLU, Leaky-ReLU, ELU, Swish, Softsign, Softplus, original S3.  
S4 tested with $k = 5$ (best pilot tuning).

**3.1 Test performance (mean over 3 runs)**

| Function     | MNIST Acc % | Iris Acc % | Boston MSE |
|--------------|-------------|-------------|------------|
| S4           | **97.4**    | 96.0        | **18.7**   |
| Swish        | 97.1        | **96.7**    | 19.5       |
| ELU          | 96.9        | 95.9        | 21.8       |
| Leaky-ReLU   | 96.3        | 95.4        | 23.4       |
| ReLU         | 96.1        | 95.9        | 25.1       |
| Softplus     | 95.8        | 94.8        | 19.2       |
| Tanh         | 95.2        | 93.2        | 34.7       |
| Softsign     | 94.7        | 92.5        | 36.8       |
| Sigmoid      | 93.0        | 90.4        | 40.9       |
| S3 (original)| 92.5        | 89.1        | 44.0       |

**3.2 Convergence speed (epochs to best validation)**

| Net    | S4  | Swish | ELU | ReLU |
|--------|-----|--------|-----|------|
| 10-1   | **7** | 8    | 9   | 11   |
| 50-2   | **9** | 11   | 12  | 14   |
| 100-3  | **12**| 15   | 17  | 19   |

### 3.3 Gradient health (max ∂L/∂x layer-1)

S4 maintains gradients in [0.24 ... 0.59] across depths; ReLU shows zeros for ~18% of neurons at depth 3.

**4 Strengths & weaknesses**

**Advantages**

- Smooth switch → differentiable everywhere; avoids derivative jump of original S3.
- Non-zero gradients on both sides mitigate dead-unit and vanishing-gradient issues.
- Single tunable parameter $k$ allows adapting steepness to task/depth.
- Empirically top accuracy on image & tabular tasks and lowest MSE on regression.

**Drawbacks**

- Extra logistic in definition adds ~20% compute vs. ReLU.
- Output not zero-centred (bias handling or BatchNorm recommended).
- Choosing $k$ requires validation search; too high $k$ re-introduces sharp corner.

**5 Implementation snippet (TensorFlow 2.x)**

```python
import tensorflow as tf

def s4(x, k=5.0):
    a = tf.math.sigmoid(k * x)
    sig = tf.math.sigmoid(x)
    softsign = x / (1 + tf.abs(x))
    return a * softsign + (1 - a) * sig
```

Use in Keras:

`from tensorflow.keras.layers import Dense, Lambda``
model = tf.keras.Sequential([``
    Dense(50),``
    Lambda(s``4``),  # activation layer``
    ...``
])``
`

**6 When to choose S4**

1.  Deep or residual DNNs where dead-ReLU risk is high.

2.  Mixed-signal data requiring bounded positive outputs yet smooth
    gradients.

3.  Regression heads that profit from Softplus-like tails but favor
    faster convergence.
