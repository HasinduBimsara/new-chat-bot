# SLM (Small Language Model) — Architecture Documentation

A ~30M-parameter autoregressive language model built from scratch, trained on the TinyStories dataset to generate coherent short stories. The architecture follows the GPT-2 design with Pre-Norm Transformer blocks, weight tying, and Flash Attention support.

---

## Table of Contents

1. [GPTConfig — The Model Blueprint](#1-gptconfig--the-model-blueprint)
2. [LayerNorm — Stabilizing the Signal](#2-layernorm--stabilizing-the-signal)
3. [CausalSelfAttention — The Core Reasoning Engine](#3-causalselfattention--the-core-reasoning-engine)
4. [MLP — The Nonlinear Feature Transformer](#4-mlp--the-nonlinear-feature-transformer)
5. [Block — The Transformer Block](#5-block--the-transformer-block)
6. [GPT — The Complete Model](#6-gpt--the-complete-model)
7. [The Forward Pass — End to End](#7-the-forward-pass--end-to-end)
8. [Parameter Count Breakdown](#8-parameter-count-breakdown)
9. [Summary — Why Each Layer Matters](#9-summary--why-each-layer-matters)

---

## 1. GPTConfig — The Model Blueprint

```python
@dataclass
class GPTConfig:
    block_size: int        # 128
    vocab_size: int        # 50257
    n_layer: int           # 6
    n_head: int            # 6
    n_embd: int            # 384
    dropout: float = 0.0   # 0.1 at training time
    bias: bool = True
```

`GPTConfig` is a Python **dataclass** that holds every hyperparameter the model needs. It acts as a single source of truth passed to every sub-module during construction. No layer decides its own size — everything derives from this config.


| Parameter    | Value  | Purpose                                                                                                                  |
| ------------ | ------ | ------------------------------------------------------------------------------------------------------------------------ |
| `block_size` | 128    | Maximum sequence length (context window). The model can attend to at most 128 tokens at a time.                          |
| `vocab_size` | 50,257 | Size of the tokenizer vocabulary (GPT-2 BPE). Determines the input/output dimensionality.                                |
| `n_layer`    | 6      | Number of stacked Transformer blocks. Depth of the network — more layers = more capacity to learn hierarchical patterns. |
| `n_head`     | 6      | Number of parallel attention heads per block. Each head learns different relationship patterns.                          |
| `n_embd`     | 384    | Embedding dimension — the width of every hidden representation in the network.                                           |
| `dropout`    | 0.1    | Probability of dropping activations during training for regularization.                                                  |
| `bias`       | True   | Whether to include bias terms in linear layers and layer norms.                                                          |


**Key derived value:** Each attention head operates on `n_embd / n_head = 384 / 6 = 64` dimensions.

**Contribution to the architecture:** GPTConfig ensures architectural consistency. Every layer reads its dimensions from this single object, making the model easy to scale up or down by simply changing these numbers. Want a bigger model? Increase `n_embd` and `n_layer`. Want a smaller one? Decrease them. The rest of the code adapts automatically.

---

## 2. LayerNorm — Stabilizing the Signal

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        self.weight = nn.Parameter(torch.ones(ndim))     # learnable scale (γ)
        self.bias = nn.Parameter(torch.zeros(ndim))       # learnable shift (β), optional
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
```

### What It Does

Layer Normalization normalizes each token's feature vector **independently** across the embedding dimension. For a single token's feature vector x ∈ ℝ³⁸⁴:

```
x̂ᵢ = (xᵢ - μ) / √(σ² + ε) · γᵢ + βᵢ
```

where:

- **μ** and **σ²** are the mean and variance computed across the 384 features of that token
- **γ** (weight) and **β** (bias) are learnable per-feature affine parameters
- **ε = 10⁻⁵** prevents division by zero

### Why We Use It

- **Training stability:** Deep networks suffer from internal covariate shift — the distribution of inputs to each layer changes as the layers below are updated. LayerNorm re-centers and re-scales the activations at every step, keeping gradients well-behaved.
- **Independent of batch size:** Unlike BatchNorm, LayerNorm normalizes per-token, so it works identically for batch size 1 and batch size 1024. This is critical for autoregressive generation where we process one token at a time.

### Where It Appears in the Architecture

- `**ln1`** — Applied **before** the self-attention sublayer (Pre-Norm architecture)
- `**ln2`** — Applied **before** the MLP sublayer
- `**ln_f`** — Applied once at the very end, **after** all Transformer blocks, before the final linear projection

### Contribution to the Full SLM

LayerNorm is the architectural glue that lets us stack 6 Transformer blocks without the activations exploding or vanishing. Without it, training a 6-layer transformer with residual connections would be numerically unstable. The Pre-Norm placement (normalizing before each sublayer rather than after) is specifically chosen because it produces smoother loss landscapes and enables more stable training, especially for smaller models.

**Learnable parameters per LayerNorm:** 384 (weight) + 384 (bias) = **768 parameters**

---

## 3. CausalSelfAttention — The Core Reasoning Engine

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)   # Q, K, V projection
        self.c_proj = nn.Linear(n_embd, n_embd)        # output projection
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
```

This is **Multi-Head Causal (Masked) Self-Attention** — the mechanism that allows each token to "look at" all previous tokens in the sequence and decide how much to attend to each one.

### Step-by-Step Through the Forward Pass

**1. Project to Q, K, V:**
A single linear layer (`c_attn`) maps each token's 384-dim representation to a 1152-dim vector (3 × 384), which is then split into three 384-dim vectors: **Query (Q)**, **Key (K)**, and **Value (V)**.

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```

**2. Split into heads:**
Each 384-dim Q/K/V vector is reshaped into 6 heads of 64 dimensions each: `(B, T, 384)` → `(B, 6, T, 64)`. Each head now operates independently in its own 64-dimensional subspace.

```python
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
```

**3. Compute attention scores:**
For each head:

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

where dₖ = 64 is the per-head dimension. The division by √64 = 8 prevents the dot products from growing too large and pushing softmax into saturation.

**4. Apply causal mask:**
Before softmax, all positions where a token would attend to a **future** token are set to -∞. This ensures the model is **autoregressive** — token at position *t* can only see tokens at positions 0, 1, …, *t*. This is what makes it "causal."

**5. Concatenate heads and project:**
The 6 heads' outputs are concatenated back to 384 dimensions and passed through `c_proj`, a final linear projection that mixes information across heads.

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)
y = self.resid_dropout(self.c_proj(y))
```

**6. Dropout:**
Applied both on attention weights (during score computation) and on the output (residual dropout).

### Flash Attention Optimization

The code checks for `F.scaled_dot_product_attention` (PyTorch 2.0+). When available, it uses **Flash Attention** — a fused CUDA kernel that computes attention without materializing the full T × T attention matrix in memory. This reduces memory from O(T²) to O(T) and is significantly faster. The fallback path implements the same math manually.

```python
self.flash = hasattr(F, 'scaled_dot_product_attention')

if self.flash:
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
else:
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)
    y = att @ v
```

### Why We Use It

- **Contextual understanding:** Attention is the only mechanism in the model that allows tokens to exchange information. Without it, each token would be processed in isolation.
- **Multi-head design:** Different heads can specialize in different types of relationships — syntactic dependencies, semantic similarity, positional patterns, etc.
- **Causal masking:** Enforces the left-to-right autoregressive property needed for language generation.

### Contribution to the Full SLM

CausalSelfAttention is the **primary mechanism for learning relationships between tokens**. It answers the question: "Given all the tokens I've seen so far, which ones are most relevant to predicting what comes next?" Every other component in the model either prepares data for attention or processes its output. The multi-head structure gives the model 6 parallel "perspectives" on the data in each block.

**Parameters per attention layer:** `c_attn` = 384 × 1152 + 1152 = **443,520** | `c_proj` = 384 × 384 + 384 = **147,840** | **Total: ~591,360**

---

## 4. MLP — The Nonlinear Feature Transformer

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc   = nn.Linear(n_embd, 4 * n_embd)    # 384 → 1536 (expand)
        self.gelu   = nn.GELU()                          # nonlinear activation
        self.c_proj = nn.Linear(4 * n_embd, n_embd)     # 1536 → 384 (compress)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
```

### What It Does

The MLP is a **position-wise feed-forward network** — it processes each token's representation independently through a two-layer fully connected network with a nonlinear activation in between.

1. **Expand:** `c_fc` projects from 384 → 1536 dimensions (4× expansion). This moves the representation into a higher-dimensional space where complex feature interactions become linearly separable.
2. **Activate:** `GELU` (Gaussian Error Linear Unit) applies a smooth nonlinear activation:
  ```
   GELU(x) = x · Φ(x)
  ```
   where Φ(x) is the CDF of the standard normal distribution. Unlike ReLU, GELU is smooth and non-monotonic near zero, allowing it to softly gate features rather than hard-zeroing them.
3. **Compress:** `c_proj` projects back from 1536 → 384, distilling the nonlinear transformations into the original embedding space.
4. **Dropout:** Regularization on the output.

### Why We Use It

- **Nonlinearity:** Self-attention is fundamentally a weighted average operation — it is almost linear. Without the MLP, stacking attention layers would have limited representational power. The MLP introduces the crucial nonlinearity that allows the model to learn complex functions.
- **4× expansion:** The wider intermediate layer gives the network more "room" to create and combine features. Research has consistently shown that this expansion ratio is effective.
- **GELU over ReLU:** GELU provides smoother gradients and empirically performs better in transformer architectures. It's the standard activation in GPT-2, BERT, and most modern transformers.

### Contribution to the Full SLM

If attention is "gathering information from other tokens," the MLP is "thinking about what that information means." It performs the per-token computation that transforms the attended representations into richer features. In practice, MLPs have been shown to store factual knowledge (key-value memories) and perform feature composition. They account for roughly **2/3 of the parameters** in each Transformer block.

**Parameters per MLP:** `c_fc` = 384 × 1536 + 1536 = **591,360** | `c_proj` = 1536 × 384 + 384 = **590,208** | **Total: ~1,181,568**

---

## 5. Block — The Transformer Block (Attention + MLP)

```python
class Block(nn.Module):
    def __init__(self, config):
        self.ln1  = LayerNorm(n_embd, bias)
        self.attn = CausalSelfAttention(config)
        self.ln2  = LayerNorm(n_embd, bias)
        self.mlp  = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # Residual + Pre-Norm Attention
        x = x + self.mlp(self.ln2(x))     # Residual + Pre-Norm MLP
        return x
```

### What It Does

A Block is the **fundamental repeating unit** of the Transformer. It combines one attention sublayer and one MLP sublayer, each wrapped with Layer Normalization and a **residual (skip) connection**.

### Data Flow Through a Single Block

```
Input x
  ├──────────────────────────┐
  │                          │ (skip connection)
  ▼                          │
LayerNorm (ln1)              │
  ▼                          │
CausalSelfAttention          │
  ▼                          │
  + ◄────────────────────────┘  (add residual)
  │
  ├──────────────────────────┐
  │                          │ (skip connection)
  ▼                          │
LayerNorm (ln2)              │
  ▼                          │
MLP                          │
  ▼                          │
  + ◄────────────────────────┘  (add residual)
  │
  ▼
Output x
```

### Residual Connections: Why `x = x + sublayer(norm(x))`

The residual (skip) connections are arguably as important as attention itself:

- **Gradient highway:** During backpropagation, the gradient can flow directly through the addition operation without being attenuated by the sublayers. This solves the vanishing gradient problem in deep networks.
- **Incremental refinement:** Each block doesn't have to learn the full representation from scratch. Instead, it learns a **delta** — a small correction to add to the existing representation. This makes optimization much easier.
- **Pre-Norm placement:** Normalizing *before* each sublayer (rather than after, as in the original Transformer) produces more stable training dynamics because the sublayer inputs always have controlled magnitude.

### Contribution to the Full SLM

The Block is the building block that gets repeated `n_layer = 6` times. Each successive block refines the token representations with increasingly abstract understanding. Early blocks tend to learn local syntactic patterns, middle blocks capture phrasal and grammatical structure, and later blocks encode higher-level semantic relationships.

**Parameters per Block:** LayerNorm × 2 = 1,536 | Attention = ~591,360 | MLP = ~1,181,568 | **Total: ~1,774,464**

---

## 6. GPT — The Complete Model

```python
class GPT(nn.Module):
    def __init__(self, config):
        self.transformer = nn.ModuleDict(dict(
            wte   = nn.Embedding(vocab_size, n_embd),          # token embeddings
            wpe   = nn.Embedding(block_size, n_embd),          # position embeddings
            drop  = nn.Dropout(dropout),                        # embedding dropout
            h     = nn.ModuleList([Block(config) for _ in range(n_layer)]),
            ln_f  = LayerNorm(n_embd, bias),                    # final layer norm
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight       # weight tying
```

### Component-by-Component Breakdown

#### 6a. Token Embedding (`wte`) — `nn.Embedding(50257, 384)`

Maps each token ID (an integer from 0 to 50,256) to a dense 384-dimensional vector. This is a simple lookup table — row *i* of the weight matrix is the embedding for token *i*. These vectors are learned during training and encode semantic meaning: similar words end up with similar vectors.

**Parameters:** 50,257 × 384 = **19,298,688**

#### 6b. Position Embedding (`wpe`) — `nn.Embedding(128, 384)`

Maps each position index (0 to 127) to a 384-dimensional vector. Since self-attention is permutation-invariant (it treats the input as a set, not a sequence), position embeddings are the **only way** the model knows the order of tokens. Position 0 gets one learned vector, position 1 gets another, etc. These are **added** to the token embeddings, so the model receives `token_meaning + position_signal` as its input.

**Parameters:** 128 × 384 = **49,152**

#### 6c. Embedding Dropout (`drop`) — `nn.Dropout(0.1)`

Applied to the combined (token + position) embeddings. Randomly zeros out 10% of the values during training. This prevents the model from over-relying on specific embedding features and acts as a regularizer.

**Parameters:** 0 (no learnable parameters)

#### 6d. Transformer Blocks (`h`) — `ModuleList of 6 Blocks`

The 6 stacked Transformer blocks, as described in Section 5. Each block sequentially refines the representations.

**Parameters:** 6 × 1,774,464 = **~~10,646,784**

#### 6e. Final Layer Norm (`ln_f`) — `LayerNorm(384)`

A final normalization applied after all 6 blocks and before the output projection. Ensures the final hidden states have stable magnitude before being projected to vocabulary logits.

**Parameters:** 768

#### 6f. Language Model Head (`lm_head`) — `nn.Linear(384, 50257, bias=False)`

Projects the final 384-dimensional hidden state to a 50,257-dimensional vector of **logits** — one score per vocabulary token. The token with the highest logit is the model's prediction for the next token.

**Weight Tying:** The line `self.transformer.wte.weight = self.lm_head.weight` makes the token embedding matrix and the output projection matrix **share the same parameters**. This means:

- The embedding layer and the lm_head are literally the same matrix (50,257 × 384), used in two different directions.
- **Benefit 1:** Reduces parameter count by ~19.3M (the lm_head doesn't need its own matrix).
- **Benefit 2:** Enforces a symmetry — if two tokens have similar embeddings, they'll have similar output logits, which is a strong and useful inductive bias.
- **Benefit 3:** Acts as a regularizer, improving generalization.

**Parameters:** 0 additional (shared with `wte`)

### Weight Initialization

All weights are carefully initialized to ensure stable training from the start:

- **All linear layers and embeddings:** Normal distribution with mean=0, std=0.02
- **Residual projection layers** (`c_proj` in both Attention and MLP): Scaled down by 1/√(2 × n_layer) = 1/√12 ≈ 0.289, so their std becomes 0.02 × 0.289 ≈ 0.0058. This prevents the residual stream from growing too large when contributions from all 6 blocks accumulate.
- **All biases:** Initialized to zero

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

# Special scaling for residual projections
for pn, p in self.named_parameters():
    if pn.endswith('c_proj.weight'):
        nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
```

---

## 7. The Forward Pass — End to End

```python
def forward(self, idx, targets=None):
    # idx shape: (B, T) — batch of token ID sequences
    # targets shape: (B, T) — shifted by 1 for next-token prediction
```

### Step 1: Embedding Lookup

```
idx (B, T) integers
    │
    ▼
tok_emb = wte(idx)          → (B, T, 384)   # lookup token embeddings
pos_emb = wpe(pos)           → (T, 384)      # lookup position embeddings
x = drop(tok_emb + pos_emb) → (B, T, 384)   # combine and apply dropout
```

Each integer token ID is replaced by its learned 384-dim vector. The position signal is added (broadcast over the batch dimension) so the model knows where each token sits in the sequence.

### Step 2: Pass Through 6 Transformer Blocks

```
for block in h:      # iterate through 6 blocks
    x = block(x)     # each block: x = x + attn(ln1(x)), then x = x + mlp(ln2(x))
```

Shape stays `(B, T, 384)` throughout. Each block reads from all previous tokens (via attention) and transforms features (via MLP), gradually building richer contextual representations.

### Step 3: Final Layer Norm

```
x = ln_f(x)  → (B, T, 384)
```

Normalize the output of the last block to ensure stable magnitudes before the final projection.

### Step 4: Project to Vocabulary (Two Modes)

**Training mode** (targets provided):

```python
logits = self.lm_head(x)                          # → (B, T, 50257) — project ALL positions
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    targets.view(-1),
    ignore_index=-1
)
return logits, loss
```

During training, we compute predictions for **every** position simultaneously (teacher forcing). The cross-entropy loss measures how well the model predicts the actual next token at each position.

**Inference mode** (no targets):

```python
logits = self.lm_head(x[:, [-1], :])              # → (B, 1, 50257) — project ONLY last position
return logits, None
```

During generation, we only care about the prediction at the **last** position (what comes next?). This saves computation.

### Complete Forward Pass Diagram

```
Token IDs: [Once, upon, a, time, ...]     (B, T)
                    │
                    ▼
        ┌─── Token Embedding (wte) ───┐
        │      50,257 → 384           │
        │                             │
        │   + Position Embedding (wpe)│
        │      128 → 384              │
        └─────────────────────────────┘
                    │
              Dropout (0.1)
                    │
                    ▼
           ╔═══════════════╗
           ║   Block 1     ║
           ║  LN → Attn    ║─── residual connection
           ║  LN → MLP     ║─── residual connection
           ╚═══════════════╝
                    │
           ╔═══════════════╗
           ║   Block 2     ║
           ║  LN → Attn    ║─── residual connection
           ║  LN → MLP     ║─── residual connection
           ╚═══════════════╝
                    │
                   ...
                    │
           ╔═══════════════╗
           ║   Block 6     ║
           ║  LN → Attn    ║─── residual connection
           ║  LN → MLP     ║─── residual connection
           ╚═══════════════╝
                    │
            Final LayerNorm
                    │
                    ▼
         Language Model Head
         384 → 50,257 logits
        (weight-tied with wte)
                    │
                    ▼
        Softmax → Next Token Probabilities
```

### Text Generation (`generate` method)

After the forward pass produces logits, the `generate` method autoregressively produces new tokens:

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

Key details:

- **Context window cropping:** If the generated sequence exceeds `block_size` (128), only the last 128 tokens are fed as input.
- **Temperature:** Controls randomness. Lower values (e.g., 0.7) make the model more deterministic; higher values (e.g., 1.2) make it more creative.
- **Top-k sampling:** Restricts sampling to the k most probable next tokens, filtering out low-probability noise.

---

## 8. Parameter Count Breakdown


| Component                          | Parameters                |
| ---------------------------------- | ------------------------- |
| Token Embedding (`wte`)            | 19,298,688                |
| Position Embedding (`wpe`)         | 49,152                    |
| 6× Attention (`c_attn` + `c_proj`) | 6 × 591,360 = 3,548,160   |
| 6× MLP (`c_fc` + `c_proj`)         | 6 × 1,181,568 = 7,089,408 |
| 6× LayerNorm (ln1 + ln2)           | 6 × 1,536 = 9,216         |
| Final LayerNorm (`ln_f`)           | 768                       |
| LM Head (`lm_head`)                | 0 (tied with `wte`)       |
| **Total**                          | **29,995,392 (~~30M)**  |


The token embedding dominates the parameter budget at ~64% of the total. The transformer blocks themselves contribute ~36%. Weight tying saves ~19.3M parameters that would otherwise be needed for the output projection.

---

## 9. Summary — Why Each Layer Matters


| Layer                   | Role                                                                        | Without It...                                                      |
| ----------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Token Embedding**     | Converts discrete token IDs into continuous vectors the network can process | The model has no input representation                              |
| **Position Embedding**  | Injects sequential order information                                        | The model treats "the cat sat" and "sat cat the" identically       |
| **LayerNorm**           | Stabilizes activations for reliable training                                | Training diverges or becomes extremely slow                        |
| **CausalSelfAttention** | Lets each token attend to previous tokens to build context                  | Tokens are processed in isolation with no contextual understanding |
| **MLP**                 | Adds nonlinearity and per-token feature transformation                      | The model is limited to approximately linear transformations       |
| **Block (Residual)**    | Wraps attention+MLP with skip connections for deep stacking                 | Gradients vanish; deep networks cannot train                       |
| **LM Head (tied)**      | Maps hidden states back to vocabulary probabilities                         | The model cannot produce text output                               |


### How the Layers Collaborate

The architecture follows a clear information processing pipeline:

1. **Encoding** (Embeddings): Raw text → continuous representations with positional awareness
2. **Contextual Processing** (6× Block):
  - **Attention** gathers relevant information from other tokens in the sequence
  - **MLP** transforms and enriches each token's representation using that gathered context
  - **LayerNorm + Residuals** keep everything numerically stable and trainable
3. **Decoding** (LM Head): Contextual representations → next-token probability distribution

Each block builds on the previous one's output. By block 6, the token representations have been refined through 6 rounds of inter-token communication (attention) and per-token reasoning (MLP), producing rich contextual features that capture syntax, semantics, and narrative structure sufficient for generating coherent short stories.