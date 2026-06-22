# SMALL MODEL CONFIG

| Component | Shape | Params |
|---|---|---|
| Embedding (E) | 32000 × 768 | 24,576,000 |
| Per layer: Wq, Wk, Wv, Wo | 4 × (768×768) | 2,359,296 |
| Per layer: 2× RMSNorm | 2 × 768 | 1,536 |
| Per layer: SwiGLU (fc1, fc2, glu_w) | 3 × (768×2048) | 4,718,592 |
| Per layer total | | 7,079,424 |
| × 12 layers | | 84,953,088 |
| Final RMSNorm | 768 | 768 |
| Output head (L) | 32000 × 768 | 24,576,000 |
| **Total** | | **≈134.1M params** |

# KV Cache calculation 

x: (768,) \
K = Wk @ x → (768,768) × (768,1) = (768,1)   ← 768 values.\
V = Wv @ x → (768,768) × (768,1) = (768,1)   ← 768 values.

Wk/Wv are the weights (768×768) — they get reused for every token. The cache only stores the output K and V vectors, each 768-dim. You got this exactly right:
result is [768,1], not [768,768].

## Bytes per token, per layer
K: 768 elements × 2 bytes (bf16) = 1536 bytes \
V: 768 elements × 2 bytes (bf16) = 1536 bytes \
K + V = 3072 bytes = 3 KB

Across all 12 layers, per token:
 3 KB × 12 layers = 36 KB \
 For your full context_length=256:
>36 KB/token × 256 tokens = 9,216 KB ≈ 9 MB   (batch=1)

