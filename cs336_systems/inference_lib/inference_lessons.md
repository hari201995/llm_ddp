# Allocation

### torch.zeros(num_layers, max_batch, num_heads, max_seq_len, d_k, device=device,dtype=torch.bfloat16) just this would allocate in hbm?

Yes, if device resolves to "cuda" — that single call physically reserves space in the GPU's HBM, immediately, not lazily.

### What happens under the hood:

  torch.zeros(..., device="cuda", dtype=torch.bfloat16)\
  PyTorch's CUDA caching allocator
           │
           ├─ already holds a free block this size from a previous tensor that was freed?
           │     → reuse it instantly, no driver call
           │
           └─ no matching free block?
                 → calls cudaMalloc, which maps a chunk of the A100's HBM2e
                   physical memory into your process's address space
           │
           ▼
  tensor's data pointer now points into HBM, zero-filled

  A few things worth being precise about:

  - It's eager, not lazy. The full byte count (num_layers × max_batch × num_heads × max_seq_len × d_k × 2 × 2 bytes) is reserved at that line — not on first write.
  That's exactly the "preallocate once" behavior you want.
  - It's device-dependent, not universal. Same line with device="mps" lands in Apple's unified memory (not HBM — Apple Silicon uses LPDDR, a different tech). With
  device="cpu", it's plain host RAM. The HBM destination only happens because your A100's device="cuda".
  - The caching allocator, not the OS, owns reuse. Once you del or overwrite this tensor, PyTorch doesn't return that HBM block to the driver — it keeps it in its own
  pool for the next torch.zeros/torch.empty call of similar size. That's why preallocating once up front (rather than torch.cat-growing every decode step) avoids
  repeated cudaMalloc calls — the first allocation is the expensive one; everything after just writes into the already-reserved block.