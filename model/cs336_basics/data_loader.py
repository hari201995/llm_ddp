import numpy as np
import torch

"""
np.memmap maps a file on disk directly into the process's virtual address space. 
Instead of reading the whole file into RAM upfront, the OS loads only the pages you
actually touch — on demand, as you index into the array. The rest stays on disk.

This matters for large datasets because your RAM usage stays bounded to only what
you're actively reading, not the full file size. For a 50 MB shard, you might only bring
a few KB into RAM per batch.

The tradeoff is that the first access to a new page incurs a page fault — a small 
latency hit as the OS fetches it from disk. Subsequent accesses to the same page are 
fast since it's cached. Closing the memmap releases the mapping and frees those cached 
pages.

In the training loop, you open a shard, load one batch from a random offset, then close
— so you're paying the page fault cost every step, but keeping memory usage minimal 
across 100 shards.

VIRTUAL ADDRESS SPACE          PHYSICAL RAM              DISK
┌─────────────────────┐        ┌──────────────┐         ┌──────────────┐
│                     │        │              │         │              │
│   Python Process    │        │  Page Cache  │         │  shard_0.bin │
│                     │        │              │         │              │
│  tokenized_data ────┼──map───┼──────────────┼─────────┼► [0 ....     │
│  (memmap object)    │        │              │         │   .....      │
│                     │        │  ┌─────────┐ │         │   .....      │
│  tokenized_data[42] │        │  │ page 42 │◄├─fetch───┼─ .....       │
│  (first access)     │        │  └─────────┘ │  fault  │   .....      │
│                     │        │              │         │   .......]   │
│  tokenized_data[42] │        │  ┌─────────┐ │         │              │
│  (second access) ───┼────────┼─►│ page 42 │ │  cache  │              │
│                     │        │  └─────────┘ │   hit   │              │
└─────────────────────┘        └──────────────┘         └──────────────┘

- First access → page fault → OS fetches from disk into RAM
- Second access to same page → served directly from RAM (cache hit)
- Close memmap → page evicted, RAM freed
"""


def data_loader(x, batch_size, context_length, device_type=None):
    """
    data_loader takes a numpy array x (integer array with token IDs), a
    batch_size, a context_length and a PyTorch device string (e.g., 'cpu' or 'cuda:0'), and returns
    a pair of tensors: the sampled input sequences and the corresponding next-token targets. Both tensors
    should have shape (batch_size, context_length) containing token IDs, and both should be
    placed on the requested device

    Args :
        x - input array of tokens ( np.array)
        batch_size -size of batch
        context_length - length of context to split the corpus
        device_type - location in which the data has to be loaded and split

        Returns :
            y - sampled input sequences ( tensor: batch_size x context_length)
            t - targets ( tensor:  batch_sizr x context_length)
    """
    # device
    if device_type is None or device_type == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, "mps") and torch.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_type)

    # Convert x to numpy, put it in device and ensure they are
    # in contiguous memory
    max_start = len(x) - context_length - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    inp = np.stack([x[s : s + context_length] for s in starts]).astype(np.int64)
    tgt = np.stack([x[s + 1 : s + context_length + 1] for s in starts]).astype(np.int64)

    y = torch.from_numpy(inp).to(device)
    t = torch.from_numpy(tgt).to(device)
    return y, t


if __name__ == "__main__":
    data_loader(x=np.array(range(1000)), batch_size=100, context_length=256)
