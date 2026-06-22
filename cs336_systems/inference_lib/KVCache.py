import torch
import torch.nn


class KVCache:
    """
    Contains read,write lib methods for kv cache for a given config
    """

    def __init__(self, num_heads, d_model, device, max_seq, max_batches):
        # initializations
        self.max_batches = max_batches
        self.seq_len = max_seq
        self.D = d_model
        self.num_heads = num_heads
        # Assign memory for K and V
        self.k_memory = torch.zeros(
            (self.max_batches, self.num_heads, self.seq_len, self.D // self.num_heads),
            dtype=torch.bfloat16,
            device=device,
        )
        self.v_memory = torch.zeros(
            (self.max_batches, self.num_heads, self.seq_len, self.D // self.num_heads),
            dtype=torch.bfloat16,
            device=device,
        )
        # sequence tracker
        self.tracker_t = 0
        self.tracker_b = 0

        # mode tracker
        self.prefill = True

    def kv_write(self, K, V):
        self.k_memory[
            self.tracker_b : self.tracker_b + K.size(0),
            :,
            self.tracker_t : self.tracker_t + K.size(2),
            :,
        ] = K
        self.v_memory[
            self.tracker_b : self.tracker_b + V.size(0),
            :,
            self.tracker_t : self.tracker_t + V.size(2),
            :,
        ] = V
        # both K and V are BKSH.so just one tracking variable is enough.
        self.tracker_t += K.size(2)
        # For multi batch enable this
        # self.tracker_b += K.size(0)

    def kv_read(self):
        return (
            self.k_memory[: self.tracker_b + 1, :, : self.tracker_t, :],
            self.v_memory[: self.tracker_b + 1, :, : self.tracker_t, :],
        )

    def reset(self):
        """Start a fresh sequence: rewind the cursor and re-enter prefill mode."""
        self.tracker_t = 0
        self.prefill = True
