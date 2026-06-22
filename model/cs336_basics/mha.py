import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from einops import rearrange, einsum
import math

from .scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.RoPE import RoPE


class MHA(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        theta=None,
        max_seq_len=None,
        device=None,
        layer=None,
        KV=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # belongs to this layer
        self.layer = layer
        # KV Cache object
        self.KV = KV
        # learable parameter
        self.dk = int(self.d_model / num_heads)
        self.dv = int(self.d_model / num_heads)

        # Rope
        if theta != None:
            self.R = RoPE(theta, self.dk, max_seq_len, device=device)  # type: ignore

        # define weights W_q W_k W_v and W_o
        sigma = math.sqrt(2 / (self.dk + self.d_model))
        self.Wq = nn.Parameter(
            torch.empty(size=(num_heads * self.dk, self.d_model), device=device)
        )
        init.trunc_normal_(self.Wq, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)
        self.Wk = nn.Parameter(
            torch.empty(size=(num_heads * self.dk, self.d_model), device=device)
        )
        init.trunc_normal_(self.Wk, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

        sigma = math.sqrt(2 / (self.dv + self.d_model))

        self.Wv = nn.Parameter(
            torch.empty(size=(num_heads * self.dv, self.d_model), device=device)
        )
        init.trunc_normal_(self.Wv, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)
        self.Wo = nn.Parameter(
            torch.empty(size=(self.d_model, num_heads * self.dv), device=device)
        )
        init.trunc_normal_(self.Wo, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def mha_self_attention(self, x, theta=None, token_position=None, device=None):

        decoding = self.KV is not None and self.KV.prefill != True

        if decoding:
            # Prefill already done: only the newest token needs Q/K/V computed.
            # Slice with -1: (keeps the time axis, size 1) — x[:, -1, :] would drop it.
            ip = x[:, -1:, :]
            prev_k, prev_v = self.KV.kv_read()  # type: ignore # only the valid cached prefix
            start = self.KV.tracker_t  # type: ignore
            positions = torch.arange(start, start + ip.shape[-2], device=x.device)
        else:
            # prefill or train: compute over the whole chunk
            ip = x
            positions = token_position

        Q = einsum(self.Wq, ip, "i j,... t j -> ... t i")
        Q = rearrange(Q, "... b t (h d) -> ... b h t d", h=self.num_heads)
        if theta != None:
            Q = self.R.forward(Q, positions)  # type: ignore

        temp_V = einsum(self.Wv, ip, "i j, ... t j ->... t i")
        temp_V = rearrange(temp_V, "... b t (h d) -> ... b h t d", h=self.num_heads)

        temp_K = einsum(self.Wk, ip, "i j,... t j ->... t i")
        temp_K = rearrange(temp_K, "... b t (h d) -> ... b h t d", h=self.num_heads)
        if theta != None:
            temp_K = self.R.forward(temp_K, positions)  # type: ignore

        if self.KV is not None:
            # Convert temp K,V,Q to bf16
            Q = Q.to(dtype=torch.bfloat16)
            temp_K = temp_K.to(dtype=torch.bfloat16)
            temp_V = temp_V.to(dtype=torch.bfloat16)

            if decoding:
                K = torch.cat((prev_k, temp_K), dim=2)
                V = torch.cat((prev_v, temp_V), dim=2)
            else:
                # first (prefill) call: nothing cached yet, nothing to concat
                K = temp_K
                V = temp_V

            # cache only the newly computed chunk, never the concatenated history
            self.KV.kv_write(temp_K, temp_V)
        else:
            # training / no-cache case
            K = temp_K
            V = temp_V

        if decoding:
            # the single new query is the newest token — it sees the entire
            # cache plus itself by construction, no masking needed
            M = torch.ones(
                x.shape[0],
                self.num_heads,
                Q.shape[-2],
                K.shape[-2],
                dtype=torch.bool,
                device=x.device,
            )
        else:
            M = torch.tril(
                torch.ones(
                    x.shape[0],
                    self.num_heads,
                    Q.shape[-2],
                    K.shape[-2],
                    dtype=torch.bool,
                    device=x.device,
                ),
            )

        output_mha = scaled_dot_product_attention(Q, K, V, mask=M)
        output_mha = rearrange(output_mha, "... b h t d -> ... b t (h d)")

        if self.KV is not None:
            # Convert output to fp32 so that Wo*output happens
            output_mha = output_mha.to(dtype=torch.float32)

        y = einsum(self.Wo, output_mha, "i j,... n j -> ... n i")

        return y
