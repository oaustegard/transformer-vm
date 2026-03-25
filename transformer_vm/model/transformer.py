"""VanillaTransformer with ReGLU FFN and KV cache generation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_vm.attention import HullKVCache

torch.set_default_dtype(torch.float64)


def add_position_encoding(x, pos):
    """Add deterministic position features to the residual stream."""
    x[0] += pos
    x[1] += 1.0 / math.log(2) - 1.0 / math.log(pos + 2)
    x[2] += pos * pos


class VanillaTransformer(nn.Module):
    def __init__(self, vocab, d_model=36, n_heads=18, n_layers=7, d_ffn=36, stop_token_id=0):
        super().__init__()
        self.stop_token_id = stop_token_id
        self.tok = nn.Embedding(vocab, d_model)

        self.attn = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
                for _ in range(n_layers)
            ]
        )
        self.ff_in = nn.ModuleList(
            [nn.Linear(d_model, 2 * d_ffn, bias=False) for _ in range(n_layers)]
        )
        self.ff_out = nn.ModuleList(
            [nn.Linear(d_ffn, d_model, bias=False) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, vocab, bias=False)

    @torch.no_grad()
    def generate_with_cache(self, idx, max_new_tokens=5000, cache_class=HullKVCache):
        """Generate tokens using a KV cache for O(n) or O(log n) inference."""
        n_heads = self.attn[0].num_heads
        n_layers = len(self.attn)
        cache = cache_class(n_layers, n_heads)
        if hasattr(self, "head_tiebreak") and hasattr(cache, "set_tiebreak"):
            for layer_idx in range(n_layers):
                for h in range(n_heads):
                    if self.head_tiebreak[layer_idx][h]:
                        cache.set_tiebreak(layer_idx, h, True)
        idx_list = idx[0].tolist()

        for pos in range(len(idx_list) + max_new_tokens):
            x = self.tok.weight[idx_list[pos]].clone()
            add_position_encoding(x, pos)

            for layer_idx, (attn, ff_in, ff_out) in enumerate(
                zip(self.attn, self.ff_in, self.ff_out, strict=True)
            ):
                q, k, v = (attn.in_proj_weight @ x).chunk(3, dim=-1)
                out = cache.layer_step(layer_idx, k, q, v)
                x = x + attn.out_proj(out)

                gate, val = ff_in(x).chunk(2, dim=-1)
                x = x + ff_out(F.relu(gate) * val)

            if pos + 1 == len(idx_list):
                next_id = self.head(x).argmax().item()
                idx_list.append(next_id)
                if next_id == self.stop_token_id:
                    break

        return torch.tensor([idx_list], dtype=torch.long, device=idx.device)
