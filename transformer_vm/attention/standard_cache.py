"""O(n) reference softmax KV cache for standard attention."""

import torch
import torch.nn.functional as F


class StandardKVCache:
    """Standard KV cache with softmax attention."""

    def __init__(self, n_layers, n_heads):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self._keys = [[] for _ in range(n_layers)]
        self._vals = [[] for _ in range(n_layers)]

    def clear(self):
        """Reset all cached keys and values."""
        self._keys = [[] for _ in range(self.n_layers)]
        self._vals = [[] for _ in range(self.n_layers)]

    def layer_step(self, layer, keys, queries, values):
        """Append KV pair and compute softmax attention output for one layer."""
        self._keys[layer].append(keys.clone())
        self._vals[layer].append(values.clone())

        K = torch.stack(self._keys[layer]).reshape(-1, self.n_heads, keys.shape[0] // self.n_heads)
        V = torch.stack(self._vals[layer]).reshape(-1, self.n_heads, keys.shape[0] // self.n_heads)
        Q = queries.reshape(self.n_heads, -1)

        scores = torch.einsum("thi,hi->th", K, Q)
        weights = F.softmax(scores, dim=0)
        out = torch.einsum("th,thi->hi", weights, V)
        return out.flatten()
