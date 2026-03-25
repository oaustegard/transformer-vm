"""O(log n) hull-based KV cache wrapper using pybind11 convex hull extension."""

import os

import torch

_hull_ext = None


def _load_ext():
    global _hull_ext
    if _hull_ext is None:
        from torch.utils.cpp_extension import load

        root = os.path.dirname(os.path.abspath(__file__))
        _hull_ext = load(
            name="hull_ext",
            sources=[os.path.join(root, "hull_ext.cpp")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_include_paths=[root],
            verbose=False,
        )
    return _hull_ext


class HullKVCache:
    """O(log n) hard-attention KV cache using 2D convex hulls."""

    def __init__(self, n_layers, n_heads):
        ext = _load_ext()
        self._cache = ext.HullKVCache(n_layers, n_heads)
        self._seq = -1

    def clear(self):
        """Reset all hull state and rewind the sequence counter."""
        self._cache.clear()
        self._seq = -1

    def set_tiebreak(self, layer, head, latest):
        """Set tiebreak mode for a head: True for latest, False for average."""
        self._cache.set_tiebreak(layer, head, 1 if latest else 0)

    def layer_step(self, layer, keys, queries, values):
        """Insert KV pair and query all heads for one layer, return attention output."""
        self._seq += 1
        out_np = self._cache.layer_step(
            layer,
            keys.reshape(-1, 2).numpy(),
            queries.reshape(-1, 2).numpy(),
            values.reshape(-1, 2).numpy(),
            self._seq,
        )
        return torch.from_numpy(out_np).flatten()
