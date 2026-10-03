"""Static sparse addressing ablation; only key/value tying differs from W0."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.Titan.common.memory import SimilarityWeightedLocalMemoryMatcher


KEY_VALUE_BACKBONE = "titantpp_key_value_static_memory"
KEY_VALUE_ROLE = "t0_key_value_static_retrieval"
KEY_VALUE_CONTRACT = "hard_lmm_key_value_v1"
KEY_VALUE_MEMORY_MODE = "static_key_value_lmm"


class KeyValueLocalMemoryMatcher(SimilarityWeightedLocalMemoryMatcher):
    """Keep hard top-k and tau=1 weighting, but learn separate addressing keys.

    ``mem`` remains the original value bank. Cloning it consumes no additional
    RNG draws, preserving all shared initial weights and initial W0 outputs.
    Top-k membership is discrete; only selected scores have an autograd path.
    """

    def __init__(self, d_model: int, mem_size: int = 64, topk: int = 4):
        if d_model < 1 or mem_size < 1 or not 1 <= topk <= mem_size:
            raise ValueError("Separate-key memory requires positive dimensions and 1 <= topk <= mem_size")
        super().__init__(d_model=d_model, mem_size=mem_size, topk=topk)
        self.memory_keys = nn.Parameter(self.mem.detach().clone())

    def retrieve(
        self,
        encoded: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if memory is not None:
            raise ValueError("Separate-key retrieval does not accept a value-only external memory")
        if encoded.ndim != 3 or encoded.size(-1) != self.d_model:
            raise ValueError("encoded must have shape [batch, length, d_model]")
        batch, length, dim = encoded.shape
        keys = self.memory_keys.expand(batch, -1, -1)
        values = self.mem.expand(batch, -1, -1)
        scores = F.normalize(encoded, p=2, dim=-1) @ F.normalize(
            keys, p=2, dim=-1
        ).transpose(-2, -1)
        selected_scores, indices = torch.topk(scores, self.topk, dim=-1)
        selected = torch.gather(
            values.unsqueeze(1).expand(-1, length, -1, -1),
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, dim),
        )
        weights = (selected_scores / self.temperature).softmax(dim=-1)
        residual = (selected * weights.unsqueeze(-1)).sum(dim=2)
        return residual, {
            "prototype_indices": indices,
            "topk_similarity": selected_scores,
            "retrieval_weights": weights,
        }
