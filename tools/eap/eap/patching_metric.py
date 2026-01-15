"""
Metrics for activation patching / attribution patching...
"""

import torch
from typing import Optional
from jaxtyping import Float
from torch import Tensor
from typing import Optional, Tuple, List


def avg_logit_diff(
    logits: Float[Tensor, 'batch seq d_vocab'],
    end_positions: List[int] = None,
    clean_token_ids: List[int] = None,
    corrupted_token_ids: List[int] = None,
    per_prompt: bool = False
):
    '''
        Return average logit difference between correct and incorrect answers
    '''
    clean_logits = logits[:, end_positions, clean_token_ids]
    corrupted_logits = logits[:, end_positions, corrupted_token_ids]
    logit_diff = clean_logits - corrupted_logits
    return logit_diff if per_prompt else logit_diff.mean()
