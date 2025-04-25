import yaml  # pip install pyyaml
from types import SimpleNamespace
from pathlib import Path
import math


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(i) for i in obj]
    return obj


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def si_sdr(pred, target, eps=1e-8):
    """Scale‑invariant SDR (higher is better). Returns a *loss* (negated)."""
    target_energy = (target**2).sum(dim=-1, keepdim=True)
    scale = (pred * target).sum(dim=-1, keepdim=True) / (target_energy + eps)
    proj = scale * target
    e_noise = pred - proj
    ratio = (proj**2).sum(dim=-1) / ((e_noise**2).sum(dim=-1) + eps)
    return -10 * torch.log10(ratio + eps)  # negate → minimise



def lr_lambda(current_step, warmup_steps, total_training_steps):
    if current_step < warmup_steps:
        # Linear warmup from 0 -> 1
        return float(current_step) / float(max(1, warmup_steps))
    else:
        # Cosine decay from 1 -> 0 after warmup
        remaining_steps = float(total_training_steps - current_step)
        decay_steps = float(max(1, total_training_steps - warmup_steps))
        # Cosine decay formula: 0.5 * (1 + cos(pi * x))
        cosine_decay = 0.5 * (
            1 + math.cos(math.pi + math.pi * (remaining_steps / decay_steps))
        )
        return max(cosine_decay, 0)