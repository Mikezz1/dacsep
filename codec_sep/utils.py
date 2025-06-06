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


def librimix_collate(batch, sr: int = 24_000):
    """
    * mix  – 2 s / 3 s / 4 s  (picked once per batch, uniform)
    * ref  – 2 … 8 s          (picked once per batch, uniform)
    Everything is trimmed or 0-padded so tensors stack cleanly.
    """
    mix_sec = random.choice((2, 3, 4))
    ref_sec = random.choice((2, 3, 4, 5, 6, 7, 8))
    mix_len = mix_sec * sr
    ref_len = ref_sec * sr

    mix_b, tgt_b, ref_b, orig_lens, spk_ids = [], [], [], [], []

    for mix, tgt, ref, orig_len, spk in batch:

        if mix.size(-1) < mix_len:
            pad = mix_len - mix.size(-1)
            mix = F.pad(mix, (0, pad))
            tgt = F.pad(tgt, (0, pad))
        mix_b.append(mix[..., :mix_len])
        tgt_b.append(tgt[..., :mix_len])

        if ref.size(-1) < ref_len:
            pad = ref_len - ref.size(-1)
            ref = F.pad(ref, (0, pad))
        ref_b.append(ref[..., :ref_len])

        orig_lens.append(min(orig_len, mix_len))
        spk_ids.append(spk)

    return (
        torch.stack(mix_b),
        torch.stack(tgt_b),
        torch.stack(ref_b),
        torch.tensor(orig_lens),
        torch.tensor(spk_ids),
    )
