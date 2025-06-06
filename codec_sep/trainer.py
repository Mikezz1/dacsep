import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Config, GPT2LMHeadModel, AutoFeatureExtractor
from transformers import MimiModel, AutoFeatureExtractor

import torchaudio
import os
from tqdm import tqdm
import numpy as np

from codec_sep.data import LibrimixDataset
from codec_sep.model import TransformerEncoderWithCodebookCondition
from torch.optim.lr_scheduler import LambdaLR
import math
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio


def set_train_mode(model):
    model.train()
    model.quantizer.eval()
    return model


class MultiScaleMelLoss(nn.Module):
    """
    Multi-scale log-Mel L1 loss (five scales).
    """

    def __init__(
        self,
        sample_rate: int = 24_000,
        device: torch.device | str = "cuda",
        # (n_fft, hop_length) for each scale, longest → shortest
        scales: list[tuple[int, int]] | None = None,
    ):
        super().__init__()

        if scales is None:
            # ≈20 ms down to ≈1.25 ms hops at 24 kHz
            scales = [
                (2048, 512, 20),  # 20.0 ms hop
                (1024, 256, 20),  # 10.0 ms
                (512, 128, 20),  # 5.0 ms
                (256, 64, 20),  # 2.5 ms
                (128, 32, 20),  # 1.25 ms
                (2048, 512, 64),  # 20.0 ms hop
                (1024, 256, 64),  # 10.0 ms
                (512, 128, 64),  # 5.0 ms
                (256, 64, 64),  # 2.5 ms
                # (128, 32, 64),    # 1.25 ms
                (2048, 512, 128),  # 20.0 ms hop
                (1024, 256, 128),  # 10.0 ms
                (512, 128, 128),  # 5.0 ms
                # (256, 64, 128),    # 2.5 ms
                # (128, 32, 128),    # 1.25 ms
            ]

        self.transforms = nn.ModuleList(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop,
                n_mels=n_mels,
                power=1.0,  # magnitude -> power=1 keeps same scale as input
            ).to(device)
            for n_fft, hop, n_mels in scales
        )

        self.device = torch.device(device)

    @staticmethod
    def _log_mel(wave: torch.Tensor, mel_fn: torchaudio.transforms.MelSpectrogram):
        # wave: (B, T) in -1…1
        return torch.log(mel_fn(wave.unsqueeze(1)).clamp_min(1e-5))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred / target: (B, T) in -1…1, same length & sample-rate.
        Returns: mean L1 over 5 scales.
        """
        loss = 0.0
        for mel_fn in self.transforms:
            loss += F.l1_loss(
                self._log_mel(pred, mel_fn), self._log_mel(target, mel_fn)
            )
        return loss / len(self.transforms)


def eval_epoch(
    loader,
    model,
    device,
    criterion,
    cfg,
):
    model.eval()
    total_loss = 0.0
    si_sdr = 0.0
    si_sdr_pseudo = 0.0
    si_sdr_pseudo_lowpass = 0.0
    dns_mos = 0.0

    for batch in tqdm(loader):
        noisy_audio = batch[0].to(device).unsqueeze(1)
        clean_audio = batch[1].to(device).unsqueeze(1)
        ref_audio = batch[2].to(device).unsqueeze(1)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if cfg.use_descript:
                    out, speaker_label_pred = model(
                        mix=noisy_audio,
                        speaker=ref_audio,
                    )
                    out_waveform = model.descript_decode(out)
                    if out_waveform.size(1) != noisy_audio.size(1):
                        out_waveform = F.pad(
                            out_waveform,
                            (0, noisy_audio.size(2) - out_waveform.size(1)),
                        )
                else:
                    if cfg.after_quant:
                        out = model(
                            mix=noisy_audio,
                            speaker=ref_audio,
                        )
                        out_waveform = model.decode_latent(out)
                    else:
                        pass

                if cfg.use_descript:

                    x = model.descript_encode(clean_audio)
                    reconstructed_target = model.descript_decode(x)
                    if reconstructed_target.size(1) != clean_audio.size(1):
                        reconstructed_target = F.pad(
                            reconstructed_target,
                            (0, clean_audio.size(2) - reconstructed_target.size(1)),
                        )
                else:

                    codes = model.quantizer.encode(
                        clean_audio, num_quantizers=8
                    ).audio_codes
                    reconstructed_target = model.quantizer.decode(
                        codes
                    ).audio_values.squeeze()

                si_sdr += (
                    scale_invariant_signal_distortion_ratio(
                        out_waveform, clean_audio.squeeze()
                    )
                    .mean()
                    .detach()
                    .cpu()
                )
                si_sdr_pseudo += (
                    scale_invariant_signal_distortion_ratio(
                        out_waveform, reconstructed_target
                    )
                    .mean()
                    .detach()
                    .cpu()
                )

    return (
        (si_sdr / len(loader)).item(),
        (si_sdr_pseudo / len(loader)).item(),
    )


USE_SPEAKER_EMBEDDER = False
N_CODEBOOKS = 8
O = False


def train_epoch(
    loader,
    model,
    optimizer,
    scheduler,
    criterion,
    writer,
    device,
    step,
    cfg,
    mel_loss,
):
    total_loss = 0
    global O

    for batch in tqdm(loader):

        noisy_audio = batch[0].unsqueeze(1)
        clean_audio = batch[1].unsqueeze(1)
        ref_audio = batch[2].unsqueeze(1)
        speaker_label = batch[-2].to(device)

        if cfg.dynamic_mixing:
            # Energy of both signals
            s1 = noisy_audio
            s2 = clean_audio

            e1 = (s1**2).sum(dim=2, keepdim=True)
            e2 = (s2**2).sum(dim=2, keepdim=True)

            # Random SNR between -10 and 10 dB
            snr = 20 * torch.rand(s1.shape[0], 1, 1, device=s1.device) - 10

            # Compute scaling factor for s2 to match SNR
            scale = torch.sqrt(e1 / e2) * (10 ** (snr / 20))
            s2_scaled = s2 * scale

            # Mix the signals
            mixed = s1 + s2_scaled

            # Normalize by the maximum amplitude across all signals
            max_amp = (
                torch.stack(
                    [
                        s1.abs().max(dim=2, keepdim=True)[0],
                        s2_scaled.abs().max(dim=2, keepdim=True)[0],
                        mixed.abs().max(dim=2, keepdim=True)[0],
                    ]
                )
                .max(dim=0)[0]
                .clamp_min(1e-8)
            )

            # Apply normalization
            mixed = mixed / max_amp

            noisy_audio = mixed
            clean_audio = s2

        if cfg.variable_len_train:  #  6, 8 # , 5, 6, 7
            max_len_sample = np.random.choice([2, 3, 4]) * cfg.sample_rate  #  6, 7, 8
            noisy_audio = noisy_audio[:, :max_len_sample].to(device)
            clean_audio = clean_audio[:, :max_len_sample].to(device)

        if cfg.variable_len_ref:
            max_len_ref = np.random.choice([2, 3, 4, 5, 6, 7, 8]) * cfg.sample_rate
            ref_audio = ref_audio[:, :max_len_ref].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

            if not cfg.use_descript:
                if cfg.after_quant:
                    _out = model(
                        mix=noisy_audio,
                        speaker=ref_audio,
                    )
                    out_waveform = model.decode_latent(_out)
                else:
                    _out = model(
                        mix=noisy_audio,
                        speaker=ref_audio,
                    )
                    out_waveform = model.quantize_and_decode(_out)
            else:

                _out, speaker_label_pred = model(
                    mix=noisy_audio,
                    speaker=ref_audio,
                    speaker_label=speaker_label,
                )
                out_waveform = model.descript_decode(_out)
                if out_waveform.size(1) != noisy_audio.size(1):
                    out_waveform = F.pad(
                        out_waveform, (0, noisy_audio.size(2) - out_waveform.size(1))
                    )

            if cfg.use_descript:
                with torch.no_grad():

                    z = model.descript_encode(clean_audio)
                    _z, _, _, _, _ = model.quantizer.quantizer(z, None)
                    with torch.no_grad():
                        reconstructed_target = model.descript_decode(z).squeeze()
                    if reconstructed_target.size(1) != clean_audio.size(2):
                        reconstructed_target = F.pad(
                            reconstructed_target,
                            (0, clean_audio.size(2) - reconstructed_target.size(1)),
                        )

            si_sdr_loss = -scale_invariant_signal_distortion_ratio(
                out_waveform, reconstructed_target.squeeze()
            )

            ce_speaker_loss = 0

            _mel_loss = 0
            if cfg.add_mel_loss:
                if cfg.use_descript:
                    _mel_loss = mel_loss(
                        out_waveform, reconstructed_target.squeeze()
                    ).mean()
                else:
                    _mel_loss = F.l1_loss(
                        _out.transpose(1, 2), model.get_latent(clean_audio)
                    ).mean()

            l1_latent_loss = 0
            if cfg.add_l1_latent_loss:
                if cfg.use_descript:
                    l1_latent_loss = F.l1_loss(_out, _z).mean()

            # print(si_sdr_loss[0])

            writer.add_scalar(f"Loss/train", si_sdr_loss.mean().detach().cpu(), step)
            total_loss += si_sdr_loss.mean().detach().cpu()

            if cfg.add_mel_loss:
                writer.add_scalar(f"latent_loss/train", _mel_loss.detach().cpu(), step)

            if cfg.add_l1_latent_loss:
                writer.add_scalar(
                    f"l1_latent_loss/train", l1_latent_loss.detach().cpu(), step
                )

            if cfg.weight_speaker > 0:
                writer.add_scalar(
                    f"speaker_loss/train", ce_speaker_loss.detach().cpu(), step
                )

            w1, w2, w3, w4 = 1.0, 1.0, 1.0, 1.0
            if cfg.add_mel_loss:
                w1, w2 = cfg.weight_sdr, cfg.weight_mel
            if cfg.add_l1_latent_loss:
                w1, w2, w3 = cfg.weight_sdr, cfg.weight_mel, cfg.weight_latent

            w4 = cfg.weight_speaker

            loss = (
                si_sdr_loss.mean() * w1
                + _mel_loss * w2
                + l1_latent_loss * w3
                + ce_speaker_loss * w4
            )
            loss.backward()

            writer.add_scalar(f"Total_loss/train", loss.detach().cpu(), step)
            writer.add_scalar(f"LR/train", scheduler.get_lr()[-1], step)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        step += 1
    return total_loss, step


def train(
    loader,
    val_loader,
    model,
    optimizer,
    scheduler,
    criterion,
    c_root,
    log_dir,
    num_epochs,
    device,
    cfg,
    eval_every=10,
    *args,
    **kwargs,
):
    print("Start training with codebook conditioning (teacher forcing) ...")

    mel_loss = MultiScaleMelLoss(device="cuda")

    writer = SummaryWriter(log_dir=log_dir)
    scheduler.step()
    step = 0
    if cfg.overfit:
        loader = [next(iter(loader))]
    for epoch in range(0, num_epochs):
        model = set_train_mode(model)

        total_loss = 0.0

        total_loss, step = train_epoch(
            loader,
            model,
            optimizer,
            scheduler,
            criterion,
            writer,
            device,
            step,
            cfg=cfg,
            mel_loss=mel_loss,
        )

        writer.add_scalar(
            f"Loss/train_epoch",
            total_loss / len(loader) if not cfg.overfit else total_loss,
            epoch,
        )
        print(
            f"Epoch {epoch}: total_loss={total_loss / len(loader) :.3f}, lr: {scheduler.get_lr()[-1]}"
        )  # len(loader) #/ len(loader)
        if (epoch % eval_every == 0) and not cfg.overfit:
            val_sdr, val_sdr_pseudo = eval_epoch(  # dnsmos
                val_loader,
                model,
                device=device,
                criterion=criterion,
                cfg=cfg,
            )
            writer.add_scalar(f"SDR/val_epoch", val_sdr, epoch)
            writer.add_scalar(f"SDR_PSEUDO/val_epoch", val_sdr_pseudo, epoch)
            # write.add_scalar(f"dnsmos/val_epoch", dnsmos, epoch)
        if (epoch % 10 == 0) and not cfg.overfit:
            torch.save(
                model,
                os.path.join(c_root, f"model.pt"),
            )
            torch.save(optimizer, os.path.join(c_root, f"optimizer.pt"))
            torch.save(scheduler, os.path.join(c_root, f"scheduler.pt"))
    writer.flush()
    print("Training complete!")
