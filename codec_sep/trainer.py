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

from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio




def eval_epoch(
    loader,
    model,
    device,
    criterion,
):
    model.eval()
    total_loss = 0.0
    si_sdr = 0.0
    si_sdr_pseudo = 0.0
    si_sdr_pseudo_lowpass = 0.0

    for batch in tqdm(loader):
        noisy_audio = batch[0].to(device).unsqueeze(1)
        clean_audio = batch[1].to(device).unsqueeze(1)
        ref_audio = batch[2].to(device).unsqueeze(1)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(
                    mix=noisy_audio,
                    speaker=clean_audio,
                )
                codes = model.quantizer.quantizer.encode(out, num_quantizers=8).transpose(0,1)
                out_waveform = model.quantizer.decode(codes).audio_values.squeeze()
                codes = model.quantizer.encode(clean_audio, num_quantizers=8).audio_codes
                reconstructed_target = model.quantizer.decode(codes).audio_values.squeeze()

                si_sdr += (
                    scale_invariant_signal_distortion_ratio(out_waveform, clean_audio.squeeze())
                    .mean()
                    .detach()
                    .cpu()
                )
                si_sdr_pseudo += (
                    scale_invariant_signal_distortion_ratio(out_waveform, reconstructed_target)
                    .mean()
                    .detach()
                    .cpu()
                )

    return (
        (si_sdr / len(loader)).item(),
        (si_sdr_pseudo / len(loader)).item(),
    )  # , (si_sdr_pseudo_lowpass  / len(loader)).item()


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
):
    total_loss = 0
    global O

    for batch in tqdm(loader):
        optimizer.zero_grad()
        noisy_audio = batch[0].to(device).unsqueeze(1)
        clean_audio = batch[1].to(device).unsqueeze(1)
        ref_audio = batch[2].to(device).unsqueeze(1)

        if cfg.variable_len_train:
            max_len_sample = np.random.randint(1, 7) * 24000
            noisy_audio = noisy_audio[:, :max_len_sample]
            clean_audio = clean_audio[:, :max_len_sample]

        if cfg.variable_len_ref:
            max_len_ref = np.random.randint(1, 7) * 24000
            ref_audio = ref_audio[:, :max_len_ref]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                mix=noisy_audio,
                speaker=ref_audio,
            )
            codes = model.quantizer.quantizer.encode(out, num_quantizers=8).transpose(0,1)
            out_waveform = model.quantizer.decode(codes).audio_values.squeeze()
            codes = model.quantizer.encode(clean_audio, num_quantizers=8).audio_codes
            reconstructed_target = model.quantizer.decode(codes).audio_values.squeeze()

            if not O:
                # print(out_waveform.float().cpu().detach().shape)
                torchaudio.save('sample.wav', out_waveform[0].float().cpu().detach().unsqueeze(0), sample_rate=24000)
                torchaudio.save('sample_rec.wav', reconstructed_target[0].float().cpu().detach().unsqueeze(0), sample_rate=24000)
                torchaudio.save('sample_tgt.wav', clean_audio[0].cpu().float(), sample_rate=24000)
                O = True

            si_sdr_loss = -scale_invariant_signal_distortion_ratio(
                out_waveform, reconstructed_target.squeeze()
            )

            # print(si_sdr_loss[0])

            si_sdr_loss = si_sdr_loss.mean()

            writer.add_scalar(
                f"Loss/train", si_sdr_loss.detach().cpu(), step
            )
            writer.add_scalar(f"LR/train", scheduler.get_lr()[-1], step)

            total_loss += si_sdr_loss
            si_sdr_loss.backward()
            # for name, p in model.quantizer.quantizer.named_parameters():
            #     if p.requires_grad:
            #         # print(name, p.shape, p.grad)
            #         print(f"{name}: grad = {p.grad:.4g}")
            # break
            optimizer.step()
            scheduler.step()
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
    model.train()
    writer = SummaryWriter(log_dir=log_dir)
    scheduler.step()
    step = 0
    if cfg.overfit:
        loader = [next(iter(loader))]
    for epoch in range(0, num_epochs):
        model.train()

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
        )

        writer.add_scalar(f"Loss/train_epoch", total_loss / len(loader) if not cfg.overfit else total_loss, epoch)
        print(
            f"Epoch {epoch}: total_loss={total_loss / len(loader) :.3f}, lr: {scheduler.get_lr()[-1]}"
        )  # len(loader) #/ len(loader)
        if (epoch % eval_every == 0) and not cfg.overfit:
            val_sdr, val_sdr_pseudo = eval_epoch(
                val_loader, model, device=device, criterion=criterion
            )
            writer.add_scalar(f"SDR/val_epoch", val_sdr, epoch)
            writer.add_scalar(f"SDR_PSEUDO/val_epoch", val_sdr_pseudo, epoch)
        if (epoch % 20 == 0) and not cfg.overfit:
            torch.save(
                model,
                os.path.join(
                    c_root, f"model_{epoch}_ep_loss_{total_loss / len(loader)}.pt"
                ),
            )
            torch.save(optimizer, os.path.join(c_root, f'optimizer_{epoch}_ep_loss_{total_loss / len(loader)}.pt'))
            torch.save(scheduler, os.path.join(c_root, f'scheduler_{epoch}_ep_loss_{total_loss / len(loader)}.pt'))
    writer.flush()
    print("Training complete!")
