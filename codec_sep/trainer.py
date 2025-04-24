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
        noisy_audio = batch[0].to(device)
        clean_audio = batch[1].to(device)
        ref_audio = batch[2].to(device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                input_codes = model.quantizer.encode(
                    noisy_audio.unsqueeze(1), num_quantizers=8
                )["audio_codes"]
                target_codes = model.quantizer.encode(
                    clean_audio.unsqueeze(1), num_quantizers=8
                )["audio_codes"]

                speaker_codes = model.quantizer.encode(
                    ref_audio.unsqueeze(1), num_quantizers=8
                )["audio_codes"]

                for j in range(model.num_codebooks):
                    x_j = input_codes
                    if j > 0:
                        past_tokens_list = target_codes[
                            :, :j, :
                        ]  # [target_codes[:, k, :] for k in range(j)]
                    else:
                        past_tokens_list = None

                    logits_j = model(
                        current_tokens=x_j,
                        codebook_idx=j,
                        speaker_emb=None,
                        past_codebook_tokens=past_tokens_list,
                        speaker_codes=speaker_codes,
                    )  # [B, T, 2048]9

                    y_j = target_codes[:, j, :]  # [B, T]

                    loss_j = criterion(logits_j.reshape(-1, 2048), y_j.reshape(-1))
                    total_loss += loss_j.item()

                predicted_codes = [None] * model.num_codebooks
                for j in range(model.num_codebooks):
                    # We'll use the PREDICTED codes from [0..j-1], if j>0
                    past_tokens_list = []
                    for past_idx in range(j):
                        # shape => [B, T_code]
                        past_tokens_list.append(predicted_codes[past_idx])

                    if len(past_tokens_list) > 0:
                        past_tokens_list = torch.stack(past_tokens_list, dim=1)
                    else:
                        past_tokens_list = None
                    x_j_input = input_codes

                    with torch.no_grad():
                        logits_j = model(
                            current_tokens=x_j_input,  # shape [B, T]
                            codebook_idx=j,
                            speaker_emb=None,
                            past_codebook_tokens=past_tokens_list,
                            speaker_codes=speaker_codes,
                        )
                        pred_j = torch.argmax(logits_j, dim=-1)

                    predicted_codes[j] = pred_j
                predicted_codes_stacked = torch.stack(predicted_codes, dim=1)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        pred_waveform = model.quantizer.decode(
                            predicted_codes_stacked
                        ).audio_values.squeeze()
                        pred_gt = model.quantizer.decode(
                            target_codes
                        ).audio_values.squeeze()
                si_sdr += (
                    scale_invariant_signal_distortion_ratio(pred_waveform, clean_audio)
                    .mean()
                    .detach()
                    .cpu()
                )
                si_sdr_pseudo += (
                    scale_invariant_signal_distortion_ratio(pred_waveform, pred_gt)
                    .mean()
                    .detach()
                    .cpu()
                )
                # si_sdr += scale_invariant_signal_distortion_ratio(
                #     torchaudio.functional.resample(pred_waveform,24000,16000),
                #     torchaudio.functional.resample(clean_audio,24000,16000)
                #     ).mean().detach().cpu()
                # si_sdr_pseudo += scale_invariant_signal_distortion_ratio(
                #     torchaudio.functional.resample(pred_waveform, 24000, 16000),
                #     torchaudio.functional.resample(pred_gt, 24000, 16000),
                #     ).mean().detach().cpu()
                # si_sdr_pseudo_lowpass += scale_invariant_signal_distortion_ratio(
                #     torchaudio.functional.lowpass_biquad(pred_waveform.float(), sample_rate=24000, cutoff_freq=8000),
                #     torchaudio.functional.lowpass_biquad(pred_gt.float(), sample_rate=24000, cutoff_freq=8000)
                #     ).mean().detach().cpu()

                # si_sdr_mix += scale_invariant_signal_distortion_ratio(noisy_audio, clean_audio).mean().detach().cpu()
                # pesqs += pesq(torchaudio.functional.resample(noisy_audio,24000,16000).float(),
                #               torchaudio.functional.resample(clean_audio,24000,16000).float(),fs=16000, mode='wb').detach().cpu()

    return (
        total_loss / len(loader),
        (si_sdr / len(loader)).item(),
        (si_sdr_pseudo / len(loader)).item(),
    )  # , (si_sdr_pseudo_lowpass  / len(loader)).item()


USE_SPEAKER_EMBEDDER = False
N_CODEBOOKS = 8


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
    for batch in tqdm(loader):
        noisy_audio = batch[0].to(device)
        clean_audio = batch[1].to(device)
        ref_audio = batch[2].to(device)

        if cfg.variable_len_train:
            max_len_sample = np.random.randint(1, 7) * 24000
            noisy_audio = noisy_audio[:, :max_len_sample]
            clean_audio = clean_audio[:, :max_len_sample]

        if cfg.variable_len_ref:
            max_len_ref = np.random.randint(1, 7) * 24000
            ref_audio = ref_audio[:, :max_len_ref]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_codes = model.quantizer.encode(
                noisy_audio.unsqueeze(1), num_quantizers=8
            )["audio_codes"]
            target_codes = model.quantizer.encode(
                clean_audio.unsqueeze(1), num_quantizers=8
            )["audio_codes"]

            if cfg.add_si_sdr_loss:
                reconstructed_target = model.quantizer.decode(
                    target_codes
                ).audio_values.squeeze()

            speaker_codes = None
            speaker_codes = model.quantizer.encode(
                ref_audio.unsqueeze(1), num_quantizers=8
            )["audio_codes"]
            speaker_emb = None

        js = list(range(N_CODEBOOKS))
        # np.random.shuffle(js)
        optimizer.zero_grad()
        _preds_final = []
        for j in js:
            x_j = input_codes
            if j > 0:
                past_tokens_list = target_codes[
                    :, :j, :
                ]  # [target_codes[:, k, :] for k in range(j)]
            else:
                past_tokens_list = None

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_j = model(
                    current_tokens=x_j,
                    codebook_idx=j,
                    speaker_emb=speaker_emb,
                    past_codebook_tokens=past_tokens_list,
                    speaker_codes=speaker_codes,
                )  # [B, T, 2048]9

                y_j = target_codes[:, j, :]  # [B, T]

                if cfg.add_si_sdr_loss:

                    _preds_final.append(torch.argmax(logits_j, dim=-1))

                    if (j == 7) or cfg.add_si_sdr_loss_every_codebook:
                        preds = torch.stack(_preds_final, dim=1)
                        reconstructed_pred = model.quantizer.decode(
                            preds
                        ).audio_values.squeeze()

                        si_sdr_loss = -scale_invariant_signal_distortion_ratio(
                            reconstructed_pred, reconstructed_target
                        ).mean()
                    else:
                        si_sdr_loss = 0.0

                loss_j = criterion(logits_j.reshape(-1, 2048), y_j.reshape(-1))
                writer.add_scalar(
                    f"Loss/train/codebook_{j}", loss_j.detach().cpu(), step
                )
                writer.add_scalar(f"LR/train", scheduler.get_lr()[-1], step)

                if cfg.add_si_sdr_loss:
                    loss_j += si_sdr_loss
                    if (j == 7) or  cfg.add_si_sdr_loss_every_codebook:
                        writer.add_scalar(
                            f"Loss/train/si_sdr_loss_{j}", si_sdr_loss.detach().cpu(), step
                        )

                total_loss += loss_j.item()
                loss_j *= 1 / (j + 1) ** 0.5
                loss_j.backward()
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

        writer.add_scalar(f"Loss/train_epoch", total_loss / len(loader), epoch)
        print(
            f"Epoch {epoch}: total_loss={total_loss / len(loader) :.3f}, lr: {scheduler.get_lr()[-1]}"
        )  # len(loader) #/ len(loader)
        if epoch % eval_every == 0:
            val_loss, val_sdr, val_sdr_pseudo = eval_epoch(
                val_loader, model, device=device, criterion=criterion
            )
            writer.add_scalar(f"Loss/val_epoch", val_loss, epoch)
            writer.add_scalar(f"SDR/val_epoch", val_sdr, epoch)
            writer.add_scalar(f"SDR_PSEUDO/val_epoch", val_sdr_pseudo, epoch)
        if epoch % 50 == 0:
            torch.save(
                model,
                os.path.join(
                    c_root, f"model_{epoch}_ep_loss_{total_loss / len(loader)}.pt"
                ),
            )
            # torch.save(optimizer, os.path.join(c_root, f'optimizer_{epoch}_ep_loss_{total_loss / len(loader)}.pt'))
            # torch.save(scheduler, os.path.join(c_root, f'scheduler_{epoch}_ep_loss_{total_loss / len(loader)}.pt'))
    writer.flush()
    print("Training complete!")
