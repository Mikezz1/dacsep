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
# from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score


def set_train_mode(model):
    model.train()
    model.quantizer.eval()
    return model


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
                    out = model(
                            mix=noisy_audio,
                            speaker=ref_audio,
                        )
                    out_waveform = model.descript_decode(out)
                    if out_waveform.size(1) != noisy_audio.size(1):
                        out_waveform = F.pad(out_waveform, (0, noisy_audio.size(2) - out_waveform.size(1)))
                else:
                    if cfg.after_quant:
                        out = model(
                            mix=noisy_audio,
                            speaker=ref_audio,
                        )
                        out_waveform = model.decode_latent(out)
                    else:
                        pass
                # out = model.quantizer.upsample(out)
                # decoder_outputs = model.quantizer.decoder_transformer(out.transpose(1, 2))
                # out = decoder_outputs[0].transpose(1, 2)
                
                # out_waveform = model.quantizer.decoder(out).squeeze()#.audio_values.squeeze()

                if cfg.use_descript:
                    x = model.quantizer.preprocess(clean_audio, 24000)
                    z, _, _, _, _ = model.quantizer.encode(x,n_quantizers=8)
                    reconstructed_target = model.quantizer.decode(z).squeeze()
                    if reconstructed_target.size(1) != clean_audio.size(1):
                        reconstructed_target = F.pad(reconstructed_target, (0, clean_audio.size(2) - reconstructed_target.size(1)))
                else:

                    codes = model.quantizer.encode(clean_audio, num_quantizers=8).audio_codes
                    reconstructed_target = model.quantizer.decode(codes).audio_values.squeeze()

                # dns_mos += deep_noise_suppression_mean_opinion_score(out_waveform.float(), fs=24000, personalized=True)

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
        # (dns_mos / len(loader)).item(),
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

                _out = model(
                        mix=noisy_audio,
                        speaker=ref_audio,
                    )
                out_waveform = model.descript_decode(_out)
                if out_waveform.size(1) != noisy_audio.size(1):
                    out_waveform = F.pad(out_waveform, (0, noisy_audio.size(2) - out_waveform.size(1)))



            #out_waveform = model.quantizer.decode(codes).audio_values.squeeze()

            # codes = model.quantizer.encode(clean_audio, num_quantizers=8).audio_codes
            # reconstructed_target = model.quantizer.decode(codes).audio_values.squeeze()

            if cfg.use_descript:
                with torch.no_grad():
                    x = model.quantizer.preprocess(clean_audio, 24000)
                    z, _, _, _, _ = model.quantizer.encode(x,n_quantizers=8)
                    reconstructed_target = model.quantizer.decode(z).squeeze()
                    if reconstructed_target.size(1) != clean_audio.size(2):
                        reconstructed_target = F.pad(reconstructed_target, (0, clean_audio.size(2) - reconstructed_target.size(1)))

            # if not O:
            #     # print(out_waveform.float().cpu().detach().shape)
            #     torchaudio.save('sample.wav', out_waveform[0].float().cpu().detach().unsqueeze(0), sample_rate=24000)
            #     torchaudio.save('sample_rec.wav', reconstructed_target[0].float().cpu().detach().unsqueeze(0), sample_rate=24000)
            #     torchaudio.save('sample_tgt.wav', clean_audio[0].cpu().float(), sample_rate=24000)
            #     O = True

            si_sdr_loss = -scale_invariant_signal_distortion_ratio(
                out_waveform, reconstructed_target.squeeze()
            )

            latent_loss = 0
            if cfg.add_latent_loss:
                if cfg.use_descript:
                    latent_loss = F.l1_loss(_out, model.descript_encode(clean_audio, n_quantizers=8)).mean()
                else:
                    latent_loss = F.l1_loss(_out.transpose(1,2), model.get_latent(clean_audio)).mean()

            # print(si_sdr_loss[0])

            writer.add_scalar(
                f"Loss/train", si_sdr_loss.mean().detach().cpu(), step
            )

            if cfg.add_latent_loss:
                writer.add_scalar(
                    f"latent_loss/train", latent_loss.detach().cpu(), step
                )

            si_sdr_loss = si_sdr_loss.mean() + latent_loss
            si_sdr_loss.backward()

            writer.add_scalar(
                f"Loss/train", si_sdr_loss.detach().cpu(), step
            )
            writer.add_scalar(f"LR/train", scheduler.get_lr()[-1], step)

            total_loss += si_sdr_loss.detach().cpu()
            # for name, p in model.encoder.named_parameters():
            #     if p.requires_grad:
            #         # print(name, p.shape, p.grad)
            #         print(f"{name}: grad = {p.grad.norm():.4g}")
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
        )

        writer.add_scalar(f"Loss/train_epoch", total_loss / len(loader) if not cfg.overfit else total_loss, epoch)
        print(
            f"Epoch {epoch}: total_loss={total_loss / len(loader) :.3f}, lr: {scheduler.get_lr()[-1]}"
        )  # len(loader) #/ len(loader)
        if (epoch % eval_every == 0) and not cfg.overfit:
            val_sdr, val_sdr_pseudo = eval_epoch( # dnsmos
                val_loader, model, device=device, criterion=criterion, cfg=cfg,
            )
            writer.add_scalar(f"SDR/val_epoch", val_sdr, epoch)
            writer.add_scalar(f"SDR_PSEUDO/val_epoch", val_sdr_pseudo, epoch)
            # write.add_scalar(f"dnsmos/val_epoch", dnsmos, epoch)
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
