from codec_sep.trainer import *
from codec_sep.data import LibrimixDataset
from codec_sep.model import TransformerEncoderWithCodebookCondition, TransformerEncoderLatent
from codec_sep.utils import _to_namespace, count_parameters, lr_lambda


import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Config, GPT2LMHeadModel, AutoFeatureExtractor
from transformers import MimiModel, AutoFeatureExtractor

import torchaudio
import yaml  # pip install pyyaml
from types import SimpleNamespace
from pathlib import Path

from tqdm import tqdm
import numpy as np
from pathlib import Path

import os

from torch.optim.lr_scheduler import LambdaLR
import math
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from functools import partial

from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

import dac

DAC = True

if __name__ == "__main__":
    device = torch.device("cuda:0")

    torch.backends.cudnn.benchmark = False
    # # torch.set_float32_matmul_precision("medium")
    # torch.backends.cuda.enable_flash_sdp(True)
    # torch.backends.cuda.enable_mem_efficient_sdp(False)
    # torch.backends.cuda.enable_math_sdp(False)

    cfg_dict = yaml.safe_load(Path("config.yaml").read_text())
    cfg = _to_namespace(cfg_dict)
    print(cfg)

    if DAC:
        model_path = dac.utils.download(model_type="24khz")
        quantizer = dac.DAC.load(model_path)

    else:
        quantizer = MimiModel.from_pretrained(
            "kyutai/mimi", cache_dir="/mike_migrate2/hf_models2"
        )
        quantizer = mimi_model.to(device)

    dataset_val = LibrimixDataset(
        data_dir="/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/dev/",
        # manifest='/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/dev/manifest.csv',
        cut_len=192000,
        sample_rate_tgt=24000,
    )

    bs = 32

    val_loader = DataLoader(
        dataset_val, batch_size=bs, shuffle=False, num_workers=6
    )


    model = TransformerEncoderLatent(
        quantizer=quantizer,
        vocab_size=2048,
        embed_dim=256,
        num_heads=8,
        num_layers=16,
        hidden_dim=1024,
        after_quant=False,
        descript=True,
        twin_tower=True,
        film=True,
        sample_rate=24000,

    ).to(device)

    # batch = next(iter(val_loader))


    # model.load_state_dict(torch.load('/mike_migrate2/checkpoints/2025_04_22/exp_poc_latent_mask_tanh_v0/model_20_ep_loss_-7.889787197113037.pt', weights_only=False).state_dict())
    model.load_state_dict(torch.load('/mike_migrate2/checkpoints/2025_04_22/ablation_4_add_prompt/model.pt', weights_only=False).state_dict())
    print(1)
    model.eval()

    # audio, sr = torchaudio.load('/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/dev/mix_clean/8842-304647-0012_3752-4943-0024.wav')
    # audio = torchaudio.functional.resample(audio,16000, 24000)

    # audio = audio[:, :48000].unsqueeze(1).to(device)
    t = 32_000
    r = -1
    for t in [-1]:  # , 96_000, 48_000]:
        for r in [24_000 * 8]:  # , 96_000, 48_000]:
            res = 0
            dns = 0
            real_dns = 0
            for batch in tqdm(val_loader):

                audio = batch[0][:, :t].to(device).unsqueeze(1)
                gt    = batch[1][:, :t].to(device).unsqueeze(1)
                ref   = batch[2][:, :r].to(device).unsqueeze(1)

                # ----- CHUNK-WISE INFERENCE PARAMETERS -----
                CHUNK_SEC   = 4                  # seconds
                CHUNK_SAMPLES = CHUNK_SEC * 24_000  # 96 000 @ 24 kHz
                # -------------------------------------------

                with torch.inference_mode():

                    if DAC:
                        a = audio
                        for _ in range(1):      # keep original DAC refinement structure
                            hidden_chunks = []  # collect _out from every chunk

                            tot_len = a.size(-1)
                            for beg in range(0, tot_len, CHUNK_SAMPLES):
                                end = min(beg + CHUNK_SAMPLES, tot_len)
                                mix_chunk = a[..., beg:end]          # (B,1,T_chunk)

                                _out_chunk, _ = model(
                                    mix=mix_chunk,
                                    speaker=ref,
                                )
                                hidden_chunks.append(_out_chunk)

                            # concatenate hidden states along temporal dimension
                            _out = torch.cat(hidden_chunks, dim=2)   # (B, ΣT_tokens, D)

                            # single decode for the whole utterance
                            out_waveform = model.descript_decode(_out).unsqueeze(1)
                            a = out_waveform                         # feedback for DAC loop

                        # ensure length match with original audio
                        if out_waveform.size(2) != audio.size(2):
                            if out_waveform.size(2) > audio.size(2):
                                out_waveform = out_waveform[..., : audio.size(2)]
                            else:
                                out_waveform = F.pad(
                                    out_waveform,
                                    (0, audio.size(2) - out_waveform.size(2)),
                                )

                        # -------- downstream metric / reconstruction code (unchanged) --------
                        x = model.quantizer.preprocess(gt, 24000)
                        z, _, _, _, _ = model.quantizer.encode(x, n_quantizers=None)
                        reconstructed_target = model.quantizer.decode(z).squeeze().unsqueeze(0)

                        if reconstructed_target.size(2) != gt.size(2):
                            reconstructed_target = F.pad(
                                reconstructed_target,
                                (0, gt.size(2) - reconstructed_target.size(2)),
                            )
                    else:
                        out = model(
                            mix=audio,
                            speaker=ref,
                        )
                        out = model.quantizer.upsample(out)
                        decoder_outputs = model.quantizer.decoder_transformer(out.transpose(1, 2))
                        out = decoder_outputs[0].transpose(1, 2)
                        out_waveform = model.quantizer.decoder(out).squeeze()#.audio_values.squeeze()


                        codes = model.quantizer.encode(gt, num_quantizers=8).audio_codes
                        reconstructed_target = model.quantizer.decode(codes).audio_values.squeeze()

                    cur = scale_invariant_signal_distortion_ratio(out_waveform.squeeze(), reconstructed_target.squeeze()).sum()
                    res += cur
                    print(cur)
                    dns += deep_noise_suppression_mean_opinion_score(out_waveform.squeeze(), fs=24000, personalized=False)[:, -1].sum()
                    real_dns += deep_noise_suppression_mean_opinion_score(reconstructed_target.squeeze(), fs=24000, personalized=False)[:, -1].sum()

            print(t, r, res / len(val_loader) / bs, dns / len(val_loader) / bs, real_dns / len(val_loader) / bs )



    # print(scale_invariant_signal_distortion_ratio(out_waveform.squeeze(), gt.squeeze()).mean())
    # print(scale_invariant_signal_distortion_ratio(out_waveform.squeeze(), reconstructed_target.squeeze()).mean())

    # torchaudio.save('mix.wav', audio.squeeze(0).cpu().detach(), sample_rate=16000)

    # torchaudio.save('out_waveform.wav', out_waveform.cpu().detach(), sample_rate=16000)
    # torchaudio.save('gt_waveform.wav', gt.squeeze(0).cpu().detach(), sample_rate=16000)
    # torchaudio.save('ref_waveform.wav', ref.squeeze(0).cpu().detach(), sample_rate=16000)
    # torchaudio.save('reconstructed_target_waveform.wav', reconstructed_target.cpu().detach(), sample_rate=16000)




