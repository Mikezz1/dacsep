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

    mimi_model = MimiModel.from_pretrained(
        "kyutai/mimi", cache_dir="/mike_migrate2/hf_models2"
    )
    mimi_model = mimi_model.to(device)

    dataset_val = LibrimixDataset(
        data_dir="/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/dev/",
        # manifest='/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/dev/manifest.csv',
        cut_len=48_000
    )

    val_loader = DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=2
    )


    model = TransformerEncoderLatent(
        quantizer=mimi_model,
        vocab_size=2048,
        embed_dim=256,
        num_heads=16,
        num_layers=16,
        hidden_dim=1024,
    ).to(device)

    batch = next(iter(val_loader))


    #model.load_state_dict(torch.load('/mike_migrate2/checkpoints/2025_04_22/exp_poc_latent_mask_tanh_v0/model_20_ep_loss_-7.889787197113037.pt', weights_only=False).state_dict())
    model.eval()

    # audio, sr = torchaudio.load('/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/dev/mix_clean/8842-304647-0012_3752-4943-0024.wav')
    # audio = torchaudio.functional.resample(audio,16000, 24000)
    # audio = audio[:, :48000].unsqueeze(1).to(device)

    audio = batch[0].to(device).unsqueeze(1)
    gt = batch[1].to(device).unsqueeze(1)
    ref = batch[2].to(device).unsqueeze(1)

    # ref, sr = torchaudio.load('/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/dev/s2/8842-304647-0012_3752-4943-0024.wav')
    # ref = torchaudio.functional.resample(ref,16000, 24000)
    # ref = ref[:, :48000].unsqueeze(1).to(device)
    
    with torch.inference_mode():

        out = model(
            mix=audio,
            speaker=ref,
        )
        codes = model.quantizer.quantizer.encode(out, num_quantizers=8).transpose(0,1)
        out_waveform = model.quantizer.decode(codes).audio_values.squeeze()


        codes = model.quantizer.encode(gt, num_quantizers=8).audio_codes
        reconstructed_target = model.quantizer.decode(codes).audio_values.squeeze()


    print(scale_invariant_signal_distortion_ratio(out_waveform.squeeze(), gt.squeeze()))
    print(scale_invariant_signal_distortion_ratio(out_waveform.squeeze(), reconstructed_target.squeeze()))

    torchaudio.save('out_waveform.wav', out_waveform.unsqueeze(0).cpu().detach(), sample_rate=24000)
    torchaudio.save('gt_waveform.wav', gt.squeeze(0).cpu().detach(), sample_rate=24000)
    torchaudio.save('ref_waveform.wav', ref.squeeze(0).cpu().detach(), sample_rate=24000)
    torchaudio.save('reconstructed_target_waveform.wav', reconstructed_target.unsqueeze(0).cpu().detach(), sample_rate=24000)



