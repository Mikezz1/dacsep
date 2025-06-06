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

import dac
from thop import profile          #  <<< NEW: import for FLOPs counting


class Mod(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.quantizer.preprocess(x, 24000)
        z, _, _, _, _ = self.model.quantizer.encode(x,n_quantizers=None)
        return z

@torch.compiler.disable(recursive=True)
def run():

    DAC = True

    if __name__ == "__main__":
        device = torch.device("cuda:0")

        sr = 24000

        torch.backends.cudnn.benchmark = False
        # # torch.set_float32_matmul_precision("medium")
        # torch.backends.cuda.enable_flash_sdp(True)
        # torch.backends.cuda.enable_mem_efficient_sdp(False)
        # torch.backends.cuda.enable_math_sdp(False)

        cfg_dict = yaml.safe_load(Path("config.yaml").read_text())
        cfg = _to_namespace(cfg_dict)
        print(cfg)

        if DAC:
            model_path = dac.utils.download(model_type="16khz" if sr ==16000 else "24khz")
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
            sample_rate_tgt=sr,
        )

        val_loader = DataLoader(
            dataset_val, batch_size=1, shuffle=True, num_workers=2
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
            sample_rate=sr,
            conformer_style=False,

        ).to(device)

        mod = Mod(model)

        batch = next(iter(val_loader))

        # model.load_state_dict(torch.load('/mike_migrate2/checkpoints/2025_04_22/exp_poc_latent_mask_tanh_v0/model_20_ep_loss_-7.889787197113037.pt', weights_only=False).state_dict())
        model.load_state_dict(torch.load('/mike_migrate2/checkpoints/2025_04_22/ablation_4_add_prompt/model.pt', weights_only=False).state_dict())
        print(1)
        model.eval()

        audio = batch[0][:, :sr*2].to(device).unsqueeze(1)
        ref = batch[2][:, :sr*2].to(device).unsqueeze(1)

        # 2/2: 150 - 147.5 = 3 GMAC
        # 2/8: 374.9 - 368.9 = 6 GMAC

        # with prompt: 379.3 - 368.9 = 10.4
        # with prompt: 151.4 - 147.5 = 3.9

        # macs, params = profile(mod, inputs=audio, verbose=False) # kwargs=dict(mix=audio, speaker=ref),
        # print(f"GMACS encode: {macs :.4e}")        # print the FLOPs

        # 1.4746e+11 
        # 1.5290e+11

        # --------- NEW: FLOPs counting ----------
        macs, params = profile(model, inputs=(audio, ref), verbose=False) # kwargs=dict(mix=audio, speaker=ref),
        print(f"GMACS: {macs / 1e9 :.4}")        # print the FLOPs


run()
