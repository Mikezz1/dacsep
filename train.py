from codec_sep.trainer import *
from codec_sep.data import LibrimixDataset
from codec_sep.model import TransformerEncoderWithCodebookCondition
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


if __name__ == "__main__":
    device = torch.device("cuda")

    torch.backends.cudnn.benchmark = False
    # torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)

    cfg_dict = yaml.safe_load(Path("config.yaml").read_text())
    cfg = _to_namespace(cfg_dict)
    print(cfg)

    mimi_model = MimiModel.from_pretrained(
        "kyutai/mimi", cache_dir="/mike_migrate2/hf_models"
    )
    mimi_model = mimi_model.to(device)

    model = TransformerEncoderWithCodebookCondition(
        quantizer=mimi_model,
        vocab_size=2048,
        embed_dim=256,
        num_heads=16,
        num_layers=16,
        hidden_dim=1024,
        num_codebooks=cfg.num_codebooks,
        use_speaker_embed=USE_SPEAKER_EMBEDDER,
    ).to(device)

    print('Model size: ', count_parameters(model) / 1e6)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )  # 2.2e-3

    dataset = LibrimixDataset(
        data_dir="/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/train-100/",
        # manifest='/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/train-100/manifest.csv',
        cut_len=cfg.train.cut_len,
    )

    dataset_val = LibrimixDataset(
        data_dir="/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/dev/",
        # manifest='/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/dev/manifest.csv',
        cut_len=cfg.val.cut_len,
    )

    # If each example in the dataset has the same length, the default collate will stack them
    loader = DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=6
    )
    val_loader = DataLoader(
        dataset_val, batch_size=cfg.val.batch_size, shuffle=False, num_workers=2
    )

    total_training_steps = cfg.num_epochs * len(loader)

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=partial(
            lr_lambda,
            warmup_steps=cfg.warmup_steps,
            total_training_steps=total_training_steps,
        ),
    )

    c_root = os.path.join("/mike_migrate2/checkpoints/2025_04_22/", cfg.exp_name)
    log_dir = (
        os.path.join("/mike_migrate2/logs/2025_04_22/", cfg.exp_name)
    )
    os.makedirs(c_root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train(
        loader,
        val_loader,
        model,
        optimizer,
        scheduler,
        criterion,
        num_epochs=cfg.num_epochs,
        device=device,
        c_root=c_root,
        log_dir=log_dir,
        variable_len_train=cfg.variable_len_train,
        variable_len_ref=cfg.variable_len_ref,
        cfg=cfg,
    )
