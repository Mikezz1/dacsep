from codec_sep.trainer import *
from codec_sep.data import LibrimixDataset
from codec_sep.model import (
    TransformerEncoderWithCodebookCondition,
    TransformerEncoderLatent,
)
from codec_sep.utils import _to_namespace, count_parameters, lr_lambda, librimix_collate


import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Config, GPT2LMHeadModel, AutoFeatureExtractor
from transformers import MimiModel, AutoFeatureExtractor

import torchaudio
import yaml  # pip install pyyaml
from pathlib import Path

from tqdm import tqdm
import numpy as np
from pathlib import Path
import shutil
import os

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from functools import partial

import dac



if __name__ == "__main__":
    device = torch.device("cuda")

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)

    cfg_dict = yaml.safe_load(Path("config.yaml").read_text())
    cfg = _to_namespace(cfg_dict)
    print(cfg)

    if not cfg.use_descript:

        quantizer = MimiModel.from_pretrained(
            "kyutai/mimi", cache_dir="/mike_migrate2/hf_models2"
        )
        quantizer = quantizer.to(device)

    else:
        model_path = dac.utils.download(
            model_type="24khz" if cfg.sample_rate == 24000 else "16khz"
        )
        quantizer = dac.DAC.load(model_path)
        print(quantizer)

    model = TransformerEncoderLatent(
        quantizer=quantizer,
        vocab_size=2048,
        embed_dim=256,
        num_heads=8,
        num_layers=16,
        hidden_dim=1024,
        after_quant=cfg.after_quant,
        descript=cfg.use_descript,
        twin_tower=cfg.twin_tower,
        film=cfg.film,
        sample_rate=cfg.sample_rate,
    ).to(device)

    print("Model size: ", count_parameters(model) / 1e6)

    criterion = nn.CrossEntropyLoss()
    trainable = (
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("quantizer")
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )  # 2.2e-3

    dataset = LibrimixDataset(
        data_dir="/mike_migrate2/data/Libri2Mix/wav8k/min/train-100/",
        # manifest='/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/train-100/manifest.csv',
        cut_len=cfg.train.cut_len,
        sample_rate_tgt=cfg.sample_rate,
        return_second_speaker_no_mix=cfg.dynamic_mixing,
    )

    dataset_val = LibrimixDataset(
        data_dir="/mike_migrate2/data/Libri2Mix/wav8k/min/test/",
        # manifest='/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/dev/manifest.csv',
        cut_len=cfg.val.cut_len,
        sample_rate_tgt=cfg.sample_rate,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=not cfg.overfit,
        num_workers=8,
        prefetch_factor=2,
        persistent_workers=True,
        # collate_fn=partial(librimix_collate, sr=cfg.sample_rate),
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
    log_dir = os.path.join("/mike_migrate2/logs/2025_04_22/", cfg.exp_name)
    os.makedirs(c_root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    shutil.copyfile(
        "/mike_migrate2/codec-source-sep/config.yaml",
        os.path.join(log_dir, "config.yaml"),
    )

    model.load_state_dict(
        torch.load(
            "/mike_migrate2/checkpoints/2025_04_22/exp_poc_DAC_v50/model.pt",
            weights_only=False,
        ).state_dict()
    )
    # scheduler.load_state_dict(torch.load('/mike_migrate2/checkpoints/2025_04_22/exp_poc_DAC_pseudoSDR_snake_high_lr_snake_film_24khz_v44/scheduler.pt', weights_only=False).state_dict())
    # optimizer.load_state_dict(torch.load('/mike_migrate2/checkpoints/2025_04_22/exp_poc_DAC_v50/optimizer.pt', weights_only=False).state_dict())

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
