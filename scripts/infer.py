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
            data_dir="/mike_migrate2/data/Libri2Mix/wav8k/min/test/",
            # manifest='/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/dev/manifest.csv',
            cut_len=None, #192000,
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

        ).to(device)

        batch = next(iter(val_loader))


        # model.load_state_dict(torch.load('/mike_migrate2/checkpoints/2025_04_22/exp_poc_latent_mask_tanh_v0/model_20_ep_loss_-7.889787197113037.pt', weights_only=False).state_dict())
        model.load_state_dict(torch.load('/mike_migrate2/checkpoints/2025_04_22/exp_poc_DAC_v50/model.pt', weights_only=False).state_dict())
        print(1)
        model.eval()

        # audio, sr = torchaudio.load('/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/dev/mix_clean/8842-304647-0012_3752-4943-0024.wav')
        # audio = torchaudio.functional.resample(audio,16000, 24000)
        # audio = audio[:, :48000].unsqueeze(1).to(device)

        audio = batch[0][:, :sr*18].to(device).unsqueeze(1)
        gt = batch[1][:, :sr*18].to(device).unsqueeze(1)
        ref = batch[2][:, :sr*18].to(device).unsqueeze(1)

        text = batch[-1]

        # ref, sr = torchaudio.load('/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/dev/s2/8842-304647-0012_3752-4943-0024.wav')
        # ref = torchaudio.functional.resample(ref,16000, 24000)
        # ref = ref[:, :48000].unsqueeze(1).to(device)
        
        with torch.inference_mode():

            if DAC:
                a = audio
                for _ in range(1):
                    _out, _ = model(
                            mix=a,
                            speaker=ref,
                        )
                    out_waveform = model.descript_decode(_out).unsqueeze(0).unsqueeze(0)
                    print(out_waveform.size(), audio.size())
                    if out_waveform.size(2) != audio.size(2):
                        out_waveform = F.pad(out_waveform, (0, audio.size(2) - out_waveform.size(2)))
                    a = out_waveform



                x = model.quantizer.preprocess(gt, sr)
                z, _, _, _, _ = model.quantizer.encode(x,n_quantizers=None)
                reconstructed_target = model.quantizer.decode(z).squeeze().unsqueeze(0)
                if reconstructed_target.size(1) != gt.size(2):
                    reconstructed_target = F.pad(reconstructed_target, (0, gt.size(2) - reconstructed_target.size(1)))
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


        print(deep_noise_suppression_mean_opinion_score(out_waveform.squeeze(), fs=24000, personalized=False))
        print(scale_invariant_signal_distortion_ratio(out_waveform.squeeze(), gt.squeeze()))
        print(scale_invariant_signal_distortion_ratio(out_waveform.squeeze(), reconstructed_target.squeeze()))
        print('text:', text)
        

        torchaudio.save('0_mix.wav', audio.squeeze(0).cpu().detach(), sample_rate=sr)

        torchaudio.save('1_out_waveform.wav', out_waveform.cpu().detach().squeeze(0), sample_rate=sr)
        torchaudio.save('2_gt_waveform.wav', gt.squeeze(0).cpu().detach(), sample_rate=sr)
        torchaudio.save('3_ref_waveform.wav', ref.squeeze(0).cpu().detach(), sample_rate=sr)
        torchaudio.save('4_reconstructed_target_waveform.wav', reconstructed_target.cpu().detach(), sample_rate=sr)



run()