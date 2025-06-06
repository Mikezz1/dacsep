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

import whisperx
import editdistance

# -----------------------------------------------------------
# Minimal WER helper for a LibriMix-style training loop
# -----------------------------------------------------------
import re, string, torch, torchaudio, editdistance
from transformers import pipeline


# Faster – one Whisper call per mini-batch instead of per utterance
from itertools import islice

# Whisper works at 16 kHz ⇒ resample from 24 kHz once per batch
_SR_IN, _SR_WHISPER = 24_000, 16_000

# pipe = pipeline(
#     task="automatic-speech-recognition",
#     model="openai/whisper-base", # whisper-large-v2
#     device='cuda:0',
#     torch_dtype=torch.bfloat16,
    
# )


_punct = f"[{re.escape(string.punctuation)}]"
def _norm(txt: str) -> str:
    """UPPER-case, no punctuation, single-spaced (≈ LibriSpeech style)."""
    return re.sub(_punct, "", txt).lower().strip()



@torch.inference_mode()
def wer_from_batch(batch_audio: torch.Tensor,
                   refs: list[str],
                   bs: int = 8) -> float:          # bs – Whisper mini-batch size
    """
    batch_audio : (B, L) @ 24 kHz   •   refs : list[str] len == B
    Returns WER for the whole batch (0-1 range).
    """
    # 1) resample whole tensor in one go
    wav16 = torchaudio.functional.resample(batch_audio, _SR_IN, _SR_WHISPER)
    wav16 = wav16.cpu().float().numpy()

    # 2) batched ASR --------------------------------------------------------
    hyps: list[str] = []
    for i in range(0, len(wav16), bs):
        chunk = [{"array": w, "sampling_rate": _SR_WHISPER}
                 for w in wav16[i: i + bs]]
        outs = pipe(chunk, batch_size=bs)          # 1 GPU ⟂ bs streams
        hyps.extend(o["text"] for o in outs)

    # 3) WER ---------------------------------------------------------------
    wers = []
    for ref, hyp in zip(refs, hyps):
        ref_n, hyp_n = _norm(ref), _norm(hyp)
        print(ref_n, hyp_n )
        ref_tok, hyp_tok = ref_n.split(), hyp_n.split()
        wers.append(editdistance.eval(hyp_tok, ref_tok) / len(ref_tok))

    return wers



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
        data_dir="/mike_migrate2/data/Libri2Mix/wav8k/min/test/",
        # manifest='/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/dev/manifest.csv',
        cut_len=None,#96000,
        sample_rate_tgt=24000,

    )
    bs=1

    val_loader = DataLoader(
        dataset_val, batch_size=bs, shuffle=False, num_workers=4
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
    model.load_state_dict(torch.load('/mike_migrate2/checkpoints/2025_04_22/exp_poc_DAC_v51_ft_emb/model.pt', weights_only=False).state_dict())
    print(1)
    model.eval()

    # audio, sr = torchaudio.load('/mike_migrate2/data_16khz/Libri2Mix/wav16k/min/dev/mix_clean/8842-304647-0012_3752-4943-0024.wav')
    # audio = torchaudio.functional.resample(audio,16000, 24000)

    # audio = audio[:, :48000].unsqueeze(1).to(device)
    t = 32_000
    r = -1
    with open('/mike_migrate2/codec-source-sep/test_results/text.txt', 'w') as f:
            
        for t in [-1]:#, 96_000, 48_000]:
            for r in [-1]:#, 96_000, 48_000]:
                res = 0
                dns = 0
                real_dns = 0
                mix_dns = 0
                wers = 0
                wers_clean = 0
                wers_mix = 0
                for i, batch in enumerate(tqdm(val_loader)):

                    audio = batch[0][:, :t].to(device).unsqueeze(1)
                    gt = batch[1][:, :t].to(device).unsqueeze(1)
                    ref = batch[2][:, :r].to(device).unsqueeze(1)
                    texts = batch[-1]
                    

                    # print(batch[-1])
                    # print(batch[-2])
                    # break

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
                                out_waveform = model.descript_decode(_out).unsqueeze(1).unsqueeze(1)
                                a = out_waveform
                            # print(out_waveform.shape,gt.shape )
                            
                            
                            out_waveform = out_waveform.permute(1,2,0)

                            # print(out_waveform.size(), audio.size())
                            if out_waveform.size(2) != audio.size(2):
                                
                                out_waveform = F.pad(out_waveform, (0, audio.size(2) - out_waveform.size(2)))


                            x = model.quantizer.preprocess(gt, 24000)
                            z, _, _, _, _ = model.quantizer.encode(x,n_quantizers=None)
                            reconstructed_target = model.quantizer.decode(z).squeeze().unsqueeze(1).unsqueeze(1)



                            reconstructed_target = reconstructed_target.permute(1,2,0)

                            # print(reconstructed_target.size(), gt.size())


                            if reconstructed_target.size(2) != gt.size(2):
                                reconstructed_target = F.pad(reconstructed_target, (0, gt.size(2) - reconstructed_target.size(2)))
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

                        _sisdr_ = scale_invariant_signal_distortion_ratio(out_waveform.squeeze(), reconstructed_target.squeeze()).sum()
                        print(audio.size(-1) / 24000, ref.size(-1)/24000, _sisdr_)
                        
                        res += _sisdr_
                        
                        # dns += deep_noise_suppression_mean_opinion_score(out_waveform.squeeze(), fs=24000, personalized=False)[ -1].sum()
                        # real_dns += deep_noise_suppression_mean_opinion_score(reconstructed_target.squeeze(), fs=24000, personalized=False)[-1].sum()
                        # mix_dns += deep_noise_suppression_mean_opinion_score(audio.squeeze(), fs=24000, personalized=False)[-1].sum()


                        torchaudio.save(f'/mike_migrate2/codec-source-sep/test_results/out_waveform_{i}.wav',out_waveform.cpu().squeeze(0),  sample_rate=24000)
                        torchaudio.save( f'/mike_migrate2/codec-source-sep/test_results/reconstructed_target_{i}.wav',reconstructed_target.cpu().squeeze(0), sample_rate=24000)
                        torchaudio.save( f'/mike_migrate2/codec-source-sep/test_results/audio_{i}.wav',audio.cpu().squeeze(0), sample_rate=24000)
                        torchaudio.save( f'/mike_migrate2/codec-source-sep/test_results/ref_{i}.wav',ref.cpu().squeeze(0), sample_rate=24000)
                        f.write(texts[0] + '\n')

                        # wer = wer_from_batch(out_waveform.squeeze(0), texts)
                        # wer_clean = wer_from_batch(reconstructed_target.squeeze(0), texts)
                        # wer_mix = wer_from_batch(audio.squeeze(0), texts)
                        # удалить обрезки
                        # print(wer, wer_clean, wer_mix)
                        # wers += wer 
                        # wers_clean += wers_clean
                        # wers_mix += wer_mix

            
            print('csi-sdr:', res / len(val_loader) / bs)
            print('dns:', dns / len(val_loader) / bs)
            print('gt dns:', real_dns / len(val_loader) / bs)
            print('mix dns :', mix_dns / len(val_loader) / bs)
            print('wer: ', wers / len(val_loader) / bs)
            print('wer clean: ', wers_clean / len(val_loader) / bs)
            print('wer mix: ', wers_mix / len(val_loader) / bs)

            print(t, r)



    # print(scale_invariant_signal_distortion_ratio(out_waveform.squeeze(), gt.squeeze()).mean())
    # print(scale_invariant_signal_distortion_ratio(out_waveform.squeeze(), reconstructed_target.squeeze()).mean())

    # torchaudio.save('mix.wav', audio.squeeze(0).cpu().detach(), sample_rate=16000)

    # torchaudio.save('out_waveform.wav', out_waveform.cpu().detach(), sample_rate=16000)
    # torchaudio.save('gt_waveform.wav', gt.squeeze(0).cpu().detach(), sample_rate=16000)
    # torchaudio.save('ref_waveform.wav', ref.squeeze(0).cpu().detach(), sample_rate=16000)
    # torchaudio.save('reconstructed_target_waveform.wav', reconstructed_target.cpu().detach(), sample_rate=16000)




