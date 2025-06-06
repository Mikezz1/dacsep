#!/usr/bin/env python
# coding: utf-8
"""
Extract 10×10 speaker embeddings, cluster them and visualise with t-SNE.
"""

import os
import random
from collections import defaultdict
import torch
import torchaudio
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

BASE_DIR = "/mike_migrate2/data/Libri2Mix/wav8k/min/test"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SPK    = 10     
N_UTT    = 10      
SEED     = 122     
MODEL_CKPT = "/mike_migrate2/checkpoints/2025_04_22/exp_poc_DAC_v51_ft_emb/model.pt"


random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def speaker_id_from_path(path: str) -> str:
    """
    Libri2Mix file names look like
    s1/61-70968-0000_8455-210777-0012.wav
    For s1 files we consider the first triplet, i.e. 61-70968-0000,
    speaker id == first part before first dash ('61').
    """
    fname = os.path.basename(path)
    first_triplet = fname.split("_")[0]
    spk = first_triplet.split("-")[0]
    return spk

wav_groups = defaultdict(list)
for root, _, files in os.walk(os.path.join(BASE_DIR, "s1")):
    for f in files:
        if f.endswith(".wav"):
            full = os.path.join(root, f)
            spk = speaker_id_from_path(full)
            wav_groups[spk].append(full)

# keep only speakers with ≥ N_UTT utterances
eligible = {s: lst for s, lst in wav_groups.items() if len(lst) >= N_UTT}
if len(eligible) < N_SPK:
    raise RuntimeError(f"Найдено всего {len(eligible)} спикеров с ≥{N_UTT} файлами")

selected_speakers = random.sample(list(eligible.keys()), N_SPK)

file_list = []
true_labels = []
for spk in selected_speakers:
    chosen = random.sample(eligible[spk], N_UTT)
    file_list.extend(chosen)
    true_labels.extend([spk] * N_UTT)

from codec_sep.model import TransformerEncoderLatent
import dac

model_path = dac.utils.download(model_type="24khz")
quantizer = dac.DAC.load(model_path)

model = TransformerEncoderLatent(
    quantizer      = quantizer,
    vocab_size     = 2048,
    embed_dim      = 256,
    num_heads      = 8,
    num_layers     = 16,
    hidden_dim     = 1024,
    after_quant    = False,
    descript       = True,
    twin_tower     = True,
    film           = True,
    sample_rate    = 24000,
).to(DEVICE)

state = torch.load(MODEL_CKPT, map_location="cuda", weights_only=False)
model.load_state_dict(state.state_dict())
model.eval()
sample_rate_model = model.sample_rate  #

@torch.no_grad()
def extract_stage_embedding(model, waveform, sr):
    """Return 1-D numpy array of stage_emb (shape: embed_dim)."""
    # resample if needed
    if sr != sample_rate_model:
        resampler = torchaudio.transforms.Resample(sr, sample_rate_model)
        waveform = resampler(waveform)
   
    wav = waveform.unsqueeze(0).to(DEVICE)
    print(wav.size())

    if model.descript:
        emb = model.descript_encode(wav).transpose(1, 2)       
    else:
        if model.after_quant:
            emb = model.get_latent(wav)
        else:
            emb = model.get_latent_pre_quant(wav)
    emb = model.ch_down(emb) 
    if model.twin_tower:
        emb = model.speaker_encoder(emb)
    stage_emb = model.pooling(emb) 
    return stage_emb.squeeze(0).cpu().numpy()


all_embs = []
for path in file_list:
    wav, sr = torchaudio.load(path)
    emb = extract_stage_embedding(model, wav, sr)
    all_embs.append(emb)
all_embs = np.stack(all_embs)    


kmeans = KMeans(n_clusters=N_SPK, n_init="auto", random_state=SEED)
clusters = kmeans.fit_predict(all_embs)


tsne = TSNE(n_components=2, init="pca", random_state=SEED, perplexity=30)
emb_2d = tsne.fit_transform(all_embs)  # (100, 2)

plt.figure(figsize=(8, 6))
uniq_spk = sorted(selected_speakers)
color_map = {spk: idx for idx, spk in enumerate(uniq_spk)}
for (x, y), spk in zip(emb_2d, true_labels):
    plt.scatter(x, y, label=spk, c=[plt.cm.tab10(color_map[spk])], s=40)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), title="Speaker ID", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.title(f"t-SNE projection of speaker embeddings {N_SPK} speakers × {N_UTT} utt.)")
plt.xlabel("Dim-1")
plt.ylabel("Dim-2")
plt.tight_layout()
plt.savefig("tsne_speakers_4.png", dpi=150, transparent=True)
