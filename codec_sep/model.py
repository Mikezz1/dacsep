import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm
from transformers.models.mimi.modeling_mimi import MimiVectorQuantization

import copy
import torch.nn as nn


from rotary_embedding_torch import RotaryEmbedding


class SDPA(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, use_rope=False):
        super().__init__()
        self.nh = nhead
        self.hd = d_model // nhead
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(dim=self.hd//2, theta=10000)

    def _split(self, t):
        b, s, _ = t.size()
        return t.view(b, s, self.nh, self.hd).transpose(1, 2)  # B H S D

    def _merge(self, t):
        b, h, s, d = t.size()
        return t.transpose(1, 2).reshape(b, s, h * d)

    def forward(self, q_in, k_in, v_in, attn_mask=None, key_padding_mask=None):
        q, k, v = self._split(self.q(q_in)), self._split(self.k(k_in)), self._split(self.v(v_in))
        if self.use_rope:
            q, k = self.rope.rotate_queries_or_keys(q), self.rope.rotate_queries_or_keys(k)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :].to(torch.bool)
            attn_mask = key_padding_mask if attn_mask is None else attn_mask.logical_or(key_padding_mask)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask)
        return self.o(self._merge(self.drop(out)))


# class ChannelStatsPooling(nn.Module):
#     """
#     Channel- & context-dependent statistics pooling
#     (ECAPA-TDNN §3.1).

#     Parameters
#     ----------
#     feat_dim : int
#         Size of per-frame feature vector (C in the paper).
#     attn_dim : int, optional
#         The shared intermediate size R (default: 128).
#     eps : float, optional
#         Small number for numerical stability (default: 1e-12).
#     """
#     def __init__(self, hidden_dim: int, attn_dim: int = 128, eps: float = 1e-12):
#         super().__init__()
#         self.eps = eps

#         # W, b  – shared projection  (B, T, C) → (B, T, R)
#         self.proj = nn.Linear(hidden_dim//2, hidden_dim//2)
#         self.proj2 =  nn.Linear(hidden_dim, hidden_dim//2)

#         # v_c, k_c – channel-specific scorer (B, T, R) → (B, T, C)
#         self.channel_attn = nn.Linear(hidden_dim//2, hidden_dim//2, bias=True)

#     def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
#         """
#         x    : (B, T, C)  – frame-level activations
#         mask : (B, T)     – 1 = valid frame, 0 = padding (optional)
#         """
#         # (1) e_{t,c} = v_cᵀ tanh(W h_t + b) + k_c
#         x = self.proj2(x)
#         e = self.channel_attn(torch.tanh(self.proj(x)))        # (B, T, C)


#         if mask is not None:
#             e = e.masked_fill(mask[:, :, None] == 0, -1e9)     # ignore padding

#         # (2) α_{t,c} = softmax_t(e_{t,c})
#         alpha = F.softmax(e, dim=1)                            # (B, T, C)

#         # (3) μ̃_c = Σ_t α_{t,c} h_{t,c}
#         mean = torch.sum(alpha * x, dim=1)                     # (B, C)

#         # (4) σ̃_c = √(Σ_t α_{t,c} h_{t,c}² − μ̃_c²)
#         var  = torch.sum(alpha * x * x, dim=1) - mean * mean
#         std  = torch.sqrt(var.clamp(min=self.eps))             # (B, C)

#         # final statistics vector [μ̃ || σ̃]  →  (B, 2 C)
#         out =  torch.cat([mean, std], dim=1).flatten(1)
#         return out



class SelfAttnPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):  
        x = self.norm(x)        # x: (B, T, H)
        w = self.score(x).squeeze(-1)         # (B, T)
        if mask is not None:                  # mask: (B, T) — 1 for valid tokens, 0 for padding
            w = w.masked_fill(mask == 0, -1e9)
        α = F.softmax(w, dim=-1).unsqueeze(-1)  # (B, T, 1)
        return torch.sum(α * x, dim=1)        # (B, H)


class AdaLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, emb):
        if emb is None:
            return self.ln(x)
        g, b = self.proj(emb).chunk(2, -1)
        y = self.ln(x)
        return (1 + g.unsqueeze(1)) * y + b.unsqueeze(1)


# class CrossTransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward, activation="relu", batch_first=True, norm_first=True, cross_attention=True, film=False):
#         super().__init__()
#         self.self_attn = SDPA(d_model, nhead, use_rope=True)
#         self.has_cross_attention = cross_attention
#         if cross_attention:
#             self.cross_attn = SDPA(d_model, nhead, use_rope=True)

#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.norm_first = norm_first
#         norm_cls = AdaLayerNorm if film else nn.LayerNorm
#         self.norm1 = norm_cls(d_model)
#         if cross_attention:
#             self.norm2 = norm_cls(d_model)
#         self.norm3 = norm_cls(d_model)
#         self.dropout = nn.Dropout(0.1)
#         self.act = {"relu": nn.ReLU(), "gelu": nn.GELU(), "snake": Snake1d(dim_feedforward)}[activation]
#         self.adapt = film

#     def _n(self, mod, x, emb):
#         return mod(x, emb) if self.adapt else mod(x)

#     def _self_attention(self, x, mask, pad):
#         return self.self_attn(x, x, x, mask, pad)

#     def _cross_attention(self, x, m, mask, pad):
#         return self.cross_attn(x, m, m, mask, pad)

#     def _ff(self, x):
#         return self.dropout(self.linear2(self.act(self.linear1(x))))

#     def forward(self, x, stage_emb=None, mem=None, src_mask=None, src_pad=None, mem_mask=None, mem_pad=None):
#         if self.norm_first:
#             x = x + self._self_attention(self._n(self.norm1, x, stage_emb), src_mask, src_pad)
#             if (mem is not None) and self.has_cross_attention:
#                 x = x + self._cross_attention(self._n(self.norm2, x, stage_emb), mem, mem_mask, mem_pad)
#             x = x + self._ff(self._n(self.norm3, x, stage_emb))
#         else:
#             y = self._n(self.norm1, x + self._self_attention(x, src_mask, src_pad), stage_emb)
#             if mem is not None:
#                 y = self._n(self.norm2, y + self._cross_attention(y, mem, mem_mask, mem_pad), stage_emb)
#             x = self._n(self.norm3, y + self._ff(y), stage_emb)
#         return x

class CrossTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        activation="relu",
        batch_first=True,
        norm_first=True,
        cross_attention=True,
        film=False,
        conformer_style=True,
    ):
        super().__init__()
        self.self_attn = SDPA(d_model, nhead, use_rope=True)
        if cross_attention:
            self.cross_attn = SDPA(d_model, nhead, use_rope=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.has_cross_attention = cross_attention
        if cross_attention:
            self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.act = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "snake": Snake1d(dim_feedforward),
        }[activation]
        self.film = film
        if self.film:
            self.film_proj = nn.Linear(d_model, 2 * d_model)
            nn.init.zeros_(self.film_proj.weight)
            nn.init.zeros_(self.film_proj.bias)

        self.conformer_style = conformer_style
        if self.conformer_style:
            self.norm_mac = nn.LayerNorm(d_model)
            self.norm_conv = nn.LayerNorm(d_model)
            self.linear_mac1 = nn.Linear(d_model, dim_feedforward)
            self.linear_mac2 = nn.Linear(dim_feedforward, d_model)
            self.conv_pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
            self.conv_dw = nn.Conv1d(
                d_model, d_model, kernel_size=9, padding=4, groups=d_model
            )
            self.conv_pw2 = nn.Conv1d(d_model, d_model, 1)
            self.bn = nn.BatchNorm1d(d_model)

    def _self_attention(self, x, mask, pad):
        x = self.self_attn(x, x, x, attn_mask=mask, key_padding_mask=pad)
        return self.dropout(x)

    def _cross_attention(self, x, m, mask, pad):
        x = self.cross_attn(x, m, m, attn_mask=mask, key_padding_mask=pad)
        return self.dropout(x)

    def _ff(self, x):
        x = self.linear2(self.dropout(self.act(self.linear1(x))))
        return self.dropout(x)

    def _ff_mac(self, x):
        x = self.linear_mac2(self.dropout(self.act(self.linear_mac1(x))))
        return self.dropout(x)

    def _conv(self, x):
        y = x.transpose(1, 2)
        y = self.conv_pw1(y)
        y = F.glu(y, dim=1)
        y = self.conv_dw(y)
        y = self.bn(y)
        y = F.silu(y)
        y = self.conv_pw2(y)
        return self.dropout(y.transpose(1, 2))

    def forward(
        self,
        x,
        stage_emb=None,
        mem=None,
        src_mask=None,
        src_pad=None,
        mem_mask=None,
        mem_pad=None,
    ):
        if (stage_emb is not None) and self.film:
            g, b = self.film_proj(stage_emb).chunk(2, -1)
            x = (1 + g.unsqueeze(1)) * x + b.unsqueeze(1)

        if self.norm_first:
            if self.conformer_style:
                x = x + 0.5 * self._ff_mac(self.norm_mac(x))
            x = x + self._self_attention(self.norm1(x), src_mask, src_pad)
            if (mem is not None) and self.has_cross_attention:
                x = x + self._cross_attention(
                    self.norm2(x), self.norm4(mem), mem_mask, mem_pad
                )
            if self.conformer_style:
                x = x + self._conv(self.norm_conv(x))
                x = x + 0.5 * self._ff(self.norm3(x))
            else:
                x = x + self._ff(self.norm3(x))
        else:
            y = self.norm1(x + self._self_attention(x, src_mask, src_pad))
            if (mem is not None) and self.has_cross_attention:
                y = self.norm2(
                    y + self._cross_attention(y, mem, mem_mask, mem_pad)
                )
            if self.conformer_style:
                y = y + 0.5 * self._ff_mac(y)
                y = y + self._conv(self.norm_conv(y))
                x = self.norm3(y + 0.5 * self._ff(y))
            else:
                x = self.norm3(y + self._ff(y))
        return x


class CrossTransformerEncoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(layers[0].linear2.out_features)

    def forward(self, x, stage_emb=None, mem=None, mask=None, src_pad=None, mem_mask=None, mem_pad=None):
        for l in self.layers:
            x = l(x, stage_emb, mem, mask, src_pad, mem_mask, mem_pad)
        return x


# class CrossTransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward, activation="relu", batch_first=True, norm_first=True, cross_attention=True):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first)
#         if cross_attention:
#             self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.norm_first = norm_first
#         self.norm1 = nn.LayerNorm(d_model)
#         if cross_attention:
#             self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(0.1)
#         self.act = {"relu": nn.ReLU(), "gelu": nn.GELU(), "snake":Snake1d(dim_feedforward)}[activation]

#     def _sa(self, x, mask, pad):
#         x, _ = self.self_attn(x, x, x, attn_mask=mask, key_padding_mask=pad)
#         return self.dropout(x)

#     def _ca(self, x, m, mask, pad):
#         x, _ = self.cross_attn(x, m, m, attn_mask=mask, key_padding_mask=pad)
#         return self.dropout(x)

#     def _ff(self, x):
#         x = self.linear2(self.dropout(self.act(self.linear1(x))))
#         return self.dropout(x)

#     def forward(self, x, mem=None, src_mask=None, src_pad=None, mem_mask=None, mem_pad=None):
#         if self.norm_first:
#             x = x + self._sa(self.norm1(x), src_mask, src_pad)
#             if mem is not None:
#                 x = x + self._ca(self.norm2(x), mem, mem_mask, mem_pad)
#             x = x + self._ff(self.norm3(x))
#         else:
#             y = self.norm1(x + self._sa(x, src_mask, src_pad))
#             if mem is not None:
#                 y = self.norm2(y + self._ca(y, mem, mem_mask, mem_pad))
#             x = self.norm3(y + self._ff(y))
#         return x


# class CrossTransformerEncoder(nn.Module):
#     def __init__(self, layer, num_layers):
#         super().__init__()
#         self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

#     def forward(self, x, mem=None, mask=None, src_pad=None, mem_mask=None, mem_pad=None):
#         for l in self.layers:
#             x = l(x, mem, mask, src_pad, mem_mask, mem_pad)
#         return x


# Scripting this brings model speed up 1.4x
@torch.compile()
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, channels))

    def forward(self, x):
        return snake(x, self.alpha)

USE_SPEAKER_EMBEDDER = False

class TransformerEncoderLatent(nn.Module):

    def __init__(
        self,
        quantizer,
        vocab_size=2048,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        hidden_dim=2048,
        num_codebooks=8,
        use_speaker_embed=False,  # how many codebook heads
        after_quant=False,
        descript=False,
        twin_tower=False,
        film=False,
        sample_rate=24000,
        conformer_style=False,

    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.quantizer = quantizer

        self.sample_rate = sample_rate

        self.film = film

        hidden_size = 512 if not descript else 1024
        # self.quantizer.quantizer.semantic_residual_vector_quantizer.output_proj = None
        # self.quantizer.quantizer.acoustic_residual_vector_quantizer.output_proj = None
        # for p in self.quantizer.quantizer.parameters():
        #     p.requires_grad = True

        self.after_quant = after_quant


        self.descript = descript
        

        self.ln = nn.LayerNorm(embed_dim)

        # Positional encoding
        # self.pos_embedding = PositionalEncoding(embed_dim)

        self.ch_down = nn.Linear(hidden_size, embed_dim)

        # s0 = Snake1d(hidden_dim)

        self.s1 = Snake1d(hidden_size)
        self.s2 = Snake1d(hidden_size)
        #'gelu'
        encoder_layers = nn.ModuleList([
            CrossTransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=hidden_dim, 
                activation='gelu', 
                batch_first=True, 
                norm_first=True,
                cross_attention=True,
                film=True,
                conformer_style=conformer_style,
            ) for i in range(8)]#num_layers if not twin_tower else num_layers // 2)]
        )
        #nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_layers, dim_feedforward=hidden_dim, activation=s0, batch_first=True, norm_first=True,)
        # self.encoder = nn.TransformerEncoder(
        #     encoder_layer, num_layers=num_layers
        # )
        self.encoder = CrossTransformerEncoder(
            encoder_layers, 
        )
        print(self.encoder)
        if twin_tower:
            encoder_layers = nn.ModuleList([
                CrossTransformerEncoderLayer(
                    d_model=embed_dim, 
                    nhead=num_heads, 
                    dim_feedforward=hidden_dim, 
                    activation='gelu', 
                    batch_first=True, 
                    norm_first=True,
                    cross_attention=False,
                    film=False,
                    conformer_style=conformer_style,
                ) for _ in range(8) #range(num_layers // 2)
            ])
            # encoder_layer = CrossTransformerEncoderLayer(
            #     d_model=embed_dim, 
            #     nhead=num_heads, 
            #     dim_feedforward=hidden_dim, 
            #     activation='snake', 
            #     batch_first=True, 
            #     norm_first=True,
            #     cross_attention=False,
            #     film=False,
            # )
            self.speaker_encoder = CrossTransformerEncoder(
                encoder_layers,
            )

        self.pooling = SelfAttnPooling(embed_dim) #SelfAttnPooling(embed_dim)


        self.twin_tower = twin_tower

        self.ch_up = nn.Linear(embed_dim, hidden_size)

        # Separate classification heads for each codebook
        self.masker = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=True), nn.Sigmoid())
        print(self.gate)
        self.gate[0].bias.data.fill_(-5.0)
        self.output = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=True), self.s1) #nn.GELU()
        self.activation = self.s2 #nn.ReLU() #nn.Tanh()

        self.speaker_clf_head = nn.Linear(embed_dim, 300)


        # for p in self.quantizer.parameters():
        #     p.requires_grad = False

        for p in self.quantizer.parameters():
            p.requires_grad = False

        # for p in self.quantizer.decoder.parameters():
        #     p.requires_grad = True

        # nn.init.normal_(self.classification_head.weight, 0, 0.1)

    def get_latent_pre_quant(self, x):
        """
        only for after_quant: false
        """
        x = self.quantizer.encoder(x)
        x = self.quantizer.encoder_transformer(
            x.transpose(1, 2)
        )[0].transpose(1, 2)
        x = self.quantizer.downsample(x)
        return x.transpose(1,2)

    def quantize_and_decode(self, x):
        """
        only for after_quant: false
        """
        feat = self.quantizer.quantizer.encode(x, num_quantizers=8).transpose(0,1)
        feat = self.quantizer.quantizer.decode(feat)
        # feat.grad = x.grad
        # feat = x + (self.quantizer.quantizer.decode(feat) - x).detach()
        x = self.decode_latent(feat)
        # x = self.quantizer.decode(x).audio_values.squeeze()
        return x 

    def get_latent(self, x):
        """
        only for after_quant: true
        """
        codes = self.quantizer.encode(x).audio_codes
        emb_current_ = self.quantizer.quantizer.decode(codes)
        return emb_current_.transpose(1, 2)

    def decode_latent(self, x):
        """
        only for after_quant: true
        """
        x = self.quantizer.upsample(x)
        decoder_outputs = self.quantizer.decoder_transformer(x.transpose(1, 2))
        x = decoder_outputs[0].transpose(1, 2)
        x = self.quantizer.decoder(x).squeeze()
        return x

    # @torch.compile()
    def descript_encode(self, x):
        x = self.quantizer.preprocess(x, self.sample_rate)
        #z, codes, latents, _, _ = self.quantizer.encode(x, n_quantizers=None)
        z = self.quantizer.encoder(x)
        return z
    
    # @torch.compile()
    def descript_decode(self, x):
        x, codes_hat, latents_hat, commitment_loss_hat, codebook_loss_hat = self.quantizer.quantizer(x, None)
        x = self.quantizer.decoder(x).squeeze()
        return x #self.quantizer.decode(x).squeeze()

    # @torch.compile()
    def forward(
        self,
        mix,
        speaker,
        speaker_label=None,
    ):

        # codes = self.quantizer.encode(mix).audio_codes
        # emb_current_ = self.quantizer.quantizer.decode(codes)

        # emb_current_ = self.quantizer.upsample(emb_current_)
        # decoder_outputs = self.quantizer.decoder_transformer(emb_current_.transpose(1, 2))
        # emb_current = decoder_outputs[0]
        # codes = self.quantizer.encode(speaker).audio_codes
        # speaker = self.quantizer.quantizer.decode(codes)

        # speaker = self.quantizer.upsample(speaker)
        # decoder_outputs = self.quantizer.decoder_transformer(speaker.transpose(1, 2))
        # speaker = decoder_outputs[0]
        # speaker_emb = self.proj[0](speaker.transpose(1, 2))

            
        # speaker_emb = self.quantizer.encoder(speaker)
        # speaker_emb = self.quantizer.encoder_transformer(
        #     speaker_emb.transpose(1, 2)
        # )[0].transpose(1, 2)
        # speaker_emb = self.quantizer.downsample(speaker_emb)
        # speaker_emb = self.proj[0](speaker_emb.transpose(1,2))


        # emb_current_ = self.quantizer.encoder(mix)
        # emb_current_ = self.quantizer.encoder_transformer(
        #     emb_current_.transpose(1, 2)
        # )[0].transpose(1, 2)
        # emb_current_ = self.quantizer.downsample(emb_current_)
        # emb_current = self.proj[0](emb_current_.transpose(1,2))
        # emb_current_ = emb_current_.transpose(1,2)


        if self.descript:
            emb_current_ = self.descript_encode(mix).transpose(1,2)
            # print(emb_current_.shape)
            emb_current = self.ch_down(emb_current_)
            speaker_emb = self.ch_down(self.descript_encode(speaker).transpose(1,2))

        else:
            if self.after_quant:
                emb_current_ = self.get_latent(mix)
                emb_current = self.ch_down(emb_current_)
                speaker_emb = self.ch_down(self.get_latent(speaker))
            else:
                emb_current_ = self.get_latent_pre_quant(mix)
                emb_current = self.ch_down(emb_current_)
                speaker_emb = self.ch_down(self.get_latent_pre_quant(speaker))


        # emb_input = torch.cat(
        #     [
        #         self.pos_embedding(speaker_emb),
        #         self.pos_embedding(emb_current),
        #     ],
        #     dim=1,
        # )

        # speaker_emb = self.pos_embedding(speaker_emb)
        speaker_emb_ =speaker_emb
        if self.twin_tower:
            speaker_emb = self.speaker_encoder(speaker_emb)

        stage_emb = None
        if self.film:
            # attention pooling here
            stage_emb = self.pooling(speaker_emb)

        speaker_label_pred = None

        # if speaker_label is not None:
        #     speaker_label_pred = self.speaker_clf_head(stage_emb)

        # self.pos_embedding
        encoded = self.encoder(x=torch.cat([speaker_emb_, emb_current],dim=1), mem=speaker_emb, stage_emb=stage_emb)[:, -emb_current.size(1) :]
        #encoded = self.encoder(x=emb_current, mem=speaker_emb, stage_emb=stage_emb)
        encoded = self.ch_up(encoded)

        mask = self.masker(encoded)  # [B, T, 2048]

        gate = self.gate(mask)
        out = self.output(mask)

        out =  self.activation(out*gate) 

        out = (1 + out) * emb_current_

        return out.transpose(1,2), speaker_label_pred



class FiLMWrapper(nn.Module):
    """ """

    def __init__(self, base_layer: nn.TransformerEncoderLayer, d_model: int):
        super().__init__()
        self.base_layer = base_layer
        # tiny 1‑layer projection:  D  →  2 D   (γ and β)
        self.film_proj = nn.Linear(d_model, 2 * d_model)

        # init γ ≈ 0, β ≈ 0 so the wrapper is identity at start‑up
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def forward(
        self, x: torch.Tensor, stage_emb: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        x:          [B, T, D]
        stage_emb:  [B, D]     (one vector per sample)
        kwargs:     any extra args you usually pass to the base layer
        """
        gamma_beta = self.film_proj(stage_emb)  # [B, 2D]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each [B, D]

        # broadcast over time dimension
        x = (1.0 + gamma.unsqueeze(1)) * x + beta.unsqueeze(1)
        return self.base_layer(x, **kwargs)


class FiLMTransformerEncoder(nn.Module):
    """
    Drop‑in replacement for nn.TransformerEncoder that threads the same
    stage‑embedding through every FiLM layer.
    """

    def __init__(
        self, num_layers: int, embed_dim: int, num_heads: int, hidden_dim: int
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FiLMWrapper(
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dim_feedforward=hidden_dim,
                        batch_first=True,
                        norm_first=True,
                        activation="gelu",
                    ),
                    d_model=embed_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, stage_emb: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, stage_emb, *args, **kwargs)
        return self.norm(x)


class TransformerEncoderWithCodebookCondition(nn.Module):
    """
    A simple Transformer model that, at each codebook j, sums
    (1) The embedding of codebook j's tokens
    (2) The embeddings of all past codebooks 1..(j-1), i.e. 'condition'
    Then passes the sum through a Transformer encoder and a classification head.

    Each codebook j has its own classification head.
    """

    def __init__(
        self,
        quantizer,
        vocab_size=2048,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        hidden_dim=2048,
        num_codebooks=8,
        use_speaker_embed=False,  # how many codebook heads
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.embed_dim = embed_dim
        self.quantizer = quantizer
        # self.quantizer.quantizer.semantic_residual_vector_quantizer.output_proj = None
        # self.quantizer.quantizer.acoustic_residual_vector_quantizer.output_proj = None
        for p in self.quantizer.parameters():
            p.requires_grad = False

        for p in self.quantizer.quantizer.parameters():
            p.requires_grad = False

        self.ln = nn.LayerNorm(embed_dim)
        # self.add_tokens = nn.Embedding(3,embed_dim)

        # Shared token embedding for all codebooks
        # self.proj_acoustic = nn.Linear(512, embed_dim)
        # self.proj_semantic = nn.Linear(512, embed_dim)
        self.proj = nn.ModuleList(
            [nn.Linear(512, embed_dim) for _ in range(3)]
        )  # nn.Linear(512, embed_dim)
        # self.codebooks = nn.ModuleList([nn.Embedding(vocab_size, embed_dim) for _ in range(num_codebooks)])

        self.stage_embeddings = nn.Embedding(num_codebooks, embed_dim)

        # Positional encoding
        self.pos_embedding = PositionalEncoding(embed_dim)

        self.use_speaker_embed = use_speaker_embed

        if self.use_speaker_embed:
            self.speaker_proj = nn.Linear(192, self.embed_dim)

        # ─── inside __init__ ─────────────────────────────────────────────────────────
        self.film_encoder = FiLMTransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_layers,
            hidden_dim=hidden_dim,
        )

        # Separate classification heads for each codebook
        self.classification_heads = nn.ModuleList(
            [nn.Linear(embed_dim, vocab_size, bias=False) for _ in range(num_codebooks)]
        )

    def forward(
        self,
        current_tokens,
        codebook_idx,
        speaker_emb=None,
        past_codebook_tokens=None,
        speaker_codes=None,
    ):
        """
        Args:
            current_tokens: Tensor of shape [B, T] with the token IDs for codebook j. (From PROMPT)
            codebook_idx: int, which codebook's classification head to use.
            past_codebook_tokens: None or a list/tuple of Tensors, each [B, T], for codebooks < j
                                  that we want to sum as condition. (From NAR)

        Returns:
            logits: [B, T, num_classes_per_head] for the codebook_idx-th head.

        """
        if not self.use_speaker_embed:
            speaker_codes_emb = self.proj[0](
                self.quantizer.quantizer.decode(speaker_codes).transpose(1, 2)
            )

        emb_current = self.proj[1](
            self.quantizer.quantizer.decode(current_tokens).transpose(1, 2)
        )

        # embd_stage = self.stage_embeddings(torch.Tensor([codebook_idx]).long().to(emb_current.device)).repeat(emb_current.size(0), 1).unsqueeze(1)

        if past_codebook_tokens is not None:
            sum_of_past = self.proj[2](
                self.quantizer.quantizer.decode(past_codebook_tokens).transpose(1, 2)
            )
        else:
            sum_of_past = torch.zeros_like(emb_current)

        if self.use_speaker_embed:
            speaker_emb = self.speaker_proj(speaker_emb)

        # 3) Combine them
        spkr_info = (
            self.pos_embedding(speaker_codes_emb)
            if not self.use_speaker_embed
            else speaker_emb
        )
        emb_input = torch.cat(
            [
                spkr_info,
                self.pos_embedding(emb_current),
                self.pos_embedding(sum_of_past),
            ],
            dim=1,
        )

        stage_emb = (
            self.stage_embeddings(
                torch.tensor(codebook_idx, device=current_tokens.device)
            )
            .unsqueeze(0)
            .expand(current_tokens.size(0), -1)
        )  # [B, D]

        encoded = self.film_encoder(emb_input, stage_emb)[:, -emb_current.size(1) :]

        # encoded = self.transformer_encoder(emb_input)[:, -emb_current.size(1):]  # [B, T, d_model]

        # 6) Classification head for codebook j
        logits = self.classification_heads[codebook_idx](encoded)  # [B, T, 2048]
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        self.d_model = d_model
        # Create learnable positional embeddings
        self.pe = nn.Embedding(max_len, self.d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)  # Get the sequence length from input tensor

        # Create a positional index tensor
        positions = torch.arange(seq_len, device=x.device).unsqueeze(
            0
        )  # Shape: [1, seq_len]
        positions = positions.expand(x.size(0), seq_len)  # Expand to match batch size

        # Look up the positional embeddings for the sequence
        pos_emb = self.pe(positions)  # Shape: [batch_size, seq_len, d_model]

        # Add positional embeddings to the input tensor
        return x + pos_emb
