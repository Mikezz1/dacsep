import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch


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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.quantizer = quantizer
        # self.quantizer.quantizer.semantic_residual_vector_quantizer.output_proj = None
        # self.quantizer.quantizer.acoustic_residual_vector_quantizer.output_proj = None
        # for p in self.quantizer.parameters():
        #     p.requires_grad = False

        # for p in self.quantizer.quantizer.parameters():
        #     p.requires_grad = True

        self.ln = nn.LayerNorm(embed_dim)

        # Positional encoding
        self.pos_embedding = PositionalEncoding(embed_dim)

        self.proj = nn.ModuleList(
            [nn.Linear(512, embed_dim) for _ in range(3)]
        )  # nn.Linear(512, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_layers, dim_feedforward=hidden_dim, activation='gelu', batch_first=True, norm_first=True,)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Separate classification heads for each codebook
        self.classification_head = nn.Sequential(nn.Linear(embed_dim, 512, bias=False), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(embed_dim, 512, bias=False), nn.Sigmoid())
        # nn.init.normal_(self.classification_head.weight, 0, 0.1)


    def forward(
        self,
        mix,
        speaker,
    ):
    
        speaker_emb = self.quantizer.encoder(speaker)
        speaker_emb = self.quantizer.encoder_transformer(
            speaker_emb.transpose(1, 2)
        )[0].transpose(1, 2)
        speaker_emb = self.quantizer.downsample(speaker_emb)
        speaker_emb = self.proj[0](speaker_emb.transpose(1,2))


        emb_current_ = self.quantizer.encoder(mix)
        emb_current_ = self.quantizer.encoder_transformer(
            emb_current_.transpose(1, 2)
        )[0].transpose(1, 2)
        emb_current_ = self.quantizer.downsample(emb_current_)
        emb_current = self.proj[0](emb_current_.transpose(1,2))
        emb_current_ = emb_current_.transpose(1,2)

        emb_input = torch.cat(
            [
                self.pos_embedding(speaker_emb),
                self.pos_embedding(emb_current),
            ],
            dim=1,
        )


        encoded = self.encoder(emb_input)[:, -emb_current.size(1) :]

        logits = self.classification_head(encoded)  # [B, T, 2048]
        gate = self.gate(encoded)

        out =  F.gelu(logits*gate) * emb_current_

        return out.transpose(1,2)



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
