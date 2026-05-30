"""Pretrained foundation EEG backbone wrappers.

Each backbone:
  - Accepts input (batch, C, T) — raw EEG epochs
  - Exposes get_features(X) → (batch, feature_dim) — frozen encoder output
  - Is loaded with pretrained weights via checkpoint_path

The architecture stubs below match the published designs as closely as
possible. Where the official implementation is publicly available, replace
the stub encoder with the real one and load the official checkpoint.

To add a new foundation model:
  1. Subclass FoundationBackbone
  2. Implement get_features() and feature_dim
  3. Add to FOUNDATION_REGISTRY
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional


class FoundationBackbone(nn.Module, ABC):
    """Common interface for pretrained EEG foundation models."""

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensionality of the feature vector returned by get_features()."""

    @abstractmethod
    def get_features(self, X: torch.Tensor) -> torch.Tensor:
        """Extract features from raw EEG.

        Args:
            X: (B, C, T) float32 tensor
        Returns:
            (B, feature_dim) float32 tensor
        """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.get_features(X)

    def freeze(self) -> "FoundationBackbone":
        for p in self.parameters():
            p.requires_grad_(False)
        return self


# ---------------------------------------------------------------------------
# MIRepNet
# ---------------------------------------------------------------------------

class _MIRepNetAttention(nn.Module):
    """Custom attention matching checkpoint keys: keys/queries/values/projection."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.keys       = nn.Linear(dim, dim)
        self.queries    = nn.Linear(dim, dim)
        self.values     = nn.Linear(dim, dim)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.shape[-1] ** -0.5
        attn = torch.softmax(
            self.queries(x) @ self.keys(x).transpose(-2, -1) * scale, dim=-1
        )
        return self.projection(attn @ self.values(x))


class _MIRepNetPreNorm(nn.Module):
    """PreNorm residual wrapper. Keys: fn.0 = LayerNorm, fn.1 = sublayer."""

    def __init__(self, norm: nn.Module, sublayer: nn.Module):
        super().__init__()
        self.fn = nn.ModuleList([norm, sublayer])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn[1](self.fn[0](x))


class _MIRepNetEmbedding(nn.Module):
    """Conv embedding block.

    Checkpoint keys:
        embedding.conv1              Conv2d(1, 64, (1, 25))
        embedding.conv2              Conv2d(64, 128, (45, 1))
        embedding.bn                 BatchNorm2d(128)
        embedding.projection.0       Conv2d(128, 256, (1, 1))
        embedding.chan_embed          Embedding(45, 256)   [unused at inference]

    Pretrained on 45-channel data. Inputs with fewer channels are zero-padded.
    """

    N_PRETRAIN_CH = 45

    def __init__(self):
        super().__init__()
        C = self.N_PRETRAIN_CH
        self.conv1      = nn.Conv2d(1, 64, (1, 25))
        self.conv2      = nn.Conv2d(64, 128, (C, 1))
        self.bn         = nn.BatchNorm2d(128)
        self.pool       = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout    = nn.Dropout(0.5)
        self.projection = nn.Sequential(nn.Conv2d(128, 256, (1, 1)))
        self.chan_embed  = nn.Embedding(C, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, T = x.shape
        C = self.N_PRETRAIN_CH
        if C_in < C:
            x = torch.cat([x, x.new_zeros(B, C - C_in, T)], dim=1)
        x = x.unsqueeze(1)                              # (B, 1, C, T)
        x = F.elu(self.conv1(x))                        # (B, 64, C, T-24)
        x = F.elu(self.bn(self.conv2(x)))               # (B, 128, 1, T-24)
        x = self.pool(x)                                # (B, 128, 1, pooled_T)
        x = self.dropout(x)
        x = self.projection(x)                          # (B, 256, 1, pooled_T)
        return x.squeeze(2).transpose(1, 2)             # (B, pooled_T, 256)


# Channel map: BCIC-IV 2a (22 ch) → MIRepNet 45-channel template position.
# Template order: F7,F5,F3,F1,FZ,F2,F4,F6,F8, FT7,FC5,FC3,FC1,FCZ,FC2,FC4,FC6,FT8,
#                 T7,C5,C3,C1,CZ,C2,C4,C6,T8, TP7,CP5,CP3,CP1,CPZ,CP2,CP4,CP6,TP8,
#                 P7,P5,P3,P1,PZ,P2,P4,P6,P8
# BCIC-IV 2a order: Fz,FC3,FC1,FCz,FC2,FC4,C5,C3,C1,Cz,C2,C4,C6,CP3,CP1,CPz,CP2,CP4,P1,Pz,P2,POz
_BCICIV2A_TO_MIREPNET45: list[int] = [
    4, 11, 12, 13, 14, 15,          # Fz  FC3 FC1 FCz FC2 FC4
    19, 20, 21, 22, 23, 24, 25,     # C5  C3  C1  Cz  C2  C4  C6
    29, 30, 31, 32, 33,             # CP3 CP1 CPz CP2 CP4
    39, 40, 41,                     # P1  Pz  P2
    -1,                             # POz — not in template, dropped
]


class MIRepNetBackbone(FoundationBackbone):
    """MIRepNet: masked EEG representation network (starself/MIRepNet on HuggingFace).

    Pretrained weights:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="starself/MIRepNet", filename="MIRepNet.pth")
    Upload to Modal:
        modal volume put eeg-data ./MIRepNet.pth /MIRepNet.pth

    Pretrained at 250 Hz on 45-channel data with EA normalization.
    For BCIC-IV 2a (22 ch, 200 Hz): get_features() resamples to 250 Hz and
    places channels at their correct positions in the 45-ch template.
    Loads embedding.* and transformer.*; skips clshead and decoder.
    """

    _EMB_DIM     = 256
    _N_LAYERS    = 6
    _FFN_DIM     = 1024
    _TARGET_SFREQ = 250.0

    def __init__(self, n_channels: int, n_times: int,
                 input_sfreq: float = 200.0,
                 channel_map: Optional[list] = None,
                 checkpoint_path: Optional[str] = None, **kwargs):
        super().__init__()
        self.input_sfreq = input_sfreq
        # channel_map[i] = position in 45-ch template for input channel i; -1 = drop
        if channel_map is not None:
            self.channel_map = channel_map
        elif n_channels == len(_BCICIV2A_TO_MIREPNET45):
            self.channel_map = _BCICIV2A_TO_MIREPNET45
        elif n_channels == _MIRepNetEmbedding.N_PRETRAIN_CH:
            self.channel_map = list(range(_MIRepNetEmbedding.N_PRETRAIN_CH))
        else:
            self.channel_map = None
        if self.channel_map is None:
            raise ValueError(
                "MIRepNet pretrained weights require either the 22-channel "
                "BCIC-IV-2a layout, the full 45-channel template, or an explicit "
                f"channel_map; got n_channels={n_channels}."
            )

        D = self._EMB_DIM
        self.embedding = _MIRepNetEmbedding()
        self.transformer = nn.ModuleList([
            nn.ModuleList([
                _MIRepNetPreNorm(nn.LayerNorm(D), _MIRepNetAttention(D)),
                _MIRepNetPreNorm(nn.LayerNorm(D), nn.Sequential(
                    nn.Linear(D, self._FFN_DIM),   # .0
                    nn.GELU(),                      # .1 (no params)
                    nn.Dropout(0.1),                # .2 (no params)
                    nn.Linear(self._FFN_DIM, D),    # .3
                )),
            ])
            for _ in range(self._N_LAYERS)
        ])

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            relevant = {k: v for k, v in state.items()
                        if k.startswith("embedding.") or k.startswith("transformer.")}
            missing, _ = self.load_state_dict(relevant, strict=False)
            if missing:
                print(f"[MIRepNet] {len(missing)} keys not loaded: "
                      f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

    @property
    def feature_dim(self) -> int:
        return self._EMB_DIM

    def get_features(self, X: torch.Tensor) -> torch.Tensor:
        B, _, T = X.shape
        if self.channel_map is None or len(self.channel_map) != X.shape[1]:
            raise ValueError(
                "MIRepNet requires an explicit channel_map matching the input "
                f"channel count. Got {X.shape[1]} channels but channel_map has "
                f"{0 if self.channel_map is None else len(self.channel_map)} entries."
            )

        # 1. Resample to 250 Hz
        if self.input_sfreq != self._TARGET_SFREQ:
            target_len = int(T * self._TARGET_SFREQ / self.input_sfreq)
            X = F.interpolate(X, size=target_len, mode="linear", align_corners=False)

        # 2. Place channels into the 45-channel template at correct positions
        X45 = X.new_zeros(B, _MIRepNetEmbedding.N_PRETRAIN_CH, X.shape[-1])
        for src, dst in enumerate(self.channel_map):
            if dst >= 0:
                X45[:, dst, :] = X[:, src, :]

        # 3. Per-channel z-score (approximates EA whitening used during pretraining)
        mu  = X45.mean(dim=-1, keepdim=True)
        std = X45.std(dim=-1, keepdim=True)
        X45 = (X45 - mu) / (std + 1e-8)

        # 4. Embedding → transformer → mean pool
        x = self.embedding(X45)                # (B, T', 256)
        for attn_block, ffn_block in self.transformer:
            x = attn_block(x)
            x = ffn_block(x)
        return x.mean(dim=1)                   # (B, 256)


# ---------------------------------------------------------------------------
# NeuroGPT
# ---------------------------------------------------------------------------

class _NeuroGPTEncoderBlock(nn.Module):
    """EEGConformer conv encoder matching NeuroGPT checkpoint keys.

    Checkpoint key structure:
        encoder.patch_embedding.shallownet.{0,1,2}  Conv2d, Conv2d, BatchNorm2d
        encoder.patch_embedding.projection.{0}       Conv2d 1×1

    The transformer is the GPT decoder (decoder.*) — not part of the encoder.
    Fixed to 22 channels and 250 Hz (500-sample chunks).
    Output: (B, 27, 40) token sequence — flattened to 1080-dim before embedder.
    """

    N_FILTERS   = 40
    FILTER_TIME = 25
    POOL_TIME   = 75
    POOL_STRIDE = 15
    # n_tokens = ((500 - 25 + 1 - 75) // 15 + 1) = 27
    N_TOKENS    = 27

    def __init__(self, n_channels: int = 22, dropout: float = 0.5):
        super().__init__()
        F = self.N_FILTERS
        self.patch_embedding = nn.ModuleDict({
            "shallownet": nn.Sequential(
                nn.Conv2d(1, F, (1, self.FILTER_TIME), bias=False),          # .0
                nn.Conv2d(F, F, (n_channels, 1), bias=False),                # .1
                nn.BatchNorm2d(F),                                            # .2
                nn.ELU(),
                nn.AvgPool2d((1, self.POOL_TIME), stride=(1, self.POOL_STRIDE)),
                nn.Dropout(dropout),
            ),
            "projection": nn.Sequential(
                nn.Conv2d(F, F, (1, 1)),                                      # .0
            ),
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) — already 500 samples at 250 Hz
        x = x.unsqueeze(1)                                      # (B, 1, C, T)
        x = self.patch_embedding["shallownet"](x)               # (B, 40, 1, 27)
        x = self.patch_embedding["projection"](x)               # (B, 40, 1, 27)
        return x.squeeze(2).transpose(1, 2)                     # (B, 27, 40)


class _NeuroGPTEmbedder(nn.Module):
    """Linear embedder matching NeuroGPT checkpoint keys.

    Checkpoint key structure:
        embedder.embed_model.model.{0}   Linear(1080, 768)
        embedder.embed_model.model.{1}   LayerNorm(768)
    """

    FLAT_DIM = _NeuroGPTEncoderBlock.N_TOKENS * _NeuroGPTEncoderBlock.N_FILTERS  # 1080
    EMB_DIM  = 1024

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.embed_model = nn.ModuleDict({
            "model": nn.Sequential(
                nn.Linear(self.FLAT_DIM, self.EMB_DIM),   # .0  Linear(1080, 1024)
                nn.LayerNorm(self.EMB_DIM),                # .1
                nn.GELU(),
                nn.Dropout(dropout),
            )
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 27, 40) → flatten → embed
        return self.embed_model["model"](x.flatten(1))    # (B, 768)


class NeuroGPTBackbone(FoundationBackbone):
    """NeuroGPT: EEGConformer encoder + linear embedder (Cui et al. 2023).

    Pretrained weights: https://huggingface.co/wenhuic/Neuro-GPT/tree/main
    Upload to Modal:
        modal volume put eeg-data ./neuro_gpt.pt /neuro_gpt.pt
    Set in modal_runner.py:
        CHECKPOINT_PATH = "/data/neuro_gpt.pt"

    Input requirements:
        - 22 EEG channels
        - Any sampling rate (internally resampled to 250 Hz)
        - Any epoch length ≥ 2 s (truncated to first 500 samples at 250 Hz)

    Checkpoint loads encoder.* and embedder.* keys; decoder.* (GPT) is skipped.
    """

    _EMB_DIM   = _NeuroGPTEmbedder.EMB_DIM     # 768
    _CHUNK_LEN = 500                             # samples at 250 Hz (2 s)
    _TARGET_SFREQ = 250.0

    def __init__(self, n_channels: int, n_times: int,
                 input_sfreq: float = 200.0,
                 checkpoint_path: Optional[str] = None, **kwargs):
        super().__init__()
        if checkpoint_path is not None and n_channels != 22:
            raise ValueError(
                "NeuroGPT pretrained encoder weights are tied to 22-channel "
                f"inputs; got n_channels={n_channels}."
            )
        self.input_sfreq = input_sfreq
        # Named to match top-level checkpoint keys: encoder.* and embedder.*
        self.encoder  = _NeuroGPTEncoderBlock(n_channels=n_channels)
        self.embedder = _NeuroGPTEmbedder()

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            # Load only encoder + embedder portions; skip GPT decoder
            relevant = {k: v for k, v in state.items()
                        if k.startswith("encoder.") or k.startswith("embedder.")}
            missing, unexpected = self.load_state_dict(relevant, strict=False)
            if missing:
                print(f"[NeuroGPT] {len(missing)} keys not loaded (architecture mismatch?): "
                      f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

    @property
    def feature_dim(self) -> int:
        return self._EMB_DIM

    def get_features(self, X: torch.Tensor) -> torch.Tensor:
        """Extract 1024-dim features from (B, 22, T) EEG input."""
        # 1. Resample to 250 Hz if input is at a different rate.
        if self.input_sfreq != self._TARGET_SFREQ:
            target_len = int(X.shape[-1] * self._TARGET_SFREQ / self.input_sfreq)
            X = F.interpolate(X, size=target_len, mode="linear", align_corners=False)

        # 2. Per-channel z-score normalisation at the model's input sampling rate.
        mu  = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True)
        X   = (X - mu) / (std + 1e-25)

        # 3. Chunk into 500-sample (2 s) windows and average features across chunks.
        #    NeuroGPT was pretrained on 2 s chunks; a 4 s trial has two valid chunks.
        #    Truncating to the first chunk would discard half the trial signal.
        T = X.shape[-1]
        n_chunks = max(1, T // self._CHUNK_LEN)
        chunk_feats = []
        for i in range(n_chunks):
            chunk = X[:, :, i * self._CHUNK_LEN : (i + 1) * self._CHUNK_LEN]
            if chunk.shape[-1] < self._CHUNK_LEN:
                break
            chunk_feats.append(self.embedder(self.encoder(chunk)))  # (B, 1024)
        return torch.stack(chunk_feats, dim=0).mean(dim=0)          # (B, 1024)


# ---------------------------------------------------------------------------
# REVE
# ---------------------------------------------------------------------------

class _REVEEncoder(nn.Module):
    """Convolutional stem + Transformer encoder approximating REVE.

    TODO: replace with official architecture and load the published checkpoint.
    """

    def __init__(self, n_channels: int, n_times: int, emb_dim: int = 256,
                 n_layers: int = 4, n_heads: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=15, padding=7),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(128, emb_dim, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)            # (B, emb_dim, T')
        x = x.transpose(1, 2)      # (B, T', emb_dim)
        x = self.transformer(x)
        return self.norm(x).mean(dim=1)  # mean pool over time


class REVEBackbone(FoundationBackbone):
    """REVE: Robust EEG Vision Encoder.

    TODO: replace _REVEEncoder with the official architecture and load
    the published pretrained checkpoint.
    """

    _EMB_DIM = 256

    def __init__(self, n_channels: int, n_times: int,
                 checkpoint_path: Optional[str] = None, **kwargs):
        super().__init__()
        self.encoder = _REVEEncoder(n_channels, n_times, self._EMB_DIM)
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            self.encoder.load_state_dict(state, strict=False)

    @property
    def feature_dim(self) -> int:
        return self._EMB_DIM

    def get_features(self, X: torch.Tensor) -> torch.Tensor:
        return self.encoder(X)


# ---------------------------------------------------------------------------
# CBraMod
# ---------------------------------------------------------------------------

class _CBraModPatchEmbedding(nn.Module):
    """Patch embedding block matching the official CBraMod checkpoint keys."""

    def __init__(self, in_dim: int = 200, d_model: int = 200):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=(19, 7),
                stride=(1, 1),
                padding=(9, 3),
                groups=d_model,
            )
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(in_dim // 2 + 1, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, C, P, 200), where P is the number of temporal patches.
        bz, ch_num, patch_num, patch_size = x.shape
        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous()
        patch_emb = patch_emb.view(bz, ch_num, patch_num, self.d_model)

        spectral = torch.fft.rfft(
            mask_x.contiguous().view(bz * ch_num * patch_num, patch_size),
            dim=-1,
            norm="forward",
        )
        spectral = torch.abs(spectral).contiguous().view(
            bz, ch_num, patch_num, patch_size // 2 + 1
        )
        patch_emb = patch_emb + self.spectral_proj(spectral)

        pos = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        return patch_emb + pos.permute(0, 2, 3, 1)


class _CBraModEncoderLayer(nn.Module):
    """Criss-cross transformer layer from the official CBraMod implementation."""

    def __init__(
        self,
        d_model: int = 200,
        nhead: int = 8,
        dim_feedforward: int = 800,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn_s = nn.MultiheadAttention(
            d_model // 2, nhead // 2, dropout=dropout, batch_first=True
        )
        self.self_attn_t = nn.MultiheadAttention(
            d_model // 2, nhead // 2, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        bz, ch_num, patch_num, d_model = x.shape
        xs = x[:, :, :, : d_model // 2]
        xt = x[:, :, :, d_model // 2 :]

        xs = xs.transpose(1, 2).contiguous().view(
            bz * patch_num, ch_num, d_model // 2
        )
        xt = xt.contiguous().view(bz * ch_num, patch_num, d_model // 2)

        xs = self.self_attn_s(xs, xs, xs, attn_mask=attn_mask, need_weights=False)[0]
        xt = self.self_attn_t(xt, xt, xt, attn_mask=attn_mask, need_weights=False)[0]

        xs = xs.contiguous().view(bz, patch_num, ch_num, d_model // 2).transpose(1, 2)
        xt = xt.contiguous().view(bz, ch_num, patch_num, d_model // 2)
        return self.dropout1(torch.cat((xs, xt), dim=3))

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        return self.dropout2(x)


class _CBraModEncoder(nn.Module):
    """Stack of criss-cross transformer layers matching checkpoint key names."""

    def __init__(
        self,
        n_layers: int = 12,
        d_model: int = 200,
        nhead: int = 8,
        dim_feedforward: int = 800,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            _CBraModEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            )
            for _ in range(n_layers)
        ])

    def forward(self, src: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask)
        return output


class _CBraModModel(nn.Module):
    """Official CBraMod architecture, kept local to avoid a runtime repo dependency."""

    def __init__(
        self,
        in_dim: int = 200,
        out_dim: int = 200,
        d_model: int = 200,
        dim_feedforward: int = 800,
        n_layer: int = 12,
        nhead: int = 8,
    ):
        super().__init__()
        self.patch_embedding = _CBraModPatchEmbedding(in_dim=in_dim, d_model=d_model)
        self.encoder = _CBraModEncoder(
            n_layers=n_layer,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.proj_out = nn.Sequential(nn.Linear(d_model, out_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.patch_embedding(x, mask)
        x = self.encoder(x)
        return self.proj_out(x)


class CBraModBackbone(FoundationBackbone):
    """CBraMod: Criss-Cross Brain Foundation Model (Wang et al. 2024).

    The official model consumes patched EEG shaped as (B, C, P, 200). This
    wrapper keeps the benchmark-facing API at (B, C, T), patches along time,
    and mean-pools CBraMod's per-channel, per-patch outputs to one feature
    vector per trial.
    """

    _EMB_DIM = 200
    _PATCH_SIZE = 200

    def __init__(self, n_channels: int, n_times: int,
                 checkpoint_path: Optional[str] = None, **kwargs):
        super().__init__()
        self.model = _CBraModModel(
            in_dim=self._PATCH_SIZE,
            out_dim=self._EMB_DIM,
            d_model=self._EMB_DIM,
            dim_feedforward=self._EMB_DIM * 4,
            n_layer=12,
            nhead=8,
        )
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            state = {
                k.removeprefix("module.").removeprefix("model."): v
                for k, v in state.items()
            }
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(
                    f"[CBraMod] loaded with {len(missing)} missing and "
                    f"{len(unexpected)} unexpected keys"
                )

    @property
    def feature_dim(self) -> int:
        return self._EMB_DIM

    def get_features(self, X: torch.Tensor) -> torch.Tensor:
        B, C, T = X.shape
        patch_size = self._PATCH_SIZE

        # CBraMod was pretrained on normalized 200-sample patches.
        mu = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True)
        X = (X - mu) / (std + 1e-8)

        n_patches = max(1, (T + patch_size - 1) // patch_size)
        target_len = n_patches * patch_size
        if T < target_len:
            X = F.pad(X, (0, target_len - T))
        elif T > target_len:
            X = X[:, :, :target_len]

        X = X.contiguous().view(B, C, n_patches, patch_size)
        return self.model(X).mean(dim=(1, 2))


# ---------------------------------------------------------------------------
# Classification head wrapper (for finetune / LoRA on foundation backbones)
# ---------------------------------------------------------------------------

class FoundationWithHead(nn.Module):
    """Foundation backbone + linear classification head.

    Foundation backbones expose get_features() → embeddings, not logits.
    This wrapper adds a trainable linear head so the combined model can be
    used with standard cross-entropy training (finetune, LoRA).

    The backbone starts frozen; call unfreeze_backbone() before full fine-tune.
    """

    def __init__(self, backbone: FoundationBackbone, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.feature_dim, n_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.get_features(X))

    def get_features(self, X: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_features(X)

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------

FOUNDATION_REGISTRY: dict[str, type[FoundationBackbone]] = {
    "mirepnet": MIRepNetBackbone,
    "neurogpt": NeuroGPTBackbone,
    "reve":     REVEBackbone,
    "cbramod":  CBraModBackbone,
}

FOUNDATION_NAMES = list(FOUNDATION_REGISTRY)


def build_foundation_model(
    name: str,
    n_channels: int,
    n_times: int,
    checkpoint_path: Optional[str] = None,
    freeze: bool = True,
    **kwargs,
) -> FoundationBackbone:
    """Build and optionally freeze a pretrained foundation EEG backbone.

    Args:
        name: one of 'mirepnet', 'neurogpt', 'reve', 'cbramod'
        n_channels: number of EEG channels in the data
        n_times: number of time samples per epoch
        checkpoint_path: path to pretrained weights (None = random init / debug)
        freeze: freeze all parameters so the backbone is a pure feature extractor
    """
    if name not in FOUNDATION_REGISTRY:
        raise ValueError(
            f"Unknown foundation model '{name}'. Choose from {FOUNDATION_NAMES}"
        )
    model = FOUNDATION_REGISTRY[name](
        n_channels=n_channels, n_times=n_times,
        checkpoint_path=checkpoint_path, **kwargs,
    )
    if freeze:
        model.freeze()
    return model
