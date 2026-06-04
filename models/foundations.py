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
        return self.embed_model["model"](x.flatten(1))    # (B, 1024)


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

    _EMB_DIM   = _NeuroGPTEmbedder.EMB_DIM     # 1024
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
# LaBraM — Large Brain Model (Jiang et al. 2024)
# ---------------------------------------------------------------------------

# Standard 10-20 electrode list used by LaBraM for positional embedding lookup.
# Channel at index i maps to pos_embed slot i+1 (slot 0 is always the CLS token).
_LABRAM_STANDARD_1020: list[str] = [
    'FP1','FPZ','FP2',
    'AF9','AF7','AF5','AF3','AF1','AFZ','AF2','AF4','AF6','AF8','AF10',
    'F9','F7','F5','F3','F1','FZ','F2','F4','F6','F8','F10',
    'FT9','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','FT10',
    'T9','T7','C5','C3','C1','CZ','C2','C4','C6','T8','T10',
    'TP9','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','TP10',
    'P9','P7','P5','P3','P1','PZ','P2','P4','P6','P8','P10',
    'PO9','PO7','PO5','PO3','PO1','POZ','PO2','PO4','PO6','PO8','PO10',
    'O1','OZ','O2','O9','CB1','CB2',
    'IZ','O10','T3','T5','T4','T6','M1','M2','A1','A2',
    'CFC1','CFC2','CFC3','CFC4','CFC5','CFC6','CFC7','CFC8',
    'CCP1','CCP2','CCP3','CCP4','CCP5','CCP6','CCP7','CCP8',
    'T1','T2','FTT9h','TTP7h','TPP9h','FTT10h','TPP8h','TPP10h',
    'FP1-F7','F7-T7','T7-P7','P7-O1','FP2-F8','F8-T8','T8-P8','P8-O2',
    'FP1-F3','F3-C3','C3-P3','P3-O1','FP2-F4','F4-C4','C4-P4','P4-O2',
]

# Pre-computed input_chans for BCIC-IV 2a (22 ch) → LaBraM pos_embed indices.
# Format: [0 (CLS), chan1_pos, ...] where pos = 1 + index in _LABRAM_STANDARD_1020.
# BCIC-IV 2a channel order (uppercased): FZ FC3 FC1 FCZ FC2 FC4 C5 C3 C1 CZ
#   C2 C4 C6 CP3 CP1 CPZ CP2 CP4 P1 PZ P2 POZ
_BCICIV2A_LABRAM_INPUT_CHANS: list[int] = [
    0,   # CLS
    20,  # FZ
    29,  # FC3
    30,  # FC1
    31,  # FCZ
    32,  # FC2
    33,  # FC4
    39,  # C5
    40,  # C3
    41,  # C1
    42,  # CZ
    43,  # C2
    44,  # C4
    45,  # C6
    51,  # CP3
    52,  # CP1
    53,  # CPZ
    54,  # CP2
    55,  # CP4
    63,  # P1
    64,  # PZ
    65,  # P2
    75,  # POZ
]


def _labram_get_input_chans(ch_names: list[str]) -> list[int]:
    """Map EEG channel names to LaBraM pos_embed indices (0 = CLS slot)."""
    normalized = [n.upper().replace('EEG-', '') for n in ch_names]
    return [0] + [_LABRAM_STANDARD_1020.index(n) + 1 for n in normalized]


def _labram_drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.rand(shape, dtype=x.dtype, device=x.device).floor_(keep_prob + keep_prob)
    return x * mask / keep_prob


class _LaBraMDropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _labram_drop_path(x, self.drop_prob, self.training)


class _LaBraMMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.):
        super().__init__()
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _LaBraMAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qk_norm=None,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = qk_norm(head_dim) if qk_norm is not None else None
        self.k_norm = qk_norm(head_dim) if qk_norm is not None else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _LaBraMBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.,
                 qk_norm=None, drop: float = 0., attn_drop: float = 0.,
                 drop_path: float = 0., norm_layer=nn.LayerNorm,
                 init_values: Optional[float] = None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn  = _LaBraMAttention(dim, num_heads=num_heads, qk_norm=qk_norm,
                                       attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = _LaBraMDropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp   = _LaBraMMlp(dim, int(dim * mlp_ratio), drop=drop)
        use_gamma = init_values is not None and init_values > 0
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if use_gamma else None
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if use_gamma else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class _LaBraMTemporalConv(nn.Module):
    """Temporal patch embedding used in LaBraM (TemporalConv from modeling_finetune.py).

    Input:  (B, C, P, 200) — C channels, P patches of 200 samples each.
    Output: (B, C*P, embed_dim) — one token per (channel, patch) pair.
    """

    def __init__(self, out_chans: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu3 = nn.GELU()
        self.norm3 = nn.GroupNorm(4, out_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, P, T=200)
        B, C, P, T = x.shape
        x = x.reshape(B, C * P, T).unsqueeze(1)          # (B, 1, C*P, 200)
        x = self.gelu1(self.norm1(self.conv1(x)))          # (B, 8, C*P, 25)
        x = self.gelu2(self.norm2(self.conv2(x)))          # (B, 8, C*P, 25)
        x = self.gelu3(self.norm3(self.conv3(x)))          # (B, 8, C*P, 25)
        _, OC, CP, T2 = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, CP, T2 * OC)  # (B, C*P, 200)
        return x


class _LaBraMModel(nn.Module):
    """Minimal NeuralTransformer backbone matching the LaBraM checkpoint layout.

    Checkpoint keys (after stripping 'student.' prefix): patch_embed.*, cls_token,
    pos_embed, time_embed, blocks.{0..11}.*, norm.*
    """

    def __init__(self, embed_dim: int = 200, depth: int = 12, num_heads: int = 10,
                 mlp_ratio: float = 4., out_chans: int = 8, drop_rate: float = 0.,
                 attn_drop_rate: float = 0., drop_path_rate: float = 0.,
                 norm_layer=nn.LayerNorm, qk_norm=None, init_values: float = 0.1):
        super().__init__()
        self.patch_embed = _LaBraMTemporalConv(out_chans=out_chans)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, 129, embed_dim))   # 128 ch + CLS
        self.time_embed  = nn.Parameter(torch.zeros(1, 16, embed_dim))    # up to 16 patches
        self.pos_drop    = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            _LaBraMBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qk_norm=qk_norm, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        nn.init.trunc_normal_(self.pos_embed,  std=.02)
        nn.init.trunc_normal_(self.time_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token,  std=.02)

    def forward_features(self, x: torch.Tensor,
                          input_chans: Optional[list] = None) -> torch.Tensor:
        """Return mean-pooled patch features: (B, embed_dim)."""
        B, C, P, _ = x.shape  # last dim is patch_size (200)
        x = self.patch_embed(x)                          # (B, C*P, 200)

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat((cls, x), dim=1)                 # (B, 1+C*P, 200)

        # Channel positional embedding (one per electrode)
        pos_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
        pos_ch   = pos_used[:, 1:, :].unsqueeze(2).expand(B, -1, P, -1).flatten(1, 2)
        pos_all  = torch.cat((pos_used[:, :1, :].expand(B, -1, -1), pos_ch), dim=1)
        x = x + pos_all

        # Temporal positional embedding (one per time patch)
        t_emb = self.time_embed[:, :P, :].unsqueeze(1).expand(B, C, -1, -1).flatten(1, 2)
        x[:, 1:, :] = x[:, 1:, :] + t_emb

        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1:].mean(dim=1)   # mean-pool all patch tokens, skip CLS


class LaBraMBackbone(FoundationBackbone):
    """LaBraM: Large Brain Model (Jiang et al. 2024).

    Pretrained weights (base variant, ~5 M backbone params):
        https://huggingface.co/935963004/LaBraM  (labram-base.pth)
    Upload to Modal:
        modal volume put eeg-data ./labram-base.pth /labram-base.pth

    Input requirements:
        - Any EEG montage whose channel names appear in the standard 10-20 list
        - Any sampling rate (internally resampled to 200 Hz)
        - Signal assumed to be in µV (divided by 100 before the model)
        - Epoch length must be ≥ 1 s (one 200-sample patch at 200 Hz)

    Channel mapping:
        - Pass channel_names=['Fz','FC3',...] to look up positions automatically.
        - Pass input_chans=[0, 20, 29, ...] directly to bypass the lookup.
        - If n_channels == 22, BCIC-IV 2a positions are used by default.

    Checkpoint loading:
        The released labram-base.pth is a pretraining checkpoint where all
        weights live under 'student.*'. This wrapper strips that prefix and
        loads patch_embed, cls_token, pos_embed, time_embed, blocks, and norm.
        The pretraining lm_head is discarded.
    """

    _EMB_DIM      = 200
    _PATCH_SIZE   = 200
    _TARGET_SFREQ = 200.0

    def __init__(self, n_channels: int, n_times: int,
                 input_sfreq: float = 200.0,
                 channel_names: Optional[list] = None,
                 input_chans: Optional[list] = None,
                 checkpoint_path: Optional[str] = None, **kwargs):
        super().__init__()
        self.input_sfreq = input_sfreq

        # Resolve input_chans
        if input_chans is not None:
            self.input_chans = input_chans
        elif channel_names is not None:
            self.input_chans = _labram_get_input_chans(channel_names)
        elif n_channels == 22:
            self.input_chans = _BCICIV2A_LABRAM_INPUT_CHANS
        else:
            raise ValueError(
                "LaBraM requires either channel_names, explicit input_chans, or "
                f"the 22-channel BCIC-IV 2a layout; got n_channels={n_channels}."
            )

        norm_layer = lambda d: nn.LayerNorm(d, eps=1e-6)
        qk_norm    = lambda d: nn.LayerNorm(d, eps=1e-6)
        self.model = _LaBraMModel(
            embed_dim=self._EMB_DIM, depth=12, num_heads=10, mlp_ratio=4.,
            out_chans=8, norm_layer=norm_layer, qk_norm=qk_norm, init_values=0.1,
        )

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if isinstance(state, dict) and 'model' in state:
                state = state['model']
            # Strip 'student.' prefix; discard lm_head, mask_token, logit_scale, etc.
            state = {
                k.removeprefix('student.'): v
                for k, v in state.items()
                if k.startswith('student.') and not k.startswith('student.lm_head')
                and not k.startswith('student.mask_token')
            }
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(
                    f"[LaBraM] loaded with {len(missing)} missing and "
                    f"{len(unexpected)} unexpected keys"
                )

    @property
    def feature_dim(self) -> int:
        return self._EMB_DIM

    def get_features(self, X: torch.Tensor) -> torch.Tensor:
        """Extract 200-dim features from (B, C, T) EEG input."""
        B, C, T = X.shape

        # 1. Resample to 200 Hz
        if self.input_sfreq != self._TARGET_SFREQ:
            target_len = int(T * self._TARGET_SFREQ / self.input_sfreq)
            X = F.interpolate(X, size=target_len, mode='linear', align_corners=False)

        # 2. LaBraM pretraining expects EEG in microvolts, scaled by 100 µV.
        # Our BCIC pipeline preserves MNE's native volt units, so convert V -> µV
        # before applying the released checkpoint's input normalization.
        X = (X * 1e6) / 100.0

        # 3. Pad/trim to a multiple of patch_size (200 samples)
        T2 = X.shape[-1]
        n_patches = max(1, (T2 + self._PATCH_SIZE - 1) // self._PATCH_SIZE)
        target_len = n_patches * self._PATCH_SIZE
        if T2 < target_len:
            X = F.pad(X, (0, target_len - T2))
        elif T2 > target_len:
            X = X[:, :, :target_len]

        # 4. Reshape to (B, C, P, 200) and extract features
        X = X.reshape(B, C, n_patches, self._PATCH_SIZE)
        return self.model.forward_features(X, input_chans=self.input_chans)


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
    "labram":   LaBraMBackbone,
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
        name: one of 'mirepnet', 'neurogpt', 'reve', 'labram'
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
