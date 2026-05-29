"""EEGNet and ShallowConvNet specialist backbone wrappers using braindecode.

Both models:
  - Accept input (batch, C, T) — braindecode handles internal Ensure4d
  - Output class logits (batch, n_classes)
  - Expose has_batchnorm property for TTA method selection
"""

import torch
import torch.nn as nn
from braindecode.models import EEGNet, ShallowFBCSPNet, Deep4Net


def build_eegnet(
    n_channels: int,
    n_classes: int,
    n_times: int,
    F1: int = 8,
    D: int = 2,
    F2: int = 16,
    kernel_length: int = 64,
    drop_prob: float = 0.25,
) -> nn.Module:
    """EEGNet (~3k params). Standard config per Lawhern et al. 2018."""
    model = EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        F1=F1,
        D=D,
        F2=F2,
        kernel_length=kernel_length,
        drop_prob=drop_prob,
        final_conv_length="auto",
    )
    return model


def build_shallowconvnet(
    n_channels: int,
    n_classes: int,
    n_times: int,
    n_filters_time: int = 40,
    n_filters_spat: int = 40,
    drop_prob: float = 0.5,
) -> nn.Module:
    """ShallowConvNet (~50k params). Standard config per Schirrmeister et al. 2017."""
    model = ShallowFBCSPNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        n_filters_time=n_filters_time,
        n_filters_spat=n_filters_spat,
        drop_prob=drop_prob,
        final_conv_length="auto",
    )
    return model


def build_deep4net(
    n_channels: int,
    n_classes: int,
    n_times: int,
    n_filters_time: int = 25,
    n_filters_spat: int = 25,
    n_filters_2: int = 50,
    n_filters_3: int = 100,
    n_filters_4: int = 200,
    drop_prob: float = 0.5,
) -> nn.Module:
    """Deep4Net (~280k params). Standard config per Schirrmeister et al. 2017."""
    model = Deep4Net(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        n_filters_time=n_filters_time,
        n_filters_spat=n_filters_spat,
        n_filters_2=n_filters_2,
        n_filters_3=n_filters_3,
        n_filters_4=n_filters_4,
        drop_prob=drop_prob,
        final_conv_length="auto",
    )
    return model


class _ConformerPatchEmbedding(nn.Module):
    """Conv embedding: temporal conv → spatial conv → pooling → projection."""

    def __init__(self, n_channels: int, emb_size: int = 40, drop_prob: float = 0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, 40, (1, 25), stride=(1, 1))
        self.spatial = nn.Conv2d(40, 40, (n_channels, 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(40)
        self.act = nn.ELU()
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.drop = nn.Dropout(drop_prob)
        self.proj = nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        x = self.proj(x)          # (B, emb, 1, tokens)
        x = x.squeeze(2).transpose(1, 2)  # (B, tokens, emb)
        return x


class EEGConformer(nn.Module):
    """EEG-Conformer (Song et al. 2022).

    Conv embedding → Transformer encoder → MLP head.
    Input: (batch, channels, time). Output: (batch, n_classes).
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        n_times: int,
        emb_size: int = 40,
        depth: int = 6,
        n_heads: int = 10,
        forward_expansion: int = 4,
        emb_drop: float = 0.5,
        forward_drop: float = 0.5,
        attn_drop: float = 0.5,
    ):
        super().__init__()
        self.embedding = _ConformerPatchEmbedding(n_channels, emb_size, emb_drop)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=n_heads,
            dim_feedforward=emb_size * forward_expansion,
            dropout=forward_drop,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Determine flattened size by a dry-run
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            tokens = self.embedding(dummy)               # (1, T_tok, emb)
            flat_size = tokens.shape[1] * tokens.shape[2]

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) — add channel dim for Conv2d
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


def build_eeg_conformer(
    n_channels: int,
    n_classes: int,
    n_times: int,
    emb_size: int = 40,
    depth: int = 6,
    n_heads: int = 10,
) -> nn.Module:
    """EEG-Conformer (~790k params). Per Song et al. 2022."""
    return EEGConformer(
        n_channels=n_channels,
        n_classes=n_classes,
        n_times=n_times,
        emb_size=emb_size,
        depth=depth,
        n_heads=n_heads,
    )


def has_batchnorm(model: nn.Module) -> bool:
    """Return True if model contains BatchNorm layers (uses TENT), else T3A."""
    return any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules())


def get_conv_layer_names(model: nn.Module) -> list[str]:
    """Return names of Conv2d and Linear layers suitable for LoRA targeting."""
    names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            names.append(name)
    return names


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


BACKBONE_REGISTRY = {
    "eegnet": build_eegnet,
    "shallowconv": build_shallowconvnet,
    "deep4net": build_deep4net,
    "conformer": build_eeg_conformer,
}


def build_backbone(name: str, n_channels: int, n_classes: int, n_times: int) -> nn.Module:
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Choose from {list(BACKBONE_REGISTRY)}")
    return BACKBONE_REGISTRY[name](n_channels=n_channels, n_classes=n_classes, n_times=n_times)
