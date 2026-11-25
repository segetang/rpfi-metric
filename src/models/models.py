"""
models.py — unified definitions of the model architectures used in the RPFI experiments
========================================================
The architectures of the earlier individual scripts (RNN.py, LSTM.py, bilstm_test.py, physdiff.py,
mamba.py, bimamba(plain).py) are carried over unchanged, while data loading, splitting,
preprocessing and prediction saving all moved to train_models.py.

Changes from the earlier scripts
-------------------------
1. the RPPGDataset / loading / split / training loop duplicated in each file was removed
   -> handled by the common harness in train_models.py (all models use the same split and preprocessing)
2. physdiff: the original `torch.utils.data.random_split` was removed
   (random splitting of 1-sample-stride sliding windows caused extreme leakage)
3. bimamba: the original KFold(shuffle=True) path was removed; only plain BiMamba is kept
4. hyperparameters keep the original script values as far as possible (see DEFAULTS below)

The Mamba family requires the `mamba_ssm` package and CUDA. If it is not installed, the error is raised
when the model is built rather than at import time (deferred import).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# 1. recurrent family — RNN / LSTM / BiLSTM
#    original: RNN.py, LSTM.py, bilstm_test.py
# ══════════════════════════════════════════════════════════════

class RNNModel(nn.Module):
    """RNNManyToManyModel of the original RNN.py (hidden 64, 2 layers, tanh)."""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, nonlinearity="tanh",
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):                     # (B, L, 1)
        out, _ = self.rnn(x)
        return self.fc(out).squeeze(-1)       # (B, L)


class LSTMModel(nn.Module):
    """LSTMManyToManyModel of the original LSTM.py (hidden 64, 2 layers, dropout 0.2)."""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out)).squeeze(-1)


class BiLSTMModel(nn.Module):
    """BiLSTMManyToManyModel of the original bilstm_test.py (hidden 64, 2 layers, bidirectional)."""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.fc(out).squeeze(-1)


# ══════════════════════════════════════════════════════════════
# 2. PhysDiff — original physdiff.py
#    dynamic disentangle(trend/amplitude) + Transformer encoder
# ══════════════════════════════════════════════════════════════

def disentangle(signal):
    """disentangle of the original physdiff.py, unchanged.
    Separates trend (sign component) and amplitude of the first difference."""
    s = signal.squeeze(-1)                     # (B, L)
    diff = s[:, 1:] - s[:, :-1]
    trend = diff / torch.sqrt(diff ** 2 + 1.0)
    amp = diff.abs()
    trend = torch.cat([trend, trend[:, -1:]], dim=1)
    amp = torch.cat([amp, amp[:, -1:]], dim=1)
    return trend.unsqueeze(-1), amp.unsqueeze(-1)


class PhysDiffModel(nn.Module):
    """Original PhysDiffTransformer (d_model 64, nhead 4, 4 layers).

    Note for the paper: the original PhysDiff (Qian et al., AAAI 2025) is a video-input
      diffusion model. This implementation is an adaptation that takes the POS waveform as input,
      and the paper must state this difference.
    """

    def __init__(self, d_model=64, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, 1)

    def forward(self, x):                      # (B, L, 1)
        trend, amp = disentangle(x)
        cond = torch.cat([x, trend, amp], dim=-1)   # (B, L, 3)
        h = F.relu(self.input_proj(cond))
        h = self.transformer(h)
        return self.output_fc(h).squeeze(-1)


# ══════════════════════════════════════════════════════════════
# 3. TemporalTransformer — new baseline
#    replaces the earlier radiant/transformer.py (2 tokens -> temporal-patch tokens)
# ══════════════════════════════════════════════════════════════

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TemporalTransformer(nn.Module):
    """Standard Transformer encoder with temporal patches as tokens.

    Differences from the earlier RadiantModel:
      - tokens: channels (= 1) + cls = 2  ->  L / patch_size = 20-30
      - self-attention along time: absent  ->  present
      - parameters: 44.0M  ->  0.53M
    """

    def __init__(self, patch_size=10, d_model=128, nhead=4, num_layers=4,
                 dim_ff=256, dropout=0.1, max_tokens=512):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos = SinusoidalPE(d_model, max_len=max_tokens)
        self.in_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, patch_size)

    def forward(self, x):
        B, L, _ = x.shape
        if L % self.patch_size != 0:
            raise ValueError(f"seq_len ({L}) must be a multiple of patch_size ({self.patch_size}).")
        h = self.patch_embed(x.transpose(1, 2)).transpose(1, 2)
        h = self.in_drop(self.pos(h))
        h = self.norm(self.encoder(h))
        return self.head(h).reshape(B, L)


# ══════════════════════════════════════════════════════════════
# 4. Mamba family — mamba.py (EMamba), bimamba (plain)
# ══════════════════════════════════════════════════════════════

def _require_mamba():
    try:
        from mamba_ssm import Mamba
        return Mamba
    except ImportError as e:
        raise ImportError(
            "mamba_ssm is required (pip install mamba-ssm, needs CUDA). "
            "Run the Mamba-family models on a GPU."
        ) from e


class MambaModel(nn.Module):
    """ImprovedRPPGModel of the original mamba.py (d_model 128, d_state 32, 6 blocks).
    pre_conv (learned low-pass) + stack of Mamba blocks + residual."""

    def __init__(self, d_model=128, d_state=32, d_conv=8, n_blocks=6,
                 dropout=0.1, conv_kernel=7):
        super().__init__()
        Mamba = _require_mamba()
        self.pre_conv = nn.Conv1d(1, d_model, kernel_size=conv_kernel,
                                  padding=conv_kernel // 2)
        self.pre_act = nn.GELU()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2),
                nn.Dropout(dropout),
            ) for _ in range(n_blocks)
        ])
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.pre_act(self.pre_conv(x.permute(0, 2, 1))).permute(0, 2, 1)
        for blk in self.blocks:
            h = h + blk(h)
        return self.output_proj(h).squeeze(-1)


class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, dropout):
        super().__init__()
        Mamba = _require_mamba()
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        f = self.fwd(x)
        b = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
        return self.norm(self.drop((f + b) / 2) + x)


class BiMambaModel(nn.Module):
    """BiMambaRPPGModel of the original bimamba(plain).py (d_model 128, 6 blocks, bidirectional).

    Note: the original KFold(shuffle=True) ensemble path (biMamba-blend) is excluded;
      it split sliding windows randomly and caused extreme leakage.
    """

    def __init__(self, d_model=128, d_state=32, d_conv=8, n_blocks=6,
                 dropout=0.1, conv_kernel=7):
        super().__init__()
        self.pre_conv = nn.Conv1d(1, d_model, kernel_size=conv_kernel,
                                  padding=conv_kernel // 2)
        self.pre_act = nn.GELU()
        self.blocks = nn.ModuleList([
            BiMambaBlock(d_model, d_state, d_conv, dropout) for _ in range(n_blocks)
        ])
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.pre_act(self.pre_conv(x.permute(0, 2, 1))).permute(0, 2, 1)
        for blk in self.blocks:
            h = blk(h)
        return self.proj(h).squeeze(-1)


# ══════════════════════════════════════════════════════════════
# Registry — keeps the hyperparameters of the original scripts
# ══════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "rnn":         (RNNModel,            dict()),
    "lstm":        (LSTMModel,           dict()),
    "bilstm":      (BiLSTMModel,         dict()),
    "physdiff":    (PhysDiffModel,       dict()),
    "transformer": (TemporalTransformer, dict(patch_size=10)),
    "mamba":       (MambaModel,          dict()),
    "bimamba":     (BiMambaModel,        dict()),
}

# Training settings of the original scripts (used as train_models.py defaults)
DEFAULTS = {
    # same for all models: loss = SmoothL1, at most 200 epochs (early stopping decides the actual end)
    # optimizer/lr/batch keep the original script values because of architectural differences
    "rnn":         dict(lr=1e-3, epochs=200, batch=32, loss="smoothl1", opt="adam"),
    "lstm":        dict(lr=1e-3, epochs=200, batch=32, loss="smoothl1", opt="adam"),
    "bilstm":      dict(lr=1e-3, epochs=200, batch=32, loss="smoothl1", opt="adam"),
    "physdiff":    dict(lr=1e-3, epochs=200, batch=32, loss="smoothl1", opt="adam"),
    "transformer": dict(lr=3e-4, epochs=200, batch=64, loss="smoothl1", opt="adamw"),
    "mamba":       dict(lr=3e-4, epochs=200, batch=32, loss="smoothl1", opt="adamw"),
    "bimamba":     dict(lr=3e-4, epochs=200, batch=32, loss="smoothl1", opt="adamw"),
}

# Note on the SmoothL1 transition point beta:
#   PyTorch default beta = 1.0. Inputs are z-normalized per window, so the effective residual range is
#   roughly [0, 2]; with beta = 1.0 a large part of that range is in the linear (L1) regime,
#   i.e. large errors are penalized less than with MSE.
#   A lower beta (e.g. 0.5) is closer to MSE. Adjustable with train_models.py --smooth-beta;
#   the paper must state the beta that was used.

def build_model(name, **override):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model: {name}. Available: {list(MODEL_REGISTRY)}")
    cls, kw = MODEL_REGISTRY[name]
    kw = {**kw, **override}
    return cls(**kw)
