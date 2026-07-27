"""
models.py — RPFI 논문 재실험용 모델 아키텍처 통합 정의
========================================================
기존 개별 스크립트(RNN.py, LSTM.py, bilstm_test.py, physdiff.py,
mamba.py, bimamba(plain).py)의 아키텍처를 그대로 옮기되,
데이터 로딩/분할/전처리/예측저장은 전부 train_models.py로 이관했다.

기존 스크립트 대비 변경점
-------------------------
1. 각 파일에 중복되어 있던 RPPGDataset / 로딩 / split / 학습루프 제거
   -> train_models.py의 공통 하니스가 담당 (모든 모델이 동일 split·전처리 사용 보장)
2. physdiff: 원본의 `torch.utils.data.random_split` 제거
   (1샘플 간격 슬라이딩 윈도우를 랜덤 분할해 극단적 leakage 발생시켰음)
3. bimamba: 원본 KFold(shuffle=True) 경로 제거. plain BiMamba만 유지
4. 하이퍼파라미터는 원본 스크립트 값을 최대한 보존 (아래 DEFAULTS 참조)

Mamba 계열은 `mamba_ssm` 패키지와 CUDA가 필요하다. 미설치 환경에서는
import 시점이 아니라 모델 생성 시점에 에러를 내도록 지연 import 처리했다.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# 1. 순환 계열 — RNN / LSTM / BiLSTM
#    원본: RNN.py, LSTM.py, bilstm_test.py
# ══════════════════════════════════════════════════════════════

class RNNModel(nn.Module):
    """원본 RNN.py의 RNNManyToManyModel (hidden 64, 2층, tanh)."""

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
    """원본 LSTM.py의 LSTMManyToManyModel (hidden 64, 2층, dropout 0.2)."""

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
    """원본 bilstm_test.py의 BiLSTMManyToManyModel (hidden 64, 2층, 양방향)."""

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
# 2. PhysDiff — 원본 physdiff.py
#    dynamic disentangle(trend/amplitude) + Transformer encoder
# ══════════════════════════════════════════════════════════════

def disentangle(signal):
    """원본 physdiff.py의 disentangle 그대로.
    1차 차분에서 trend(부호 성분)와 amplitude를 분리."""
    s = signal.squeeze(-1)                     # (B, L)
    diff = s[:, 1:] - s[:, :-1]
    trend = diff / torch.sqrt(diff ** 2 + 1.0)
    amp = diff.abs()
    trend = torch.cat([trend, trend[:, -1:]], dim=1)
    amp = torch.cat([amp, amp[:, -1:]], dim=1)
    return trend.unsqueeze(-1), amp.unsqueeze(-1)


class PhysDiffModel(nn.Module):
    """원본 PhysDiffTransformer (d_model 64, nhead 4, 4층).

    ※ 논문 기술 주의: 원 PhysDiff(Qian et al., AAAI 2025)는 영상 입력
      diffusion 모델이다. 본 구현은 POS 파형을 입력으로 받는 각색판이며,
      리뷰어 R1-9 대응 시 이 차이를 반드시 명시해야 한다.
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
# 3. TemporalTransformer — 신규 baseline
#    기존 radiant/transformer.py 대체 (토큰 2개 -> 시간패치 토큰)
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
    """시간 패치를 토큰으로 하는 표준 Transformer encoder.

    기존 RadiantModel과의 차이:
      - 토큰: 채널수(=1)+cls = 2개  ->  L/patch_size = 20~30개
      - 시간축 self-attention: 부재  ->  존재
      - 파라미터: 44.0M  ->  0.53M
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
            raise ValueError(f"seq_len({L})이 patch_size({self.patch_size})의 배수여야 합니다.")
        h = self.patch_embed(x.transpose(1, 2)).transpose(1, 2)
        h = self.in_drop(self.pos(h))
        h = self.norm(self.encoder(h))
        return self.head(h).reshape(B, L)


# ══════════════════════════════════════════════════════════════
# 4. Mamba 계열 — mamba.py(EMamba), bimamba(plain)
# ══════════════════════════════════════════════════════════════

def _require_mamba():
    try:
        from mamba_ssm import Mamba
        return Mamba
    except ImportError as e:
        raise ImportError(
            "mamba_ssm이 필요합니다 (pip install mamba-ssm, CUDA 필요). "
            "Mamba 계열 모델은 GPU 환경에서 실행하세요."
        ) from e


class MambaModel(nn.Module):
    """원본 mamba.py의 ImprovedRPPGModel (d_model 128, d_state 32, 6블록).
    pre_conv(학습형 저역통과) + Mamba 블록 스택 + residual."""

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
    """원본 bimamba(plain).py의 BiMambaRPPGModel (d_model 128, 6블록, 양방향).

    ※ 원본 KFold(shuffle=True) 앙상블 경로(biMamba-blend)는 제외됨.
      해당 경로는 슬라이딩 윈도우를 랜덤 분할해 극단적 leakage를 유발했다.
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
# 레지스트리 — 원본 스크립트의 하이퍼파라미터 보존
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

# 원본 스크립트의 학습 설정 (train_models.py 기본값으로 사용)
DEFAULTS = {
    # 전 모델 통일: loss=SmoothL1, epochs 상한 200 (early stopping이 실제 종료점 결정)
    # optimizer/lr/batch는 아키텍처 특성상 원본 스크립트 값 유지
    "rnn":         dict(lr=1e-3, epochs=200, batch=32, loss="smoothl1", opt="adam"),
    "lstm":        dict(lr=1e-3, epochs=200, batch=32, loss="smoothl1", opt="adam"),
    "bilstm":      dict(lr=1e-3, epochs=200, batch=32, loss="smoothl1", opt="adam"),
    "physdiff":    dict(lr=1e-3, epochs=200, batch=32, loss="smoothl1", opt="adam"),
    "transformer": dict(lr=3e-4, epochs=200, batch=64, loss="smoothl1", opt="adamw"),
    "mamba":       dict(lr=3e-4, epochs=200, batch=32, loss="smoothl1", opt="adamw"),
    "bimamba":     dict(lr=3e-4, epochs=200, batch=32, loss="smoothl1", opt="adamw"),
}

# ※ SmoothL1의 전환점 beta 주의:
#   PyTorch 기본 beta=1.0. 입력이 윈도우별 z-normalize된 상태이므로 잔차의
#   실질 범위가 대략 [0, 2]이고, beta=1.0이면 그 범위 상당 부분이 선형(L1) 구간에
#   들어간다. 즉 큰 오차에 대한 페널티가 MSE보다 약해진다.
#   beta를 낮추면(예: 0.5) 더 MSE에 가까워진다. train_models.py --smooth-beta로 조정 가능하며,
#   논문에는 사용한 beta 값을 반드시 명시할 것 (리뷰어 R1-9의 loss function 보고 요구).

def build_model(name, **override):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"알 수 없는 모델: {name}. 사용 가능: {list(MODEL_REGISTRY)}")
    cls, kw = MODEL_REGISTRY[name]
    kw = {**kw, **override}
    return cls(**kw)
