"""Shared style for the paper figures (same visual vocabulary as the v1 figures)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
for p in (ROOT / "rpfi", ROOT / "experiments", ROOT / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "savefig.dpi": 300,
})

REF = "#111111"        # reference (label) trace
PRED = "#3E7CB1"       # prediction trace
PRED2 = "#D9534F"      # second prediction / contrast
GRID = "#bbbbbb"
COMP = {"SHAPE": "#3E7CB1", "WCR": "#4C9A4C", "RHY": "#B5651D", "RPFI": "#333333"}
COMP_LABEL = {"SHAPE": "$s_{\\mathrm{SHAPE}}$", "WCR": "$s_{\\mathrm{WCR}}$", "RHY": "$s_{\\mathrm{RHY}}$",
              "RPFI": "RPFI"}
FAMILY = {"shape": "#3E7CB1", "local": "#B5651D", "drift": "#4C9A4C"}
DATASET = {"ubfc": "#3E7CB1", "pure": "#D9534F", "cohface": "#7B2D8E"}
DS_NAME = {"ubfc": "UBFC-rPPG", "pure": "PURE", "cohface": "COHFACE"}
MODEL_NAME = {"bilstm": "BiLSTM", "bimamba": "BiMamba", "bimamba_blend": "BiMamba-blend", "lstm": "LSTM",
              "mamba": "Mamba", "physdiff": "PhysDiff", "pos": "POS", "rnn": "RNN", "transformer": "Transformer"}


def panel_label(ax, letter, title=None, fontsize=15):
    """Bold '(A)' at the top-left of a panel. Panel titles are not drawn (descriptions go in the caption);
    `title` is accepted only so that existing calls keep working."""
    ax.set_title(f"({letter})", fontsize=fontsize, fontweight="bold", loc="left")


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


PRINT_WIDTH = {"full": 7.16, "col": 3.5}   # inches: two-column page width / single column (IEEE; Frontiers ~7.1 / 3.4)
MIN_PT = 8.0                             # smallest text size on the printed page
DPI = 300                                # raster resolution of saved figures (PNG; PDF stays vector)


def enforce_min_font(fig, width="full", min_pt=MIN_PT):
    """Enlarge every text that would print smaller than min_pt when the figure is scaled to the page width."""
    from matplotlib.text import Text
    scale = PRINT_WIDTH[width] / fig.get_figwidth()
    floor = min_pt / scale
    for t in fig.findobj(Text):
        if t.get_text().strip() and t.get_fontsize() < floor:
            t.set_fontsize(floor)
    return scale


def save(fig, name, width="full"):
    scale = enforce_min_font(fig, width)
    print(f"{name}: {fig.get_figwidth():.1f} in wide -> {PRINT_WIDTH[width]} in ({width}), scale {scale:.2f}, "
          f"min text {MIN_PT:.0f} pt printed")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved docs/figures/{name}.pdf / .png")
