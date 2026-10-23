"""Central configuration, taken from the CS229 paper (Li, Xue, Zhang 2018)."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "genres_original"   # GTZAN: <DATA_DIR>/<genre>/<genre>.00000.wav
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

GENRES = ["classical", "hiphop", "metal", "pop", "blues"]  # the paper's 5 genres

# Audio / MFCC. 20 coeffs x 1290 frames = 25,800 flattened (matches the paper).
SR = 22050
DURATION = 30
N_MFCC = 20
HOP_LENGTH = 512
N_FRAMES = 1290

SEED = 42
TEST_SIZE = 0.2          # 400 train / 100 test, fixed for every experiment


@dataclass(frozen=True)
class UnitCfg:
    """One dilated convolution unit: Conv1D -> AvgPool -> Dropout -> BatchNorm."""
    filters: int
    kernel: int
    dilation: int
    pool: int


# Simplest config in Table V.2 of the paper (0.85 test acc, little overfitting):
#   "8, 8, 4, 32" and "16, 4, 2, 8"  = (filters, filter size, dilation, pool)
DEFAULT_UNITS = (UnitCfg(8, 8, 4, 32), UnitCfg(16, 4, 2, 8))

LEARNING_RATE = 1e-3     # Adam, per the paper
DROPOUT = 0.5            # paper used 50%
BATCH_SIZE = 32
EPOCHS = 100

PCA_COMPONENTS = 50
RF_MAX_DEPTH = 7
