import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
NUM_CLASSES = 2
SEQ_LEN = 8
IMAGE_SIZE = 32

TRAIN_CONFIG = {
    "batch_size": 64,
    "epochs": 3,
    "lr": 1e-3
}