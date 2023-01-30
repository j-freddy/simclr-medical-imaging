import os

SEED = 1969
NUM_WORKERS = os.cpu_count()
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
