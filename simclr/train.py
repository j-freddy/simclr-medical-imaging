import pytorch_lightning as pl
from const import NUM_WORKERS, SEED
from utils import setup_device

if __name__=="__main__":
  pl.seed_everything(SEED)
  device = setup_device()
  print(f"Device: {device}")
  print(f"Number of workers: {NUM_WORKERS}")
