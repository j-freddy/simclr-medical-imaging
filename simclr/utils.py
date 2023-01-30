from const import SEED
import torch

def setup_device():
  # Use GPU if available
  device = torch.device("cuda") if torch.cuda.is_available()\
    else torch.device("cpu")

  if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Enforce all operations to be deterministic on GPU for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

  return device
