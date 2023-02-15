import os
from pretrain.simclr.simclrlm import SimCLRLM


CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "pretrain/simclr/models/")


def get_pretrained_model(path):
    # Check if pretrained model exists
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Pretrained model does not exist at: {path}"
        )

    return SimCLRLM.load_from_checkpoint(path)


def summarise():
    print("Done! :)")
