#!/bin/bash

# ==============================================================================
# Currently, there is no early stopping. Validation accuracy may start going
# down as model continues to train.
# 
# The saved model under models/ is the version after training for specified
# epochs. To replace it with the best-performing checkpoint during training,
# copy this script in ../models and run it.
#
#   $ chmod +x replace-with-best-checkpoint.sh
#   $ ./replace-with-best-checkpoint.sh
#
# The original .ckpt file will be deleted and replaced with the best checkpoint.
# ==============================================================================

SOURCE_DIR="models/tb_logs"
DEST_DIR="models"

for folder in "$SOURCE_DIR"/*/; do
  folder_name=$(basename "$folder")
  checkpoint_file=$(find "$folder" -name '*.ckpt' -type f -exec basename {} \;)
  new_checkpoint_file="$folder_name.ckpt"
  cp "$folder/version_0/checkpoints/$checkpoint_file" "$DEST_DIR/$new_checkpoint_file"
  rm "$folder/version_0/checkpoints/$checkpoint_file"
done
