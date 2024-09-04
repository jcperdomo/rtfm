#!/bin/bash

echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate rtfm

# Note: this script evaluates a pretrained (*not* fine-tuned) model, as an integration
# test of the eval loop. If you want to test the (fine-tune --> evaluate) loop, then
# use the file train_and_eval_tiny_local_test.sh .
USER_CONFIG_DIR="./sampledata" \
python -m rtfm.evaluation.evaluate_checkpoint_v2 \
  --eval-task-names "multiclass_logistic" \
  --model_name "mlfoundations/tabula-8b" \
  --eval_max_samples 10 \
  --context_length 8192 \
  --feature_value_handling "map" \
  --feature_name_handling "map" \
  --pack_samples "False" \
  --num_shots 4 \
  --shot-selector "random" \
  --output_dir "checkpoints/tiny_trainer" \
  --outfile "tmp.csv"

echo "removing outfile tmp.csv"
rm "tmp.csv"
