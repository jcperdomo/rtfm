#!/bin/bash

#SBATCH --job-name=tabfm-eval
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --chdir=/p/project/transfernetx/perdomosilva1/rtfm
#SBATCH --output=/p/project/transfernetx/perdomosilva1/slurm-out/%j.out
#SBATCH --ntasks=1
#SBATCH --nodes=1



# JUWELS jobs
#SBATCH --account=transfernetx
#SBATCH --time=00-05:00:00
#SBATCH --partition=booster

set -x -e

echo "START TIME: $(date)"

echo "SLURM_JOB_ID is ${SLURM_JOB_ID}"
echo "SLURM_NODEID is ${SLURM_NODEID}"

echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
which conda
conda activate rtfm
which python
# source ~/.bashrc

VALUES="map"
NAMES="map"

export HF_DATASETS_OFFLINE=1

# : ${CKPT_DIR:=""}
# echo "CKPT_DIR is ${CKPT_DIR}"
# CKPT_DIR_CLEANED="${CKPT_DIR//\//_}"
# : ${MAX_SAMPLES:="10"}
# : ${SERIALIZER_CLS:="BasicSerializerV2"}
# : ${SHUFFLE_TABLE_FEATURES:="False"}
# : ${CONTEXT_LENGTH:="8192"}
# : ${SHOT_SEL:="rices"}

# OUTPUT_DIR="results/fewshot/${CKPT_DIR_CLEANED}__${SERIALIZER_CLS}"
OUTPUT_DIR="results"

echo "TASK is ${TASK}"
echo "OUTPUT_DIR is ${OUTPUT_DIR}"
echo "JUANKY SHOT_SEL is ${SHOT_SEL}"


# Initialize GPU index
GPU_INDEX=0

# Array to hold background process IDs
declare -a PIDS

# for NUM_SHOTS in 0 1 2 3 4 8 16 32; do
for NUM_SHOTS in 5 10; do

  # OUTPUT_CSV="${TASK}-vals${VALUES}-names${NAMES}-${SERIALIZER_CLS}-${NUM_SHOTS}shots.csv"
  OUTPUT_CSV="${TASK//\//_}-${SHOT_SEL}--${NUM_SHOTS}shots.csv" 
  OUTFILE="${OUTPUT_DIR}/${OUTPUT_CSV}"
  echo "OUTFILE is ${OUTFILE}"
  USER_CONFIG_DIR="/p/project/transfernetx/perdomosilva1/rtfm/evaldatasets/" \
  CUDA_VISIBLE_DEVICES=$GPU_INDEX \
  python -m rtfm.evaluation.evaluate_checkpoint_v2 \
    --eval-task-names ${TASK} \
    --model_name ${CKPT_DIR} \
    --eval_max_samples $MAX_SAMPLES \
    --context_length ${CONTEXT_LENGTH} \
    --feature_value_handling "map" \
    --feature_name_handling "map" \
    --pack_samples "False" \
    --num_shots $NUM_SHOTS \
    --shot-selector ${SHOT_SEL} \
    --outfile $OUTFILE &


  # Save PID of the background process
  PIDS+=($!)

  # Increment and reset GPU index
  GPU_INDEX=$(( (GPU_INDEX + 1) % 4 ))

  # If we've launched one process per GPU, wait for all to complete
  if [ $GPU_INDEX -eq 0 ]; then
    wait ${PIDS[@]}
    PIDS=()  # Clear PIDs array
  fi
done

# Wait for any remaining processes
wait ${PIDS[@]}
