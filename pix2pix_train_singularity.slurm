#!/usr/bin/env bash
#
#SBATCH -J pix2pix_facades           # Job name
#SBATCH -p normal                    # Partition on Nano5
#SBATCH -N 1                         # Number of nodes
#SBATCH -n 1                         # Number of tasks
#SBATCH --gres=gpu:1                 # Number of GPUs

#SBATCH --cpus-per-task=8            # CPU cores per task
#SBATCH --mem=32G                    # Memory
#SBATCH -t 24:00:00                  # Max run time (hh:mm:ss)
#SBATCH --account=MSTXXXXX           # <-- replace with your project/account ID
#SBATCH -o %x-%j.out                 # Stdout log
#SBATCH -e %x-%j.err                 # Stderr log

set -e

echo "Job started on $(date)"
echo "Running on host: $(hostname)"
echo "SLURM job ID: $SLURM_JOB_ID"

########################
# 1. Load modules
########################
module purge
# Adjust these to match Nano5 module names:
module load singularity          # or: module load apptainer
# module load cuda/12.1          # only if needed; many containers bundle CUDA

########################
# 2. Paths
########################
CONTAINER="/work/hpc_sys/sifs/pytorch_23.11-py3.sif"

WORKDIR="$HOME/NCHC-GAN-Pix2Pix-CycleGAN-demo/pytorch-CycleGAN-and-pix2pix"
cd "$WORKDIR" || { echo "Cannot cd to $WORKDIR"; exit 1; }

echo "Working directory: $(pwd)"
echo "Using container:   $CONTAINER"

########################
# 3. (Optional) Install Pix2Pix deps in user site-packages inside container
########################
echo "Checking / installing Python dependencies inside container (user space)..."

singularity exec --nv "$CONTAINER" python -m pip install --user --upgrade pip
singularity exec --nv "$CONTAINER" python -m pip install --user dominate visdom numpy pillow wandb

########################
# 4. Set dataset & training parameters
########################
DATASET_NAME="facades"
DATAROOT="./datasets/${DATASET_NAME}"
EXP_NAME="${DATASET_NAME}_pix2pix_normal"

echo "Dataset root: ${DATAROOT}"
echo "Experiment name: ${EXP_NAME}"

########################
# 5. Run Pix2Pix training inside Singularity
########################
echo "Starting training with Singularity + GPU..."

srun singularity exec --nv "$CONTAINER" python train2.py \
  --dataroot "${DATAROOT}" \
  --name "${EXP_NAME}" \
  --model pix2pix \
  --direction BtoA \
  --batch_size 16 \
  --n_epochs 100 \
  --num_threads 8

echo "Job finished on $(date)"

