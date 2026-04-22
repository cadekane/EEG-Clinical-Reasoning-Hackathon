#!/bin/bash
#SBATCH --job-name=datalad_eeg_auditory
#SBATCH --output=datalad_eeg_auditory_%j.out
#SBATCH --error=datalad_eeg_auditory_%j.err
#SBATCH --partition=shared
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G

echo "========================================================"
echo "Starting OpenNeuro EEG Download Job (Auditory ds005048)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "========================================================"

cd /mnt/lustre/koa/scratch/$USER || exit 1

echo "Purging modules..."
module purge

echo "Loading Python module..."
module load lang/Python/3.13.5-GCCcore-14.3.0

echo "Python info:"
which python
python --version

echo "Setting user-local PATH..."
export PATH=$HOME/.local/bin:$PATH
hash -r

echo "Checking pip..."
python -m pip --version || python -m ensurepip --user

echo "Installing OpenNeuro CLI..."
python -m pip install --user --upgrade openneuro-py || exit 1

echo "Checking OpenNeuro CLI..."
which openneuro-py || true
openneuro-py --help || exit 1

echo "Creating target directory..."
mkdir -p /mnt/lustre/koa/scratch/$USER/ds005048_eeg

echo "Downloading auditory EEG dataset (ds005048)..."
openneuro-py download \
  --dataset=ds005048 \
  --target-dir=/mnt/lustre/koa/scratch/$USER/ds005048_eeg

echo "Checking downloaded files..."
find /mnt/lustre/koa/scratch/$USER/ds005048_eeg -name "*.set" | head -20
find /mnt/lustre/koa/scratch/$USER/ds005048_eeg -name "*.fdt" | head -20

echo "========================================================"
echo "End time: $(date)"
echo "Job finished."
echo "========================================================"
