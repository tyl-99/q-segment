#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --signal=B:INT@900
#SBATCH --output=logs/%x_%j.out

cd "$SLURM_SUBMIT_DIR" || exit 1
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname) n=$N seed=$SEED cap=$CAP start $(date -Is)"
exec "$HOME/qseg_env/bin/python" -u 18_exact_time_3zone.py --n "$N" --seed "$SEED" --cap-seconds "$CAP" --out results_18
