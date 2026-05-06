#!/bin/bash -l

#$ -l h_rt=24:00:00
#$ -l gpus=1
#$ -l gpu_type=A40
#$ -pe omp 4
#$ -l mem_per_core=4G
#$ -N exp318
#$ -o logs/exp318.out
#$ -e logs/exp318.err
#$ -j y


module load python3/3.9.9

module load cuda/12.8

# Activate your venv

source /projectnb/eb-mlhfi/workspaces/atuladas/ml_hfi_multiyear/venv/bin/activate

# Run

cd /projectnb/eb-mlhfi/workspaces/atuladas/ml_hfi_multiyear

echo "PWD=$(pwd)"
echo "PYTHONPATH=$PYTHONPATH"
python -u init.py
python -u driver.py exp318










