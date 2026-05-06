#!/bin/bash -l

#$ -l h_rt=48:00:00
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -N build_tags
#$ -o logs/build_tags.out
#$ -e logs/build_tags.err
#$ -j y

module load python3/3.9.9

source /projectnb/eb-mlhfi/workspaces/atuladas/ml_hfi_multiyear/venv/bin/activate

cd /projectnb/eb-mlhfi/workspaces/atuladas/ml_hfi_multiyear

python -u init.py
python -u build_only_tags.py
