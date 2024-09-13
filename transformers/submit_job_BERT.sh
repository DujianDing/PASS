#!/bin/bash
#SBATCH --time=0-8:00
#SBATCH --account=XXXXXXXXXX
#SBATCH --job-name=PASS_QQP_sanitycheck
#SBATCH --output=PASS_QQP_sanitycheck.out
#SBATCH --mail-user=XXXXXXXXXX
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1

lscpu
hostname
nvidia-smi

module load StdEnv/2020 gcc/9.3.0 arrow/8.0.0 python/3.8.2
module list
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install -e .
cd examples
pip install -r requirements.txt
cd pruning

export TASK_NAME=QQP
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for j in 1
do
for i in 16
do
  echo "K = $i"
  export OUT_DIR=PASS_${TASK_NAME}_sanitycheck$j$i
  python run_pass.py \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --data_dir /data/$TASK_NAME/ \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /outputs/$OUT_DIR/ \
    --cache_dir /cache/$OUT_DIR/ \
    --save_steps 1000 \
    --num_of_heads $i \
    --joint_pruning \
    --pruning_lr 0.5

done
done

