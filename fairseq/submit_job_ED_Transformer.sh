#!/bin/bash
#SBATCH --time=0-8:00
#SBATCH --account=XXXXXXXXXX
#SBATCH --job-name=PASS_sanitycheck
#SBATCH --output=PASS_sanitycheck.out
#SBATCH --mail-user=XXXXXXXXXX
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1

lscpu
hostname

module load nixpkgs/16.09 gcc/7.3.0 python/3.7.4 geos/3.7.2
module list
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install geos --no-index
pip install Cython==0.29.24 --no-index
pip install numpy==1.20.2 --no-index
pip install -e .
pip install -r requirements.txt
cd examples/pruning
mkdir -p data-bin

if [ ! -f ./data-bin/dict.de.txt ]; then
  echo "Creating data-bin directory"
  source ./prepare-iwslt14.sh
  export TEXT=iwslt14.tokenized.de-en
  fairseq-preprocess --source-lang de --target-lang en \
      --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
      --destdir data-bin/\
      --workers 20
fi

ls data-bin

for j in 1
do

for i in 32
do
echo "K = $i"
export SAVE_DIR=~/scratch/adaptive_hp_checkpoint/PASS_sanitycheck$i$j
python run_pass.py \
		data-bin/ \
		--arch transformer_iwslt_de_en --share-decoder-input-output-embed \
		--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
		--dropout 0.3 --weight-decay 0.0001 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--max-tokens 4096 \
		--max-epoch 30 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
		--sparsity-rate $i \
		--save-dir $SAVE_DIR/

python generate_pass.py \
	data-bin \
	-s de -t en \
	--path $SAVE_DIR/checkpoint_best.pt \
	--quiet \
	--batch-size 32 --beam 5 --remove-bpe
echo "-end of round-"

done
done

