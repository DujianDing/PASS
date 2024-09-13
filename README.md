# PASS
This is the repository for the paper "PASS: Pruning Attention Heads with Almost-sure Sparsity Targets".

## How To Run

If you are using Slurm to submit jobs, you can use the provided `.sh` files to run the experiments.

```aiignore
sbatch fairseq/submit_job_ED_Transformer.sh
sbatch transformers/submit_job_BERT.sh
```
Otherwise, you can manually run the experiments from the command line.
### ED-Transformer on IWSLT
Install the fairseq library adapted from [Fairseq](https://github.com/pytorch/fairseq).
```aiignore
cd fairseq
pip install -e .
```
Download and prepare the IWSLT dataset.
```aiignore
cd examples/pruning
mkdir -p data-bin
source ./prepare-iwslt14.sh
export TEXT=iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/\
  --workers 20
```
Prune the model and evaluate.
```aiignore
for i in 32
do
echo "K = $i"
export SAVE_DIR=checkpoints/PASS_sanitycheck$i
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

echo "head mask with BMM"
python generate_pass.py \
	data-bin \
	-s de -t en \
	--path $SAVE_DIR/checkpoint_best.pt \
	--quiet \
	--batch-size 32 --beam 5 --remove-bpe
echo "-end of round-"

done
```
where `sparsity-rate` is the target number of heads to keep unpruned. 

### BERT on GLUE Tasks
Download [GLUE datasets](https://gluebenchmark.com/tasks) (MNLI, QNLI, QQP, and SST-2) to `data/` directory.

Install the Transformer library adapted from [Huggingface](https://github.com/huggingface/transformers).
```
cd transformers
pip install -e .
```
Then install other dependencies.
```
cd examples
pip install -r requirements.txt
```
Specify the task name such as `MNLI`, `QNLI`, `QQP`, or `SST-2`.
```
cd pruning
export TASK_NAME=MNLI
```
Prune the model and evaluate.
```
for i in 16
do
  echo "K = $i"
  export OUT_DIR=PASS_${TASK_NAME}_sanitycheck$i
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
```
where `num_of_heads` is the target number of heads to keep unpruned. 

