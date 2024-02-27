#!/usr/bin/env

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training). 
# Please set the options below according to the comments. 
# For multi-gpu workers training, these options should be manually set for each worker. 
# After setting the options, please run the script on each worker.
# To use the shuffled data (if exists), please uncomment the Line 24.

# Number of GPUs per GPU worker
GPUS_PER_NODE=4
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1 
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
# export MASTER_ADDR=XX.XX.XX.XX
# The port for communication
export MASTER_PORT=1081
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0 

data_dir=../../dataset
data=${data_dir}/MINS/train.tsv,${data_dir}/val.tsv
# Note: If you have shuffled the data in advance, please uncomment the line below.

# Note: If you have shuffled the data in advance, please uncomment the line below.
# data=${data_dir}/vqa_train_1.tsv,${data_dir}/vqa_train_2.tsv,${data_dir}/vqa_train_3.tsv,${data_dir}/vqa_train_4.tsv,${data_dir}/vqa_train_5.tsv,${data_dir}/vqa_train_6.tsv,${data_dir}/vqa_train_7.tsv,${data_dir}/vqa_train_8.tsv,${data_dir}/vqa_train_9.tsv,${data_dir}/vqa_train_10.tsv,${data_dir}/vqa_val.tsv

restore_file=../../checkpoints/ofa_large.pt
selected_cols=0,3,5,6

log_dir=./instr_logs
save_dir=../../checkpoints/tuned
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

task=instr_tuning
arch=ofa_large
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1 # default
batch_size=2  # default
update_freq=8
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=1024
max_tgt_length=30
num_bins=1000
log_interval=100

uses_ema="--uses-ema"
store_ema="--store-ema"
ema_fp32="--ema-fp32"
ema_decay=0.9999
ema_start_update=0

# Specify the inference type in validation after each fine-tuning epoch
# As mentioned in the readme, you can choose from allcand or beamsearch evaluation, default to allcand
val_inference_type=allcand

# Specify whether to activate unconstrained VQA finetuning, which does not use a pre-defined candidate answer set
# If --unconstrained-training is acitvated, --ans2label-file will **not be used even if it is specified**
# Meanwhile, --val-inference-type must be set to **beamsearch**
# By default, we follow the constrained finetuning as we mentioned in OFA paper, the candidate answer set shall be specified by --ans2label-file
# For more details about this option, please refer to issue #123 and PR #124
unconstrained_training_flag=""
# unconstrained_training_flag="--unconstrained-training"

TORCH_DISTRIBUTED_DEBUG=DETAIL
TORCH_SHOW_CPP_STACKTRACES=1

for max_epoch in 4; do
  echo "max_epoch "${max_epoch}
  for warmup_ratio in 0; do
    echo "warmup_updates "${warmup_updates}  
    for lr in 1e-5; do
      echo "lr "${lr}
      for patch_image_size in 480; do
        echo "patch_image_size "${patch_image_size}

        log_file=${log_dir}/${max_epoch}"_"${warmup_ratio}"_"${lr}"_"${patch_image_size}"_rank"${RANK}".log"
        save_path=${save_dir}/${max_epoch}"_"${warmup_ratio}"_"${lr}"_"${patch_image_size}
        mkdir -p $save_path

        python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} --master_port=${MASTER_PORT} ../../train.py \
            ${data} \
            --selected-cols=${selected_cols} \
            --bpe-dir=${bpe_dir} \
            --user-dir=${user_dir} \
            --restore-file=${restore_file} \
            --reset-optimizer --reset-dataloader --reset-meters \
            --save-dir=${save_path} \
            --task=${task} \
            --arch=${arch} \
            --criterion=${criterion} \
            --label-smoothing=${label_smoothing} \
            --batch-size=${batch_size} \
            --update-freq=${update_freq} \
            --encoder-normalize-before \
            --decoder-normalize-before \
            --share-decoder-input-output-embed \
            --share-all-embeddings \
            --layernorm-embedding \
            --patch-layernorm-embedding \
            --code-layernorm-embedding \
            --resnet-drop-path-rate=${resnet_drop_path_rate} \
            --encoder-drop-path-rate=${encoder_drop_path_rate} \
            --decoder-drop-path-rate=${decoder_drop_path_rate} \
            --dropout=${dropout} \
            --attention-dropout=${attention_dropout} \
            --weight-decay=0.01 \
            --optimizer=adam \
            --adam-betas="(0.9,0.999)" \
            --adam-eps=1e-08 \
            --clip-norm=1.0 \
            --lr-scheduler=polynomial_decay \
            --lr=${lr} \
            --end-learning-rate=8e-6 \
            --max-epoch=${max_epoch} \
            --warmup-ratio=${warmup_ratio} \
            --log-format=simple \
            --log-interval=${log_interval} \
            --fixed-validation-seed=7 \
            --keep-last-epochs=10 \
            --save-interval=1 --validate-interval=20 \
            --maximize-best-checkpoint-metric \
            --max-src-length=${max_src_length} \
            --max-tgt-length=${max_tgt_length} \
            --find-unused-parameters \
            --freeze-encoder-embedding \
            --freeze-decoder-embedding \
            ${unconstrained_training_flag} \
            --add-type-embedding \
            --scale-attn \
            --scale-fc \
            --scale-heads \
            --disable-entangle \
            --num-bins=${num_bins} \
            --patch-image-size=${patch_image_size} \
            --fp16 \
            --fp16-scale-window=512 \
            ${store_ema} \
            ${ema_fp32} \
            --ema-decay=${ema_decay} \
            --ema-start-update=${ema_start_update} \
            --num-workers=0
      done
    done
  done
done
