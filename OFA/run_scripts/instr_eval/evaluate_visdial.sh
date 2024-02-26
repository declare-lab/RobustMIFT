©#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1081
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

data=../../dataset/test/visual_dialog.tsv
# path=../../checkpoints/ofa_large.pt
path=$1
result_path=../../results/visdial
selected_cols=0,3,4,7,8
split='test'

for iid in {0..4};do
    echo 'now evaluate on instruction' $iid
    cd /home/henry/OFA-MMInstr/MultiInstruct
    python build_dataset.py --instr_id $iid
    cd /home/henry/OFA-MMInstr/OFA/run_scripts/instr_eval
    python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} ../../evaluate.py \
        ${data} \
        --path=${path} \
        --user-dir=${user_dir} \
        --task='visdial' \
        --batch-size=8 \
        --selected-cols=${selected_cols} \
        --bpe-dir=${bpe_dir} \
        --patch-image-size=480 \
        --zero-shot \
        --log-format=simple --log-interval=10 \
        --seed=7 \
        --gen-subset=${split} \
        --results-path=${result_path} \
        --fp16 \
        --beam=5 \
        --unnormalized \
        --temperature=1.0 \
        --num-workers=0
done