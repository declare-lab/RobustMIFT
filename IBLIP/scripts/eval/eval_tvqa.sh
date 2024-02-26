#!/bin/bash

for iid in {0..4};do
    rm -rf /home/henry/OFA-MMInstr/LAVIS/lavis/output/BLIP2/iblip_vte7b/*
    echo 'now evaluate on instruction' $iid
    cd /home/henry/OFA-MMInstr/MultiInstruct
    python build_dataset.py --instr_id $iid

    cd /home/henry/OFA-MMInstr/LAVIS
    python -m torch.distributed.run --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/iblip/eval/tvqa_eval.yaml
done