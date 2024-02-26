#!/usr/bin/env bash

# ckpt_path=../../checkpoints/tuned/checkpoint1.pt
ckpt_path=$1

echo 'now evaluate on instruction: -1'
cd /home/henry/OFA-MMInstr/MultiInstruct
python build_dataset.py --instr_id -1
cd /home/henry/OFA-MMInstr/OFA/run_scripts/instr_eval_mixed

# TODO: Remove this comment
echo 'Now evaluate VTE task...'
bash evaluate_vte.sh $ckpt_path

echo 'Now evaluate Text-VQA task...'
bash evaluate_textvqa.sh $ckpt_path

echo 'Now evaluate Visual Dialogue task...'
bash evaluate_visdial.sh $ckpt_path

echo 'Now evaluate commonsense VQA task...'
bash evaluate_commonsensevqa.sh $ckpt_path

echo 'Now evaluate commonsense VQA MC task...'
bash evaluate_commonsensevqa_mc.sh $ckpt_path

echo 'Now evaluate on disaster classification task...'
bash evaluate_dc.sh $ckpt_path

echo 'Now evaluate on Grounded VQA task...'
bash evaluate_groundedVQA_mc.sh $ckpt_path

echo 'Now evaluate on Visual Entailment task...'
bash evaluate_visnli.sh $ckpt_path

echo 'Now evaluate on Commonsense Reasoning task...'
bash evaluate_vsr.sh $ckpt_path

echo 'Now evaluate on NLVR task...'
bash evaluate_nlvr.sh $ckpt_path
# done

