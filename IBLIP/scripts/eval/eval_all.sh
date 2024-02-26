#!/usr/bin/env bash

ckpt_path=$1

echo 'Now evaluate VTE task...'
bash evaluate_vte.sh $ckpt_path

echo 'Now evaluate Text-VQA task...'
bash evaluate_tvqa.sh $ckpt_path

echo 'Now evaluate Visual Dialogue task...'
bash evaluate_visdial.sh $ckpt_path

echo 'Now evaluate commonsense VQA task...'
bash evaluate_commonsensevqa.sh $ckpt_path

echo 'Now evaluate on disaster classification task...'
bash evaluate_dc.sh $ckpt_path

echo 'Now evaluate on Grounded VQA task...'
bash evaluate_gvqa.sh $ckpt_path

echo 'Now evaluate on Visual Entailment task...'
bash evaluate_visnli.sh $ckpt_path

echo 'Now evaluate on Visual Spatial Reasoning task...'
bash evaluate_vsr.sh $ckpt_path

echo 'Now evaluate on NLVR task...'
bash evaluate_nlvr.sh $ckpt_path