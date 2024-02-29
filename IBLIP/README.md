# MINS on Instruct-BLIP
To run training and testing experiments on Instruct-BLIP, the minimum hardware requirement (recommend) is 4x NVIDIA A6000 48G GPU.

## Set up
Use the provided environment configuration file to create the environment (need conda prerequisite)
```bash
conda env create -f env.yaml
conda activate iblip
```

Download vicuna-7B from source website
```bash
git lfs clone https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main
```

## Training
Check if paths are correct in the following configuration files
  - `lavis/projects/iblip/train/instr_tuning.yaml`: results and log output path
  - `lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml`: pretrained model path for Vicuna-7B
  - `lavis/configs/datasets/instr/defaults_tuning`: textual dataset path and visual input folder

Run training
```bash
export CUDA_VISIBLE_DEVICES='0,1'; bash scripts/train_instr.sh
```

## Evaluation
Check if paths are correct in the files regarding each evaluationt task, then run the evaluation script
```bash
export CUDA_VISIBLE_DEVICES='0,1'; bash scripts/eval_all.sh
```