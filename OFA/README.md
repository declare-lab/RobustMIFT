# MINS on OFA

To run training and testing experiments on OFA, the minimum hardware requirement (recommend) is 4x NVIDIA A6000 48G GPU.

## Set up
Use the provided environment configuration file to create the environment (need conda prerequisite)
```bash
conda env create -f env.yaml
conda activate ofa
```
Create (or soft link) the dataset and checkpoint folder to save MINS dataset and saved model.
```bash
mkdir dataset
mkdir checkpoints
```
Download pretrained OFA model to `./checkpoints` from [OFA-ModelCard](README_OFA.md#model-card).

## Training
You can directly start training using the default script, make sure that you have pretrained models and dataset prepared and the directories in the script are correct.
```bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'; bash run_scripts/instr_train/train_instr_distributed.sh
```
Here we assume you have a server of 4 GPUs, and use all of them for training.

## Evaluation
Start evaluation using the provided script
```bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'; bash run_scripts/instr_eval/evaluate_all.sh [path_to_checkpoint]
```
