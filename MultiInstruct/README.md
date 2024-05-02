# InstrAug
This folder contains the code for the entire _InstrAug_ pipeline, which includes generation, post-processing (filtering) and dataset reconstruction. During generation stage, we "ask" LLaMA2-Chat-13B to generate augmented instructions from original ones.

## Set up
1. Create the environment to run llama using the provided configuration file.
```bash
conda env create -f env.yaml
conda activate llama
```

2. Download llama2 checkpoint from this [link](https://huggingface.co/models?search=llama2) (require application for access on [Meta website](https://llama.meta.com/llama-downloads/)). Put the checkpoint file in the folder
```bash
mkdir -p ./llama/ckpt/

cd llama/ckpt
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-13b-chat
```

3. Download and preprocess MINS dataset following [README_MINS.md](README_MINS.md)

## How to Run
### Generation and Post-processing
_Guiding Instructions_ are already included in `llama/gen_instr.txt`. You can also use customized guiding instructions by replacing the original content with them. 
Simply run the generation script (we use 2 NVIDIA RTX A6000 48GB GPU in this step). 

```bash
CUDA_VISIBLE_DEVICES='0,1' bash gen_new_inst.sh 
```
The script first saves raw instructions into `RAW_FILE`, then generate into `GEN_TRG_FILE`. 
The process next filter instructions in `SRC_FILE` according to predefined rules to `TRG_FILE`.
You must specified the filename in `instruction_gen.py` before generation.

### Build dataset
Run the following command to build instructions with augmented instructions. You should specify the number of instances per task (`SIZE`) and filtered instruction version.
```bash
python build_new_dataset.py
```
