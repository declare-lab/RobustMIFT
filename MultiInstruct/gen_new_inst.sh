#!/bin/bash

# Single temperaure generation
torchrun --nproc_per_node 2 instruction_gen.py  --ckpt_dir llama/ckpt/Llama-2-13b-chat/  --tokenizer_path llama/ckpt/Llama-2-13b-chat/tokenizer.model --max_seq_len 512 --max_batch_size 2 --instr_dir ./llama/gen_instr.txt --temperature 0.6


# Multi Temperature generation
# for temp in {0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0};do
#     torchrun --nproc_per_node 2 instruction_gen.py  --ckpt_dir llama/ckpt/Llama-2-13b-chat/  --tokenizer_path llama/ckpt/Llama-2-13b-chat/tokenizer.model --max_seq_len 512 --max_batch_size 2 --instr_dir ./llama/gen_instr.txt --temperature $temp
# done
