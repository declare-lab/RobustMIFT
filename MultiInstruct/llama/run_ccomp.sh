#!/bin/bash

torchrun --nproc_per_node 2 my_chat_completion.py  --ckpt_dir ckpt/Llama-2-13b-chat/  --tokenizer_path ckpt/Llama-2-13b-chat/tokenizer.model --max_seq_len 512 --max_batch_size 2 --instr_dir gen_instr.txt

