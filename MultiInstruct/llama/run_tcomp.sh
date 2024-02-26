#!/bin/bash

torchrun --nproc_per_node 8 example_text_completion.py  --ckpt_dir ckpt/Llama-2-13b/  --tokenizer_path ckpt/Llama-2-13b/tokenizer.model --max_seq_len 128 --max_batch_size 4

# torchrun --nproc_per_node 2 my_text_completion.py  --ckpt_dir ckpt/Llama-2-13b/  --tokenizer_path ckpt/Llama-2-13b/tokenizer.model --max_seq_len 128 --max_batch_size 4
