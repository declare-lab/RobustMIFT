# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama

def get_inputs():
    # multiple outputs
    # prefix = 'Give a revision of the following sentence: '

    # 
    prefix = 'Revise the following setence. '

    raw_sents = [
        "I believe the meaning of life is to find the happiness that comes from within. ",
        "The speed of light is constant.",
        "I'm really excited to see the site live and it looks amazing",
        "I just wanted to say congrats on the launch!"
    ]
    inputs = [prefix + sent for sent in raw_sents]
    return inputs

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    inputs = get_inputs()
    results = generator.text_completion(
        inputs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(inputs, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
