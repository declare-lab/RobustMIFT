# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import re

from .llama import Llama
from typing import Optional


def main(
    ckpt_dir: str, tokenizer_path: str, temperature: float = 0.75,
    top_p: float = 0.9, max_seq_len: int = 512, max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs = [
        [
            {"role": "system", "content": 'You are a good writer.'},
            {"role": "user", "content": 'Use simpler vocabulary and sentence structures to make the text more accessible and easier to understand. Keep the contents in "{}" (including "{}") unchanged. \nFor task {A}, select the immediate next step {B} to the step specified by {C}{A}.'}
        ]
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")
        # with open('gen_instr.txt', 'w') as f:
        #     f.write(result['generation']['content'])

    
class Generator(object):
    def __init__(
        self, ckpt_dir: str, tokenizer_path: str, temperature: float = 0.6,
        top_p: float = 0.9, max_seq_len: int = 1024, max_batch_size: int = 4,
        max_gen_len: Optional[int] = None, instr_dir: str = None
    ):
        self.gen_param=dict(
            temperature=temperature, top_p=top_p, max_gen_len=max_gen_len
        )
        self.gen_instrs = [line for line in open(instr_dir, 'r')][:5]
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

    def preprocess(self, sent):
        regex = re.compile('{.*?}')
        res = regex.findall(sent)
        rep_dict = dict()
        for i, r in enumerate(res):
            rep_token = '{' + chr(ord('A')+i) + '}'
            sent = sent.replace(res[i], rep_token)
            rep_dict[res[i]] = rep_token 
        return sent, rep_dict

    def postprocess(self, sent, rep_dict):
        for k, v in rep_dict.items():
            sent = sent.replace(v, k)
        return sent

    def display(self, dialogs, results):
        for dialog, result in zip(dialogs, results):
            print('=' * 50 + 'Segment Line' + '=' * 50)
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )

    def construct_prompt(self, instr):
        instr_list = instr.split(':')
        instr_list[0] = instr_list[0] + ' for the input text'
        return ':'.join(instr_list)

    def generate(self, task_instr: str = None):
        gen_results = []
        
        human_instr = 'Rephrase the input sentence. '
        if re.findall('{.*?}', task_instr): # gen_instr has format string
            # human_instr += 'Keep the content within "{}" unchanged.'
            task_instr, rep_dict = self.preprocess(task_instr)
            preprocessed = True
        else:
            preprocessed = False
        
        for gen_instr in self.gen_instrs:
            sys_prmpt = self.construct_prompt(gen_instr)
            dialogs = [
                [
                    {"role": "system", "content": human_instr},
                    {"role": "user", "content": task_instr}
                ],
                [
                    {"role": "system", "content": sys_prmpt},
                    {"role": "user", "content": task_instr}
                ]
            ]

            results = self.generator.chat_completion(
                dialogs,  # type: ignore
                **self.gen_param
            )
            # self.display(dialogs, results)
            for result in results:
                res_content = result['generation']['content'].split('\n')
                if len(res_content) >= 2:
                    res = res_content[2].strip('"')
                else:
                    res = res_content[0]

                if preprocessed:
                    res = self.postprocess(res, rep_dict)

                if res in gen_results or res == task_instr:
                    continue
                gen_results.append(res)

        return gen_results

def run(ckpt_dir: str, tokenizer_path: str, temperature: float = 0.6,
    top_p: float = 0.9, max_seq_len: int = 512, max_batch_size: int = 4,
    max_gen_len: Optional[int] = None, instr_dir: str = None):

    instr = 'abcde'
    generator = Generator(
        ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, temperature=temperature, top_p=top_p, max_seq_len=max_seq_len, max_batch_size=max_batch_size, max_gen_len=max_gen_len, instr_dir=instr_dir)
    
    generator.generate(instr)

if __name__ == "__main__":
    fire.Fire(run)
