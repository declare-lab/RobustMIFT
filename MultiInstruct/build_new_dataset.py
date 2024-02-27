import os
import json
import pandas as pd
import glob
# from instruction_templates import build_instruction
from tqdm import tqdm
from pathlib import Path
from os.path import exists
from instruction_gen import build_with_new_instructions, build_with_new_instructions_sample
from util import process_region


OUTPUT_REGION_TASK = {
    "detection", "VG", "object_region_selection", "region_generation","pointing_grounded_VQA","descriptive_object_region_generate", "text_localization", "visual_object_region", "visual_subject_region", "descriptive_object_region_select", "VG_selection", "grounded_VQA", "select_overlap_most_region", "select_overlap_least_region", "select_overlaped_region", "select_nonoverlaped_region"
}

OUTPUT_IMAGE_CODE_TASK = {"image_generation", "infilling", "im_region_extraction", "im_descriptive_infilling", "image_completion",  "image_completion_w_image_caption", "image_completion_w_region_caption", "im_descriptive_extraction"}

NO_IMAGE_AS_INPUT = {'image_generation'}

OPTIONS_REGION_TASK = {
    "object_region_selection","pointing_grounded_VQA", "text_localization", "descriptive_object_region_select","VG_selection", "grounded_VQA", "select_overlap_most_region", "select_overlap_least_region","select_overlaped_region", "select_nonoverlaped_region"
}

META_REGION_TASK = {
    "visual_answer_justification", "commonsense_VQA", "visual_object_region", "visual_subject_region", "select_overlap_most_region", "select_overlap_least_region", "select_overlaped_region", "select_nonoverlaped_region", "if_region_overlap"
}

MISSING_TASK = {'VQA_absurd','region_area'}

SIZE='1K'
VERSION='1'
INSTR_SRC_FILE=f'./instruction_data/gen_instructions_v{VERSION}.tsv'
TRG_ROOT=f'./mminstr_dataset/augv{VERSION}_{SIZE}'
FILE_NAME='train'
TRG_FILE=os.path.join(TRG_ROOT, '{}.jsonl'.format(FILE_NAME))
TRG_TSV_FILE=os.path.join(TRG_ROOT, '{}.tsv'.format(FILE_NAME))

def gen_train_jsonl():
    data_frame = pd.read_csv(INSTR_SRC_FILE, sep='\t')
    with open(TRG_FILE,'w') as fout:
        all_task = glob.glob('training_data_{}/*/*'.format(SIZE),recursive=True)
        for i, file_name in enumerate(all_task):
            assert '.json' in file_name
            if not 'train.jsonl' in file_name:
                continue
            with open(file_name,'r') as fin:
                print(f"Processing Train Data on {file_name.split('/')[-2]} dataset... ({i+1}/{len(all_task)})")
                for l in tqdm(list(fin)):
                    line = json.loads(l)
                    image_path = line['image_path']
                    image_path = os.path.abspath(image_path)
                    # assert exists(image_path)
                    task = line['task_name']
                    
                    if task in OUTPUT_IMAGE_CODE_TASK or task in MISSING_TASK: # next version
                        continue
                    
                    if task in OUTPUT_REGION_TASK:
                        line['region'] = [process_region(r) for r in line.get('region')]
                        line['target_txt'] = ' '.join(line['region'])
                    elif line.get('region') is not None:
                        line['region'] = [process_region(r) for r in line.get('region')]
                    if task in OPTIONS_REGION_TASK:
                        line['options'] = [process_region(r) for r in line.get('options')]
                    if task in META_REGION_TASK:
                        for k in line['meta_data']['object_regions']:
                            meta_regions = line['meta_data']['object_regions'][k]
                            line['meta_data']['object_regions'][k] = [process_region(r) for r in meta_regions]

                    prompt, target = build_with_new_instructions(data_frame,task, text=line.get('text'), options=line.get('options'), region=line.get('region'), context=line.get('context'), question=line.get('question'), explanation=line.get('explanation'), response=line.get('response'), premise=line.get('premise'), hypothesis=line.get('hypothesis'),answer=line.get('answer'), meta_data=line.get('meta_data'), target=line.get('target_txt'))

                    line['prompt'] = prompt
                    line['target'] = target
                    fout.write(json.dumps(line)+'\n')

def gen_train_jsonl_with_sample():
    data_frame = pd.read_csv(INSTR_SRC_FILE, sep='\t')
    with open(TRG_FILE,'w') as fout:
        all_task = glob.glob(f'training_data_{SIZE}/*/*', recursive=True)
        for i, file_name in enumerate(all_task):
            assert '.json' in file_name
            if not 'train.jsonl' in file_name:
                continue
            with open(file_name,'r') as fin:
                print(f"Processing Train Data on {file_name.split('/')[-2]} dataset... ({i+1}/{len(all_task)})")
                for l in tqdm(list(fin)):
                    line = json.loads(l)
                    image_path = line['image_path']
                    image_path = os.path.abspath(image_path)
                    # assert exists(image_path)
                    task = line['task_name']
                    
                    if task in OUTPUT_IMAGE_CODE_TASK or task in MISSING_TASK: # next version
                        continue
                    
                    if task in OUTPUT_REGION_TASK:
                        # print(line)
                        line['region'] = [process_region(r) for r in line.get('region')]
                        line['target_txt'] = ' '.join(line['region'])
                    elif line.get('region') is not None:
                        line['region'] = [process_region(r) for r in line.get('region')]
                    if task in OPTIONS_REGION_TASK:
                        line['options'] = [process_region(r) for r in line.get('options')]
                    if task in META_REGION_TASK:
                        for k in line['meta_data']['object_regions']:
                            meta_regions = line['meta_data']['object_regions'][k]
                            line['meta_data']['object_regions'][k] = [process_region(r) for r in meta_regions]

                    prompt, target = build_with_new_instructions_sample(data_frame,task, text=line.get('text'), options=line.get('options'), region=line.get('region'), context=line.get('context'), question=line.get('question'), explanation=line.get('explanation'), response=line.get('response'), premise=line.get('premise'), hypothesis=line.get('hypothesis'),answer=line.get('answer'), meta_data=line.get('meta_data'), target=line.get('target_txt'))
                    line['prompt'] = prompt
                    line['target'] = target
                    fout.write(json.dumps(line)+'\n')

# transform .jsonl to .tsv file
def jsonl_to_tsv():
    table = pd.read_json(TRG_FILE, lines=True)
    table.to_csv(TRG_TSV_FILE, sep='\t', index=False)

def shuffle_json(num_split=4):
    table = pd.read_csv(os.path.join(TRG_ROOT, 'train.tsv'), sep='\t')
    last_table = table
    for i in range(num_split):
        new_split_name = os.path.join(TRG_ROOT, 'train{}.tsv'.format(i))
        new_table = last_table.sample(frac=1).reset_index(drop=True)
        new_table.to_csv(new_split_name, sep='\t', index=False)
        last_table = new_table


def main(args):
    if args.with_prob:
        gen_train_jsonl_with_sample()
    else:
        gen_train_jsonl()
    jsonl_to_tsv()
    # shuffle_json()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--instr_id', type=int, default=-1, help='The id of the instruction to be selected')
    parser.add_argument('--with_prob', action='store_true', help='whether to use specified probability for sampling')
    args = parser.parse_args()
    main(args)