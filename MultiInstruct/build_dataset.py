import os
import json
import pandas as pd
import glob
import re
from instruction_templates import build_instruction
from tqdm import tqdm
from pathlib import Path
from os.path import exists
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
SAMPLES_PER_TASK = 1000

def convert_name(fname):
    return re.sub('\.(jpg|png|jpeg|jfif)$', '.txt', fname)

def gen_train_jsonl_raw():
    output = []
    with open('./train.jsonl','w') as fout:
        for file_name in glob.glob('training_data_1K/*/*',recursive=True):
            assert '.json' in file_name
            if not 'train.jsonl' in file_name:
                continue
            with open(file_name,'r') as fin:
                print('Processing Train Data on {} dataset...'.format(file_name.split('/')[-2]))
                for l in tqdm(list(fin)):
                    line = json.loads(l)
                    image_path = line['image_path']
                    if 'VG_100K_2' in image_path:
                        import re
                        image_path = re.sub('VG_100K_2', 'VG_100K', image_path)
                        print(image_path)
                    image_path = os.path.abspath(image_path)
                    assert exists(image_path) or exists(convert_name(image_path))
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

                    prompt, target = build_instruction(task, text=line.get('text'), options=line.get('options'), region=line.get('region'), context=line.get('context'), question=line.get('question'), explanation=line.get('explanation'), response=line.get('response'), premise=line.get('premise'), hypothesis=line.get('hypothesis'),answer=line.get('answer'), meta_data=line.get('meta_data'), target=line.get('target_txt'))
                    line['prompt'] = prompt
                    line['target'] = target
                    fout.write(json.dumps(line)+'\n')

def gen_test_jsonl(instruction_id=-1):
    for file_name in glob.glob('testing_data/*/*',recursive=True):
        # task name
        # if 'SciQA' in file_name or 'icon' in file_name or 'GQA' in file_name or 'VizWiz' in file_name or 'MMMU' in file_name:
        import re
        if re.match('(.+)(SciQA|icon|GQA|VizWiz|MMMU)(.*)', file_name):
            continue
        task_name = file_name.split('/')[-2]
        with open(os.path.join('./mminstr_dataset/test', f'{task_name}.jsonl'),'w') as fout:
            assert '.json' in file_name
            if not 'test.jsonl' in file_name:
                continue
            with open(file_name, 'r') as fin:
                print('Processing Test Data on {} dataset...'.format(file_name.split()[-1]))
                for line in tqdm(list(fin)[:SAMPLES_PER_TASK]):
                    line = json.loads(line)
                    image_path = line['image_path']
                    image_path = os.path.abspath(image_path)
                    # assert exists(image_path)
                    task = line['task_name']
                    
                    if task in OUTPUT_IMAGE_CODE_TASK or task in MISSING_TASK: # next version
                        continue
                    # print(line)
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
                            line['meta_data']['object_regions'][k] = [process_region(meta_regions)]

                    # print(line)
                    prompt, target = build_instruction(task, text=line.get('text'), options=line.get('options'), region=line.get('region'), context=line.get('context'), question=line.get('question'), explanation=line.get('explanation'), response=line.get('response'), premise=line.get('premise'), hypothesis=line.get('hypothesis'), answer=line.get('answer'), meta_data=line.get('meta_data'), target=line.get('target_txt'), instruction_id=instruction_id)
                    line['prompt'] = prompt
                    line['target'] = target
                    fout.write(json.dumps(line)+'\n')

# transform .jsonl to .tsv file
def jsonl_to_tsv(split=['test']):
    if split == 'both': split = ['train', 'test']
    if 'train' in split:
        table = pd.read_json('./train.jsonl', lines=True)
        table.to_csv('./train.tsv', sep='\t', index=False)

    if 'test' in split:
        for file in glob.glob('./mminstr_dataset/test/*.jsonl', recursive=True):
            table = pd.read_json(file, lines=True)
            task_name = file.split('/')[-1].split('.')[0]
            table.to_csv(f'./mminstr_dataset/test/{task_name}.tsv', sep='\t', index=False)

def main(args):
    if args.split == 'train':
        gen_train_jsonl_raw()
        jsonl_to_tsv(split=['train'])
    else:
        gen_test_jsonl(args.instr_id)
        jsonl_to_tsv(split=['test'])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--instr_id', type=int, default=-1, help='The id of the instruction to be selected')
    parser.add_argument('--split', type=str, default='test', help='The split that ')
    args = parser.parse_args()
    main(args)