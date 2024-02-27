import os
import string
import random
import re, fire
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from typing import Optional
from myllama.my_chat_completion import Generator

random.seed(7)

RAW_FILE = './instruction_data/raw_instructions.tsv'
GEN_TRG_FILE = './instruction_data/gen_instructions.tsv'
SRC_FILE = './instruction_data/gen_instructions.tsv'  # for v2 10K
TRG_FILE = './instruction_data/gen_instructions_filtered.tsv'
# MT_TGT_FILE = './instruction_data/gen_instructions_temp{temperature}.tsv'

TEST_TASKS = ['visual_spatial_reasoning', 'visual_text_extraction', 'text_vqa', 'medic_disaster_types', 'grounded_VQA','visual_nli','visual_spatial_reasoning','natural_language_visual_reasoning']

def save_raw_instruction(use_natural=False, instruction_id=-1):
    image_token = "image" # this show only appear before the output token
    options_token = "\n\n[Options]:"
    region_token = "Regions: "
    split_token = "||||" # or 
    region_split_token = "||||" # or
    
    df = pd.DataFrame()
    df_dicts = list()
    # --------------------------- training tasks ---------------------------
    # image_caption
    instructs=[
        """In this task, you will look at the image and briefly describe the image.""",
        """What is the caption of the image?""",
        """Generate some text to describe the image.""",
        """Look at image and tell me what is the content.""",
        """In this task, you are given an image and you will need to generate some text to describe it."""
    ]
    df_dicts.extend(dict(task_name='image_caption', instructions=instruct) for instruct in instructs)
    
    # open-domain_VQA
    instructs = [
        """{question}""",
        """{question}""",
        """{question}""",
        """{question}""",
        """{question}"""
    ]
    df_dicts.extend(dict(task_name='open-domain_VQA', instructions=instruct) for instruct in instructs)

    # VQA
    instructs = [
        "{question}{options_token} {split_token. join(options)}",
        "{question}{options_token} {split_token. join(options)}",
        "{question}{options_token} {split_token. join(options)}",
        "{question}{options_token} {split_token. join(options)}",
        "{question}{options_token} {split_token. join(options)}"]
    df_dicts.extend(dict(task_name='VQA', instructions=instruct) for instruct in instructs)

    # GC
    instructs = [
        """The goal of this task is to generate description for one part of the image. The part is specified by {region_split_token.join(region)}.""",
        """What is the content of {region_split_token.join(region)}?""",
        """Describe the content of {region_split_token.join(region)} in image.""",
        """Generate a caption for {region_split_token.join(region)}.""",
        """{region_split_token.join(region)} is a region in image. Locate the region first and generate a description for that part of image.""",
    ]
    df_dicts.extend(dict(task_name='GC', instructions=instruct) for instruct in instructs)
    
    # GC Selection
    instructs = [
        """Select the description for one part of the image. The part is specified by {region_split_token.join(region)}.{options_token} {split_token.join(options)}""",
        """What is the content of {region_split_token.join(region)}?{options_token} {split_token.join(options)}""",
        """Select the content of {region_split_token.join(region)} from options.{options_token} {split_token.join(options)}""",
        """What is the caption for {region_split_token.join(region)}?{options_token} {split_token.join(options)}""",
        """{region_split_token.join(region)} is a region in image. Select a description for that part of image.{options_token} {split_token.join(options)}""",
    ]
    df_dicts.extend(dict(task_name='GC_selection', instructions=instruct) for instruct in instructs)
    
    # VG
    instructs = [
        """The region in image that \"{text}\" describes is""",
        """Find the region in image that \"{text}\" describes.""",
        """The goal of this task is to find the part of the image with the description: \"{text}\"""",
        """ \"{text}\" describes part of the image. Find the part.""",
        """In this task, you are asked to localize the region in image that is described by the given text. The text is \"{text}\""""
    ]
    df_dicts.extend(dict(task_name='VG', instructions=instruct) for instruct in instructs)

    # VG_selection
    instructs = [
        """Select region in the image that \"{text}\" describes.{options_token} {split_token.join(options)}""",
        """What is the region in the image that \"{text}\" describes?{options_token} {split_token.join(options)}""",
        """The goal of this task is to select the region of the image with the description: \"{text}\"{options_token} {split_token.join(options)}""",
        """ \"{text}\" describes part of the image. Find the part.{options_token} {split_token.join(options)}""",
        """In this task, you are asked to localize the region in image that is described by the given text. The text is \"{text}\"{options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='VG_selection', instructions=instruct) for instruct in instructs)
    
    # object_grounding
    instructs = [
        """What is the object in {region_split_token.join(region)}""",
        """Identify the object in {region_split_token.join(region)}.""",
        """The goal of this task is to identify the object in given regions in image. The region is {region_split_token.join(region)}. What is the object?""",
        """The object contained in {region_split_token.join(region)} is""",
        """In this task, you are given the coordinates of some rectangular region in the image. You need to first localize each rectangular region and then identify what is the object in the region. The region is {region_split_token.join(region)}."""
    ]
    df_dicts.extend(dict(task_name='object_grounding', instructions=instruct) for instruct in instructs)

    # object_region_match
    instructs = [
        """Is the object \"{text}\" in {region_split_token.join(region)}? {options_token} {split_token.join(options)}""",
        """Does the region {region_split_token.join(region)} contain \"{text}\"? {options_token} {split_token.join(options)}""",
        """Answer if the region {region_split_token.join(region)} contains \"{text}\". {options_token} {split_token.join(options)}""",
        """In this task, you will need to decide if the object in {region_split_token.join(region)} is \"{text}\". {options_token} {split_token.join(options)}""",
        """Decide if the object in {region_split_token.join(region)} matches \"{text}\". {options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='object_region_match', instructions=instruct) for instruct in instructs)
    
    # elif task == 'object_match':
    instructs = [
        """Are the object in {region[0]} and object in {region[1]} the same type? {options_token} {split_token.join(options)}""",
        """In this task you are given two objects. Each object is specified by its location in the image. One object is in {region[0]} and another object is in {region[1]}. Decide if two objects have the same type. {options_token} {split_token.join(options)}""",
        """The goal of this task is to check if two regions contain the same type of object in the image. The two regions are {region_split_token.join(region)}. {options_token} {split_token.join(options)}""",
        """Do objects in {region_split_token.join(region)} have the same type? {options_token} {split_token.join(options)}""",
        """Determine whether the same kind of object is present in both given regions of the image. The two regions are {region_split_token.join(region)}. {options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='object_match', instructions=instruct) for instruct in instructs)
        
    # elif task == 'question_image_match':
    instructs = [
        """In this task, you need to decide if the image has enough information to answer \"{question}\" {options_token} {split_token.join(options)}""",
        """Given content of image, do you have enough information to answer \"{question}\" {options_token} {split_token.join(options)}""",
        """In this task, you are given the question \"{question}\" and you need to decide if the image provide you enough info to answer the question. {options_token} {split_token.join(options)}""",
        """Is it possible to answer \"{question}\" given the content of image? {options_token} {split_token.join(options)}""",
        """Does the image contain the answer to \"{question}\"? {options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='question_image_match', instructions=instruct) for instruct in instructs)
    
    # elif task == 'object_region_selection':
    instructs = [
        """Select the region containing \"{text}\".{options_token} {split_token.join(options)}""",
        """What is the regions in the options that contain \"{text}\"?{options_token} {split_token.join(options)}""",
        """Which option contains \"{text}\"?{options_token} {split_token.join(options)}""",
        """Select the option that contains the object \"{text}\".{options_token} {split_token.join(options)}""",
        """You are given regions as options and select the option that contains the object \"{text}\".{options_token} {split_token.join(options)}""",
    ]
    df_dicts.extend(dict(task_name='object_region_selection', instructions=instruct) for instruct in instructs)

    # modify
    # elif task == 'missing_object_selection':
    instructs = ["""Select objects that do not appear in any of {region_split_token.join(region)}. Select "None" if you can't find any.{options_token} {split_token.join(options)}""",
                    """Select options that do not appear in any of {region_split_token.join(region)}.{options_token} {split_token.join(options)}""",
                    """Given {region_split_token.join(region)}, select objects that do not appear in any of the regions. Select "None" if you can't find it.{options_token} {split_token.join(options)}""",
                    """Which objects in options do not in appear in any of {region_split_token.join(region)}? Select "None" if you can't find it.{options_token} {split_token.join(options)}""",
                    """In this task, you are given some regions {region_split_token.join(region)}. Decide which object in options that do not appear in any of the given region.{options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='missing_object_selection', instructions=instruct) for instruct in instructs)

    # elif task == 'ITM':
    instructs = ["""Does \"{text}\" describes image? {options_token} {split_token.join(options)}""",
                    """Does the text: \"{text}\" and the content of image match? {options_token} {split_token.join(options)}""",
                    """Is the text: \"{text}\" the caption of image? {options_token} {split_token.join(options)}""",
                    """In this task you are given some text and you need to decide if the text describe the image. {options_token} {split_token.join(options)}""",
                    """Is the caption of image \"{text}\"? {options_token} {split_token.join(options)}""",
    ]
    df_dicts.extend(dict(task_name='ITM', instructions=instruct) for instruct in instructs)

    # modify    
    # elif task == 'region_object_selection': 
    instructs = ["""Select objects from the options that appear in at least one of the regions. Select "None" if you can't find it.{region_token} {region_split_token.join(region)}. {options_token} {split_token.join(options)}""",
                    """Given objects in the options, select options that appear in at least one of {region_split_token.join(region)}.Select "None" if you can't find any.{options_token} {split_token.join(options)}""",
                    """What are the objects in the options that appear in at least one of the regions: {region_split_token.join(region)}?{options_token} {split_token.join(options)}""",
                    """Given {region_token} {region_split_token.join(region)}, decide which object appears in at least one of the region.{options_token} {split_token.join(options)}""",
                    """Given some regions, select object that appears in at least one of the region. {region_token} {region_split_token.join(region)}{options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='region_object_selection', instructions=instruct) for instruct in instructs)
    
    # elif task == 'region_generation':
    instructs = ["""What are the regions contain the object \"{text}\"?""",
                    """Given object: \"{text}\", what are the regions that contain this objects?""",
                    """The regions that contain \"{text}\" are""",
                    """The parts of image that have \"{text}\" are""",
                    """Identify the regions that contain \"{text}\".""",
                    """In this task, you are asked to identify all the regions in the image that contain the object \"{text}\".""",
                    """Which parts of image contain \"{text}\"?"""
                    ]
    df_dicts.extend(dict(task_name='region_generation', instructions=instruct) for instruct in instructs)
                
    # elif task == 'region_caption_match':
    instructs = ["""Decide if \"{text}\" is the description of {region_split_token.join(region)}. {options_token} {split_token.join(options)}""",
                    """Does \"{text}\" matches the content of {region_split_token.join(region)}. {options_token} {split_token.join(options)}""",
                    """In this task, you need to decide if \"{text}\" is a caption of {region_split_token.join(region)} in the image. {options_token} {split_token.join(options)}""",
                    """Can \"{text}\" describe {region_split_token.join(region)}? {options_token} {split_token.join(options)}""",
                    """Does {region_split_token.join(region)} and given text match? Text: {text} {options_token} {split_token.join(options)}"""
                    ]
    df_dicts.extend(dict(task_name='region_caption_match', instructions=instruct) for instruct in instructs)

    # elif task == 'object_relationship':
    instructs = [
        """In this task, you are given the regions of a subject A and an object B in the image. Determine what is their relationship. The relationship can be the position of subject A relative to object B or what is subject A doing to object B. Region of subject A is {region[0]} and region of object B is {region[1]}.""",
        """What is the relationship between the subject in {region[0]} and object in {region[1]}?""",
        """Given a subject in {region[0]} and an object in {region[1]}, what's their relationship?""",
        """Subject A: {region[0]} Object B: {region[1]} and their relationship is""",
        """Tell me the relationship between the subject in {region[0]} and the object in {region[1]}."""
    ]
    df_dicts.extend(dict(task_name='object_relationship', instructions=instruct) for instruct in instructs)

    # elif task == 'visual_object_identification':
    instructs = [
        """Given the image, the subject in {region[0]} {meta_data['relation']} what?""",
        """Given the image, the subject in {region[0]} {meta_data['relation']} an object. What is the object?""",
        """Given the subject in {region[0]} and relationship \"{meta_data['relation']}\". What is the object?""",
        """Identify the name of the object, given the subject in {region[0]} and relationship: {meta_data['relation']}. """,
        """In this task, you are asked to identify the object given tne region of the subject in the image and their relationship. The subject is in {region[0]} and relationship is {meta_data['relation']}. The object is""",
    ]
    df_dicts.extend(dict(task_name='visual_object_identification', instructions=instruct) for instruct in instructs)
    
    # elif task == 'visual_subject_identification':
    instructs = [
        """Given the image and the object in {region[1]}, predict what is the subject {meta_data['relation']} the object?""",
        """Given the object in {region[1]}, and the relationship {meta_data['relation']}. What is the subject.""",
        """Identify the subject that {meta_data['relation']} the object.\nThe object is in {region[1]}""",
        """Which subject in the image that has {meta_data['relation']} with the object in {region[1]}""",
        """In this task, you are given the region of the object and the relation. What is the name of the subject? \n\nRelationship: {meta_data['relation']}\nObject: {region[1]}""",
        
    ]
    df_dicts.extend(dict(task_name='visual_subject_identification', instructions=instruct) for instruct in instructs)
    
    # elif task == 'visual_object_region':
    # region=  region_split_token.join(meta_data['object_regions']['subject'])
    instructs = [
        """Which object has the relationship \"{meta_data['relation']}\" with the subject in {region}? Answer the question by generating the region of the object.""",
        """Find the region of the object that has the relationship \"{meta_data['relation']}\" with the subject in {region}.""",
        """Given the image, where is the object that has the relatipnship \"{meta_data['relation']}\" with the the subject in {region}?""",
        """Identify the region of the object given the subject in {region} and relationship \"{meta_data['relation']}\".""",
        """What is the object region, given subject in {region} and relationship \"{meta_data['relation']}\"?""",
        """What is the object region, given the subject region and the relationship?\n\nSubject region: {region} Relationship: \"{meta_data['relation']}\"?"""
    ]
    df_dicts.extend(dict(task_name='visual_object_region', instructions=instruct) for instruct in instructs)

    # elif task == 'visual_subject_region':
    # region=  region_split_token.join(meta_data['object_regions']['object'])
    instructs = [
        """Given the object in {region}, where is the subject in the image that has relationship: \"{meta_data['relation']}\" with the object?""",
        """The object is in {region}. Identify the region of the subject that has relationship: {meta_data['relation']} with the object.""",
        """What is the region of the object, given subject in {region} and relationship \"{meta_data['relation']}\"?""",
        """Subject is in {region} and relationship is \"{meta_data['relation']}\". Generate the region of the object.""",
        """Based on the relationship and the subject, identify the object region. Subject region: {region} Relationship: {meta_data['relation']}"""
    ]
    df_dicts.extend(dict(task_name='visual_subject_region', instructions=instruct) for instruct in instructs)

    # elif task == 'descriptive_object_region_generate':
    instructs = ["""Given the description of an object, generate the region that contains this object. The description is: \"{text}\"""",
                """In this task, you are required to identify the object that is described by \"{text}\" and output the region of that object.""",
                """What is the region of the object described by \"{text}\" in image?""",
                """Where is the object described by \"{text}\"?""",
                """Find the region of {text}""",
    ]
    df_dicts.extend(dict(task_name='descriptive_object_region_generate', instructions=instruct) for instruct in instructs)

    # elif task == 'descriptive_object_region_select':
    instructs = [
                """Given the description of an object, select the region that contains this object.\n\nThe description is: \"{text}\"{options_token} {split_token.join(options)}""",
                """In this task, you are required to identify the object that is described by \"{text}\" and select the region of that object from options.{options_token} {split_token.join(options)}""",
                """What is the region of the object described by \"{text}\" in the picture?{options_token} {split_token.join(options)}""",
                """Select the region of the object described by \"{text}\".{options_token} {split_token.join(options)}""",
                """Given the image, select the region of {text}.{options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='descriptive_object_region_select', instructions=instruct) for instruct in instructs)   
    
    # elif task == 'object_description_generate':
    instructs = ["Generate a sentence to describe the object in the given bounding box. The description should help people to distinguish the object from other objects in the image.\n\nBounding box: {region_split_token.join(region)}",
                    "Describe the object in the given region {region_split_token.join(region)}. The description should be about the location and appearance of the object so it can be distinguished from other object in the image.",
                    "Given the object in {region_split_token.join(region)}, write a sentence to describe it. So it can be easily identified by people.",
                    "Write a sentence to describe the object in the given region.\n\nRegion: {region_split_token.join(region)}",
                    "Write a description of the object in region: {region_split_token.join(region)}. The description should help people to locate the object without causing confusion."
    ]
    df_dicts.extend(dict(task_name='object_description_generate', instructions=instruct) for instruct in instructs)

    # elif task == 'image_quality':
    instructs = ["Select the reason from options to explain why the image quality is bad. {options_token} {split_token.join(options)}",
                    "Explain why the image quality is bad. {options_token} {split_token.join(options)}",
                    "Tell me what is wrong with the image. {options_token} {split_token.join(options)}",
                    "The image quality might be low. Tell me why. {options_token} {split_token.join(options)}",
                    "Select a reason for the bad quality of the image. {options_token} {split_token.join(options)}"
                    ]
    df_dicts.extend(dict(task_name='image_quality', instructions=instruct) for instruct in instructs)

    # elif task == 'text_localization':
    instructs = [
        """Select the region from options that contains the given letters: \"{text}\". {options_token} {split_token.join(options)}""",
        """Determine which region contains the letters: \"{text}\"? {options_token} {split_token.join(options)}""",
        """Select the region that contains the text \"{text}\" {options_token} {split_token.join(options)}""",
        """Which region contains \"{text}\" {options_token} {split_token.join(options)}""",
        """Identify the region that has \"{text}\" written on. {options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='text_localization', instructions=instruct) for instruct in instructs)

    # elif task == 'text_legibility':
    instructs = [
        """Look at the given text region of the {image_token} and decide whether the text in the region is clear and complete. {region_token} {split_token.join(region)} {options_token} {split_token.join(options)}""",
        """Decide if the text in {split_token.join(region)} is clear and complete. {options_token} {split_token.join(options)}""",
        """Decide if the text in the given region is legible. Region {split_token.join(region)} {options_token} {split_token.join(options)}""",
        """In this task, you are given a region which has some text written on it. Tell me if the text on that region is clear. Region {split_token.join(region)} {options_token} {split_token.join(options)}""",
        """Tell me if the text on {split_token.join(region)} is clear and readable. {options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='text_legibility', instructions=instruct) for instruct in instructs)    

    # elif task == 'text_type':
    instructs = [
        """Look at the text in the given region of the {image_token} and determine the type of text in the region from options. {region_token} {split_token.join(region)} {options_token} {split_token.join(options)}""",
        """Read the text in {split_token.join(region)} of the {image_token} and select the type of text from options. {options_token} {split_token.join(options)}""",
        """What type is the text in {split_token.join(region)}? {options_token} {split_token.join(options)}""",
        """The type of the text in {split_token.join(region)} is {options_token} {split_token.join(options)}""",
        """look at the text in {split_token.join(region)} and tell me it's type. {options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='text_type', instructions=instruct) for instruct in instructs)    

    # elif task == 'region_text_match':
    instructs = [
        """Look at the letters in {region_token} {split_token.join(region)} and determine if the letters in the region are the same as \"{text}\". {options_token} {split_token.join(options)}""",
        """Is the text \"{text}\" in {split_token.join(region)}? {options_token} {split_token.join(options)}""",
        """Does {split_token.join(region)} have the letters \"{text}\"? {options_token} {split_token.join(options)}""",
        """Is the text in {split_token.join(region)} the same as \"{text}\"? {options_token} {split_token.join(options)}""",
        """Do the letters in {split_token.join(region)} match \"{text}\"? {options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='region_text_match', instructions=instruct) for instruct in instructs)

    # elif task == 'multimodal_factual_checking':
    instructs = [
        "Deicide if the claim can be supported by the image and the context.\n\nContext: {context}\n\nClaim: \"{text}\"{options_token} {split_token.join(options)}",
        "Context: {context}\nCan the context support \"{text}\"? {options_token} {split_token.join(options)}",
        "{context}\n\nRead previous text and decide if \"{text}\" is factually correct? {options_token} {split_token.join(options)}",
        "Does the context support \"{text}\"?\n\nContext: {context} {options_token} {split_token.join(options)}",
        "Context: {context}\n\nDoes the context support \"{text}\"? {options_token} {split_token.join(options)}"
    ]
    df_dicts.extend(dict(task_name='multimodal_factual_checking', instructions=instruct) for instruct in instructs)  
    
    # elif task == 'wikihow_next_step':
    # context = '\n'.join(context) if len(context)>0 else '\"nothing\"'
    instructs = [
        "For the task {meta_data['method']}, given the history steps {context} and the current step with its corresponding image, what is the next step for this task? The current step is {text}, what is the next step?",
        "What is the next step? You are doing {meta_data['method']} and you have finished\n\n{context}\nYou currently are at the step given by the image and the text \"{text}\". The next step is",
        "You are doing {meta_data['method']}. You have done\n\n{context}\nNow you are at the step described by the image. What is the next step?",
        "The goal is to \"{meta_data['method']}\". Given current step specified by the content of the image and you have finished.\n\n\All previous steps: {context}.\nWhat is the next step?",
        "You are doing {meta_data['method']} and you are at \"{text}\" step. The previous steps you finished are\n\n{context}\nWhat is the next step?",
    ]
    df_dicts.extend(dict(task_name='wikihow_next_step', instructions=instruct) for instruct in instructs)  

    # elif task == 'wikihow_text_image_step_order':
    options = ['next','previous']
    random.shuffle(options)
    instructs = [
        "For the task \"{meta_data['method']}\", given the current step, decide if the content of the image is the next or previous step.\nThe current step is {text}.{options_token} {split_token.join(options)}",
        "Is the image the next or previous step? You are doing \"{meta_data['method']}\" and you are currently at \"{text}\".{options_token} {split_token.join(options)}",
        "The overall goal is to {meta_data['method']}. You are at \"{text}\" step. Is the image the next or the previous step?{options_token} {split_token.join(options)}",
        "The goal is to \"{meta_data['method']}\". Given the current step \"{text}\", Is the picture the next or the previous step?{options_token} {split_token.join(options)}",
        "You are doing {meta_data['method']}. Is the step specified in the picture the next or previous step to \"{text}\"?{options_token} {split_token.join(options)}",
    ]
    df_dicts.extend(dict(task_name='wikihow_text_image_step_order', instructions=instruct) for instruct in instructs)  
    
    # elif task == 'wikihow_image_text_step_order':
    options = ['next','previous']
    random.shuffle(options)
    instructs = [
        "For the task \"{meta_data['method']}\", decide if \"{text}\" is the next or previous step to the step specified by the image.{options_token} {split_token.join(options)}",
        "Is \"{text}\" the next or previous step? You are doing \"{meta_data['method']}\" and you are currently at the step described by the image.{options_token} {split_token.join(options)}",
        "The overall goal is to {meta_data['method']}. You are at the step specified by the content of the image. Is \"{text}\" the next or the previous step?{options_token} {split_token.join(options)}",
        "The goal is to \"{meta_data['method']}\". Given the current step in the picture, Is \"{text}\" the next or the previous step?{options_token} {split_token.join(options)}",
        "You are doing {meta_data['method']}. Is the step \"{text}\" the next or previous step to the step in the image?{options_token} {split_token.join(options)}",
    ]
    df_dicts.extend(dict(task_name='wikihow_image_text_step_order', instructions=instruct) for instruct in instructs)  
       

    # elif task == 'wikihow_immediate_next_step_selection':
    instructs = [
        "For the task \"{meta_data['method']}\", select the immediate next step to the step specified by the image.{options_token} {split_token.join(options)}",
        "You are doing \"{meta_data['method']}\" and you are currently at the step described by the image. What is your next step?{options_token} {split_token.join(options)}",
        "The overall goal is to {meta_data['method']}. You are at the step specified by the content of the image. Select the immediate next step from the options.{options_token} {split_token.join(options)}",
        "The goal is to \"{meta_data['method']}\". Given the current step in the picture, what is the next step?{options_token} {split_token.join(options)}",
        "You are doing {meta_data['method']}. What is the next step to step in the image?{options_token} {split_token.join(options)}",
    ]
    df_dicts.extend(dict(task_name='wikihow_immediate_next_step_selection', instructions=instruct) for instruct in instructs)  

    # elif task == 'image_text_selection':
    instructs = ["""Select the text from options that best describes the image. {options_token} {split_token.join(options)}""",
                    """Which text in the options best describes the image? {options_token} {split_token.join(options)}""",
                    """In this task, you are given some sentences and you need to decide which sentence best matches the image.{options_token} {split_token.join(options)}""",
                    """Which option in the options that is the caption of the image. {options_token} {split_token.join(options)}""",
                    """Select the caption of the image. {options_token} {split_token.join(options)}""",
                    ]
    df_dicts.extend(dict(task_name='image_text_selection', instructions=instruct) for instruct in instructs)        

    # elif task == 'visual_attribute':
    instructs = [
        """Decide which option is the attribute of the object in the given region.\nRegion: {region_split_token.join(region)}{options_token} {split_token.join(options)}""",
        """Select the attribute of the object in {region_split_token.join(region)}{options_token} {split_token.join(options)}""",
        """Given object in {region_split_token.join(region)}, select its attribute.{options_token} {split_token.join(options)}""",
        """Given the region of the object, select its attribute from the options.\n\nRegion: {region_split_token.join(region)}{options_token} {split_token.join(options)}""",
        """Given the bounding box {region_split_token.join(region)} of the object, select its attribute.{options_token} {split_token.join(options)}""",
    ]
    df_dicts.extend(dict(task_name='visual_attribute', instructions=instruct) for instruct in instructs)  

    # image generation tasks
    # elif task == 'infilling':
    instructs = [
        "Fill in the missing part of the image.",
        "Generate the missing part of the image.",
        "Generate masked part of the image.",
        "Generate the part of the image covered by the black square.",
        "Generate the part of the image covered by black.",
    ]
    df_dicts.extend(dict(task_name='infilling', instructions=instruct) for instruct in instructs)  

    # elif task == 'im_region_extraction':
    instructs = [
        "Extract part of the image specified by the given region. Region: {region}.",
        "Extract the part of image in {region}",
        "Generate a copy of the image in the given region {region}.",
        "Output a new image that is identical to the part of the given image specified by {region}",
        "Generate a new image that is a precise replica of the area {region} in the given image.",
    ]
    df_dicts.extend(dict(task_name='im_region_extraction', instructions=instruct) for instruct in instructs)  
    
    # elif task == 'im_descriptive_infilling':
    instructs = [
        "Fill in the missing part of the image based on the description \"{text}\".",
        "Generate the missing part of the image. The caption of the missing part is \"{text}\".",
        "Using the caption \"{text}\" to generate the region of the image covered by black.",
        "Based on the description \"{text}\", generate the masked part in the current image.",
        "Generate the image that fills in the black square in the given image. The description of the black square is \"{text}\".",
    ]
    df_dicts.extend(dict(task_name='im_descriptive_infilling', instructions=instruct) for instruct in instructs)  
    
    # elif task == 'image_completion_w_region_caption':
    instructs = [
        "Fill in the missing part of the image based on the description \"{text}\" and output the whole image.",
        "Base on the caption \"{text}\", fill in the missing part of the image and generate the complete image.",
        "Generate a full version of the given image using the caption to fill in the black area. Caption: {text}.",
        "Create a new image based on the original, with the missing area filled in by \"{text}\".",
        "Generate a complete version of the image with the missing area filled in. The caption of the missing area is \"{text}\"",
    ]
    df_dicts.extend(dict(task_name='image_completion_w_region_caption', instructions=instruct) for instruct in instructs)  

    # elif task == 'image_completion_w_image_caption':
    instructs = [
        "Complete the image based on the description \"{text}\".",
        "Generate an image with description \"{text}\" by filling in the black area in the given image",
        "Use the provided caption to produce a complete image by filling in the black area. Caption: \"{text}\"",
        "Generate a new image that is the same as the given image with the missing area filled. Caption for the new image is \"{text}\".",
        "Use the given caption to generate a new image based on the given image with the masked part filled in. Caption: \"{text}\".",
    ]
    df_dicts.extend(dict(task_name='image_completion_w_image_caption', instructions=instruct) for instruct in instructs)  

    # elif task == 'VQA_activity_recognition':
    instructs = [
        """{question}{options_token} {split_token.join(options)}""",
        """In this task, you will answer a question about the activity of an object in the image. The question is "{question}"{options_token} {split_token.join(options)}""",
        """You are asked about the activity of animals or people in the image. Look at the image and answer "{question}" You should select your answer from the given options.{options_token} {split_token.join(options)}""",
        """Question: {question} Answer the question by first finding the object in the image and identify its activity. The answer is in the options.{options_token} {split_token.join(options)}""",
        """In this task, you will be asked about the activity of some object in the image. Select the best answer from options. Question: {question}{options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='VQA_activity_recognition', instructions=instruct) for instruct in instructs)
  
    # elif task == 'VQA_attribute':
    instructs = [
        """{question}{options_token} {split_token.join(options)}""",
        """In this task, you will be asked a question about the attribute of an object in the image. The question is "{question}"{options_token} {split_token.join(options)}""",
        """Answer the following question about the attribute of an object, "{question}" Select your answer from the given options.{options_token} {split_token.join(options)}""",
        """Question: {question}\n\nAnswer above question by first finding the object in the image and select its attribute from options.{options_token} {split_token.join(options)}""",
        """In this task, you will be asked about the attribute of some object. Select the best answer from given options. Question: {question}{options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='VQA_attribute', instructions=instruct) for instruct in instructs)

    # elif task == 'VQA_color':
    instructs = [
        """{question}{options_token} {split_token.join(options)}""",
        """In this task, you are asked the color of some object in the image. Question: {question}{options_token} {split_token.join(options)}""",
        """Question: {question}\n\nAnswer the above question by first finding the object in the image and then select its color from options,{options_token} {split_token.join(options)}""",
        """Answer {question} based on the image. {options_token} {split_token.join(options)}""",
        """Answer the question: "{question}" based on the color of an object."""
    ]
    df_dicts.extend(dict(task_name='VQA_color', instructions=instruct) for instruct in instructs)
    
    # elif task == 'VQA_counting':
    instructs = [
        """{question}{options_token} {split_token.join(options)}""",
        """In this task, you are asked a question about the number of some objects in the image. The question is: {question}{options_token} {split_token.join(options)}""",
        """The question is: {question} Select your answer from options.{options_token} {split_token.join(options)}""",
        """Question: {question}\n\nPlease answer the question by counting the object mentioned in the question.{options_token} {split_token.join(options)}""",
        """This task tests your ability to count number of objects. Here is the question "{question}". Select the correct answer from options.{options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='VQA_counting', instructions=instruct) for instruct in instructs)
    
    # elif task == 'VQA_object_presence':
    instructs = [
        """{question}{options_token} {split_token.join(options)}""",
        """This task asks you to identify if an object appears in the image. {question}{options_token} {split_token.join(options)}""",
        """In this task, you are required to answer a question about the appearance of an object.{question}{options_token} {split_token.join(options)}""",
        """{question} Decide if the object mentioned in previous question appears in the image.{options_token} {split_token.join(options)}""",
        """Question: {question} look at the image and answer the question.{options_token} {split_token.join(options)}""",
    ]
    df_dicts.extend(dict(task_name='VQA_object_presence', instructions=instruct) for instruct in instructs)
        
    # elif task == 'VQA_object_recognition':
    instructs = [
        """{question}{options_token} {split_token.join(options)}""",
        """In this task you are asked a question about the type of an object in the image. {question}{options_token} {split_token.join(options)}""",
        """In this task, you will answer a question about the subclass of an object in the image. {question}{options_token} {split_token.join(options)}""",
        """In this task, you will be presented with an image. Your task is to answer a question about the type of object. Question: {question}{options_token} {split_token.join(options)}
        """,
        """Please answer a question regarding the type of an object in the image. Question: {question}{options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='VQA_object_recognition', instructions=instruct) for instruct in instructs)
    
    # elif task == 'VQA_positional_reasoning':
    instructs = [
        """{question}{options_token} {split_token.join(options)}""",
        """In this task, you need to analyze the position of objects in an image and answer the following question. {question}{options_token} {split_token.join(options)}""",
        """This task requires an understanding of object location within the presented image. Please select the correct answer to the provided question. {question}{options_token} {split_token.join(options)}""",
        """In this task, the goal is to understand the location of objects within the presented image and provide a answer to the question provided. {question}{options_token} {split_token.join(options)}""",
        """Question: {question}{options_token}\n\n Please answer the question by reasoning about the positions of objects and select an answer from options. {split_token.join(options)}."""
    ]
    df_dicts.extend(dict(task_name='VQA_positional_reasoning', instructions=instruct) for instruct in instructs)
        
    # elif task == 'VQA_scene_recognition':
    instructs = [
        """{question}{options_token} {split_token.join(options)}""",
        """In this task, you need to pay attention to the scene in the image and answer the following question.\n {question}{options_token} {split_token.join(options)}""",
        """Question: {question}{options_token}. \n Please answer the question by analyzing the scene in the provided image. Here are some possible answers. {options_token} {split_token.join(options)}""",
        """Look at the environment in the image and answering the question accordingly.\n {question}{options_token} {split_token.join(options)}""",
        """Given a picture of certain environment, answer the following question by select an answer from the options. \n {question}{options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='VQA_scene_recognition', instructions=instruct) for instruct in instructs)
        
    # elif task == 'VQA_sentiment_understanding':
    instructs = [
        """{question}{options_token} {split_token.join(options)}""",
        """This task requires an understanding of the feeling conveyed in the image. Please select the correct answer to the provided question. {question}{options_token} {split_token.join(options)}""",
        """Question: {question}{options_token} {split_token.join(options)}.\n Please answer the question by interpreting the sentiment in the image.""",
        """Please analyze the sentiment depicted in the image and answer the question.\n {question}{options_token} {split_token.join(options)}""",
        """In this task, you will be asked a question regarding the emotion conveyed in the image. The question is {question}{options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='VQA_sentiment_understanding', instructions=instruct) for instruct in instructs)
        
    # elif task == 'VQA_sport_recognition':
    instructs = [
        """{question}{options_token} {split_token.join(options)}""",
        """In this task, you need to pay attention to the sports depicted in the image and answer the following question. \n {question}{options_token} {split_token.join(options)}""",
        """Given a picture about sports, answer the following question by select an answer from the options. \n {question}{options_token} {split_token.join(options)}""",
        """There are some sports taking place in the image. {question}{options_token} {split_token.join(options)}""",
        """Please answer the following question by analyzing the sport in the given image.\n {question}{options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='VQA_sport_recognition', instructions=instruct) for instruct in instructs)
        
    # elif task == 'VQA_utility_affordance':
    instructs = [
        """{question}{options_token} {split_token.join(options)}""",
        """In this task, you need to pay attention to the possible actions can be taken to the objects in the image and answer the following question. {question}{options_token} {split_token.join(options)}""",
        """Please take a look at the picture and answer the following question by thinking about what each object in the picture can be used for. {question}{options_token} {split_token.join(options)}""",
        """Question: {question}{options_token}\n Please select a correct answer for the question by analyzing the affordance of the objects in the image. {split_token.join(options)}""",
        """This task tests your ability to understand the potential actions that you can take on the objects or the usage of the objects in the image. Here is the question "{question}". Select the correct answer from options.{options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='VQA_utility_affordance', instructions=instruct) for instruct in instructs)
        
    # elif task == 'select_overlap_most_region':
    # TODO: This task needs extra coding to handle
    # given_region = region_split_token.join(meta_data['object_regions']['given_region'])
    instructs = [
        """Given the region {given_region}, decide which region in the options overlaps most with given region.{options_token} {split_token.join(options)}""",
        """Select the region that shares the most common area with {given_region}.{options_token} {split_token.join(options)}""",
        """Which option overlaps most with {given_region}?{options_token} {split_token.join(options)}""",
        """Decide the region that has the most common area with the given region. Region: {given_region}{options_token} {split_token.join(options)}""",
        """Region: {given_region}\n\nIdentify the region overlaps most with the above given region from options.{options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='select_overlap_most_region', instructions=instruct) for instruct in instructs)

    # elif task == 'select_overlap_least_region':
    # TODO: This task needs extra coding to handle
    # given_region = region_split_token.join(meta_data['object_regions']['given_region'])
    instructs = [
        """Given the region {given_region}, decide which region in the options shares the least common area with given region.{options_token} {split_token.join(options)}""",
        """In this task, you are given a region: {given_region}, you need to select a region from the options that has the least overlap with the given region.{options_token} {split_token.join(options)}""",
        """Which option has the least shared area with {given_region}?{options_token} {split_token.join(options)}""",
        """Select the region that has the least overlap with {given_region}.{options_token} {split_token.join(options)}""",
        """Given region: {given_region}, decide which option has the least common area with it.{options_token} {split_token.join(options)}""",
    ]
    df_dicts.extend(dict(task_name='select_overlap_least_region', instructions=instruct) for instruct in instructs)

    # elif task == 'select_overlaped_region':
    # TODO: This task needs extra coding to handle
    # given_region = region_split_token.join(meta_data['object_regions']['given_region'])
    instructs = [
        """Given the region {given_region}, select an overlapping region from options.{options_token} {split_token.join(options)}""",
        """Select a region from options that overlaps with {given_region}{options_token} {split_token.join(options)}""",
        """Which region from options that shares common area with {given_region}?{options_token} {split_token.join(options)}""",
        """Region: {given_region}\n\nSelect a region that has overlap with the given region.{options_token} {split_token.join(options)}""",
        """Which region from options that has common area with {given_region}?{options_token} {split_token.join(options)}""",
    ]
    df_dicts.extend(dict(task_name='select_overlaped_region', instructions=instruct) for instruct in instructs)

    # elif task == 'select_nonoverlaped_region':
    # TODO: This task needs extra coding to handle
    # given_region = region_split_token.join(meta_data['object_regions']['given_region'])
    instructs = [
        """Given the region {given_region}, select an non-overlapping region from options.{options_token} {split_token.join(options)}""",
        """Region: {given_region}, select an non-overlapping region with the given region from options.{options_token} {split_token.join(options)}""",
        """Select an option that does not overlap with {given_region}{options_token} {split_token.join(options)}""",
        """Which option does not share common area with {given_region}?{options_token} {split_token.join(options)}""",
        """Tell me which option does not have shared area with {given_region}?{options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='select_nonoverlaped_region', instructions=instruct) for instruct in instructs)

    # elif task == 'if_region_overlap':
    # TODO: This task needs extra coding to handle
    # given_region = region_split_token.join(meta_data['object_regions']['given_region'])
    instructs = [
        """Given the region {given_region}, decide if {region[0]} overlaps with it.{options_token} {split_token.join(options)}""",
        """Do the following two regions overlap? Region 1: {region[0]} and Region 2: {given_region}{options_token} {split_token.join(options)}""",
        """Does {given_region} share common area with {region[0]}?{options_token} {split_token.join(options)}""",
        """Tell me if {region[0]} and {given_region} have common area.{options_token} {split_token.join(options)}""",
        """Do {region[0]} and {given_region} overlap?{options_token} {split_token.join(options)}"""
    ]
    df_dicts.extend(dict(task_name='if_region_overlap', instructions=instruct) for instruct in instructs)
    
    # ----------------------- testing tasks ----------------------------- #
    
    # elif task == 'visual_nli':
    instructs = [
        """In this task, you are given a short sentence and a image. You need to decide if the content of the image can support the text. Text: {text} {options_str}""",
        """Can you conclude \"{text}\" from the content of image? Select your answer from the options.{options_str}""",
        """Can you conclude \"{text}\" from the content of image?{options_str}""",
        """Can the content of the image support \"{text}\"?{options_str}""",
        """Does the image support the given text?\nText: {text}{options_str}"""
    ]
    df_dicts.extend(dict(task_name='visual_nli', instructions=instruct) for instruct in instructs)

    # elif task == 'natural_language_visual_reasoning':
    instructs = [
        """Does the content of the image support the given text? Text: {text}{options_str}""",
        """Based on the image, is \"{text}\" true?{options_str}""",
        """Decide if the content of the image supports the sentence. Sentence:{text}{options_str}""",
        """\"{text}\"\n\nIs the above text true based on the picture?{options_str}""",
        """Look at the picture and is \"{text}\" true?{options_str}""",
    ]
    df_dicts.extend(dict(task_name='natural_language_visual_reasoning', instructions=instruct) for instruct in instructs)

    # elif task == 'visual_spatial_reasoning':
    instructs = [
        """The text is about the spatial relationship between two objects in the image. If the text is true?\n\nText{text}{options_str}""",
        """In this task, you need to decide if the positional relationship in the text is true, based on the image. The text is: {text}{options_str}""",
        """Does the content of the image support the sentence below?\n\nSentence:{text}{options_str}""",
        """Can the image support \"{text}\"?{options_str}""",
        """Is \"{text}\" true, by referring to the image?{options_str}""",
    ]
    df_dicts.extend(dict(task_name='visual_spatial_reasoning', instructions=instruct) for instruct in instructs)

    # elif task == 'commonsense_VQA':
    # TODO: This task needs extra coding to handle
    # region_info= ' '.join([ f"{k} is in {v[0]}."  for k, v in meta_data['object_regions'].items()])
    # if instruction_id >=2:
    #     for k, v in meta_data['object_regions'].items():
    #         if k in question:
    #             question = question.replace(k, f"the {k} in {' '.join(v)}")

    instructs = [
        """{question} {region_info}{options_str}""",  
        """The region information: {region_info}\nBased on the region information and the image, {question}{options_str}""",
        """{question}{options_str}""",
        """Look at the image and the regions in the question, {question}{options_str}""",
        """Based on the image, answer {question}{options_str}""",
    ]
    df_dicts.extend(dict(task_name='commonsense_VQA', instructions=instruct) for instruct in instructs)

    # ----------------------------------------- VQA
    # elif task == 'text_vqa':
    instructs = [
        """Based on the image and the text on the image, answer the question below.\n\n{question}""",
        """There is some text on the image. Answer {question} based on the text in image.""",
        """Look at the text on image and answer: {question}""",
        """Look at the image and {question}""",
        """{question}""",
    ]
    df_dicts.extend(dict(task_name='text_vqa', instructions=instruct) for instruct in instructs)

    # elif task == 'grounded_VQA':
    instructs = [
        """In this task, you are given a question and you need to identify a region from options in order to answer it. The question is: {question}{options_str}""",
        """Select a region from the options to answer \"{question}\"{options_str}""",
        """Which region can be used to answer \"{question}\"{options_str}""",
        """Which region is the answer to \"{question}\"{options_str}""",
        """{question}{options_token} {split_token.join(options)}""",
    ]
    df_dicts.extend(dict(task_name='grounded_VQA', instructions=instruct) for instruct in instructs)
    
    # elif task == 'ok_vqa':
        # instructs = [
        #     """In this task, you will be asked a question about image. However, in order to answer this question, you need knoweldge outside of this image. The question is: {question}""",
        #     """Question: {question}\n\nUse external knoweldge and the content of the image to answer the question.""",
        #     """Based on the content of the image and external knowledge, {question}""",
        #     """Based on your knowledge, {question}""",
        #     """{question}""",
        # ]
        # raise NotImplementedError
        # instructs = [
        #     """Answer \"{question}\" based on the content of image and external knowledge.""",
        #     """Based the image and background knowledge, {question}""",
        #     """Use external knoweldge and the content of the image to answer: {question}""",
        # ]

    # elif task == 'ocr':
    instructs = [
        """What is the text in the given region {region[0]} in the {image_token}?""",
        """Look at image and answer what is the text in {region[0]}?""",
        """In this task, you are require to extract the text in a region of the image. The region is {region[0]}. What is the text?""",
        """What is the text in the given region of the {image_token}. {region_token} {split_token.join(region)}""",
        """What is the text written on {split_token.join(region)} of the image.""",
    ]
    df_dicts.extend(dict(task_name='ocr', instructions=instruct) for instruct in instructs)     
    
        
    # elif task == 'visual_answer_justification':
    # region_info= ' '.join([ f"{k} is in {v[0]}."  for k, v in meta_data['object_regions'].items()])
    
    instructs = [
        """Given the image and question: \"{question}\"\nThe regions of the objects are: {region_info} Select an explanation from options to exlpain why \"{answer}\" is the answer.{options_str}""",
        """{region_info}\n\nGiven the question \"{question}\" Why \"{answer}\" is the answer?{options_str}""",
        """{region_info}\n\nWhy \"{answer}\" is the answer to the question \"{question}\"? {options_str}""",
        """Why \"{answer}\" is the answer to \"{question}\"?\nThe regions are {region_info}{options_str}""",
        """Given the image and question: \"{question}\" {region_info} Select an explanation from options to exlpain why \"{answer}\" is the answer.{options_str}""",
    ]
    df_dicts.extend(dict(task_name='visual_answer_justification', instructions=instruct) for instruct in instructs)  

    # ------------------------------- misc
    # elif task == 'visual_dialog':
    # dial_history = [ f"{dial_turn['q']}, {dial_turn['a']};" for dial_turn in meta_data['dialog_hist']]
    # if len(dial_history) > 0:
    instructs = [
        """Dialogue history: {' '.join(dial_history)}\nBased on the image and dialogue history, answer: {question}?""",
        """Context: {' '.join(dial_history)}\n\nGiven the image, answer the question.\n\nQuestion: {question}?""",
        """Given the image and the dialog history below:\n\n{' '.join(dial_history)}\n{question}?""",
        """Context: {dial_history[-1]}\n\nBased on the image, answer {question}?""",
        """{question}?"""
    ]
    df_dicts.extend(dict(task_name='visual_dialog_has_hist', instructions=instruct) for instruct in instructs)
    
    # else:
    instructs = [
        """Given the image, answer the question.\n\nQuestion: {question}?""",
        """Based on the image, answer: {question}?""",
        """Based on the image, answer {question}?""",
        """Given the image, {question}?""",
        """{question}?"""
    ]
    df_dicts.extend(dict(task_name='visual_dialog_no_hist', instructions=instruct) for instruct in instructs)

    # elif task == 'purpose_driven_affordance': # remove
        # instructs = [
        #     """Given the image what can you do to the object in {region[0]}?""",
        #     """What does {region[1]} do to the object in {region[0]}?"""
        # ]
        # raise NotImplementedError
    # elif task == 'visual_text_extraction':
    instructs = [
                    """This image contains some text. For this task, you need to look at the image carefully and identify all the text in the image. The text in the image is""",
                    """There is some text written on the image. Tell me what is the text.""",
                    """What is the text written on the image?""",
                    """The text written on the image is""",
                    """Tell me all the text on the image."""
    ]
    df_dicts.extend(dict(task_name='visual_text_extraction', instructions=instruct) for instruct in instructs)

    # elif task == 'hateful_content_detection':
    instructs = [
                    """In this task, you need to decide if there is hateful content in the given image. The image itself may not contain hateful content but when combined with the text written on the image, it may have.{options_str}""",
                    """Considering both the content of the image and the text on the image, decide if it contains hateful intension.{options_str}""",
                    """Look at the image and the text on the image. Decide if there is hateful    intention.{options_str}""",
                    """Decide if there is hateful content in the given image.{options_str}""",
                    """Is there hateful intention in the given image?{options_str}"""
    ]
    df_dicts.extend(dict(task_name='hateful_content_detection', instructions=instruct) for instruct in instructs)

    # elif task == 'medic_damage_severity':
    instructs = [
        """What is the damage level in the image?""",
        """Look at the image and decide the damage level. The damage level is given in options.{options_str}""",
        """Select the damage severity from options.{options_str}""",
        """In this task, you are required to decide how bad is the damage in the image. The levels of damage are severe, mild, and little or none.""",
        """Tell me how bad is the damage in the image.{options_str}"""
    ]
    df_dicts.extend(dict(task_name='medic_damage_severity', instructions=instruct) for instruct in instructs)

    # elif task == 'medic_informative':
    instructs = [
        """Does this image provide any information about the disaster? Choose the correct answer from options.{options_str}""",
        """Is this picture informative about a disaster?{options_str}""",
        """Is this a informative picture of disaster?{options_str}""",
        """Can you gain any information about a disaster from the image? If yes, select informative, if not select not informative from the options.{options_str}""",
        """If this picture is about a disaster, select informative. Otherwise, select not informative from options.{options_str}""",
    ]
    df_dicts.extend(dict(task_name='medic_informative', instructions=instruct) for instruct in instructs)


    # elif task == 'medic_disaster_types':
    instructs = [
        """According to the image, what kind of disaster happened? Choose the correct answer from options.{options_str}""",
        """What kind of disaster happens in the image? If no disaster happens in the image, select not disaster.{options_str}""",
        """What disaster happens in the image?{options_str}""",
        """Based on the image, what is the disaster. Select your answer from options.{options_str}""",
        """Look at the image and tell me what kind of disater is in the image. If no disaster, select not disaster.{options_str}""",
    ]
    df_dicts.extend(dict(task_name='medic_disaster_types', instructions=instruct) for instruct in instructs)

    # elif task == 'image_generation': 
    instructs = [
        """what is the complete image? caption: {text}.""", # ofa instruction
        """Generate an image with the caption \"{text}\".""",
        """Create an image of \"{text}\".""",
        """Generate the image that corresponds to the description \"{text}\".""",
        """Generate the missing part of the image, based on the text \"{text}\"."""
    ]
    df_dicts.extend(dict(task_name='image_generation', instructions=instruct) for instruct in instructs)

    # elif task == 'im_descriptive_extraction':
    instructs = [
        """Extract the part of the image with caption \"{text}\"""", 
        """Extract part of the image specified by the given caption. Caption: \"{text}\"""",
        """Given an image, generate a copy of the image with \"{text}\" as description.""",
        """Find the region that most accurately depicts \"{text}\" and then create an image of that region.""",
        """Create a new image by extracting the region that most accurately corresponds to \"{text}\" in the original image."""
    ]
    df_dicts.extend(dict(task_name='im_descriptive_extraction', instructions=instruct) for instruct in instructs)
    
    data_frame = pd.DataFrame(df_dicts)
    data_frame.to_csv(RAW_FILE, index=False, sep='\t')

def generate_new_instruction(ckpt_dir: str, tokenizer_path: str, temperature: float = 0.6, top_p: float = 0.9, max_seq_len: int = 512, max_batch_size: int = 4, max_gen_len: Optional[int] = None, instr_dir: str = None):
    df = pd.read_csv(RAW_FILE, sep='\t')
    task_names = df['task_name'].unique()

    generator = Generator(
        ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, temperature=temperature, top_p=top_p, max_seq_len=max_seq_len, max_batch_size=max_batch_size, max_gen_len=max_gen_len, instr_dir=instr_dir)

    new_df = df.copy()
    new_df['origin'] = [-1] * len(df)
    for task_name in tqdm(task_names):
        # pure format-string instructions, no way to augment
        subdf = df[df.task_name == task_name]
        if task_name in ['open-domain_VQA', 'VQA'] or task_name in TEST_TASKS:
            continue
        for i, row in subdf.iterrows():       
            raw_inst = row['instructions']
            gen_insts = generator.generate(raw_inst)
            for gen_inst in gen_insts:
                new_row = dict(task_name=task_name, instructions=gen_inst, origin=i)
                new_df.loc[len(new_df)] = new_row

        print(f'Gen List length is {len(new_df)}')

    len_map = {instr: len(instr.split()) for instr in new_df['instructions']}
    new_df['length'] = new_df['instructions'].map(len_map)
    new_df.to_csv(TRG_FILE, sep='\t', index=False)


def generate_new_instruction_multi_temp(ckpt_dir: str, tokenizer_path: str, temperature: 0.6, top_p: float = 0.9, max_seq_len: int = 256, max_batch_size: int = 4, max_gen_len: Optional[int] = None, instr_dir: str = None):
    df = pd.read_csv(RAW_FILE, sep='\t')
    task_names = df['task_name'].unique()

    print(f'temperature is {temperature}')
    generator = Generator(
        ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, temperature=temperature, top_p=top_p, max_seq_len=max_seq_len, max_batch_size=max_batch_size, max_gen_len=max_gen_len, instr_dir=instr_dir)

    new_df = df.copy()
    new_df['origin'] = [-1] * len(df)
    for task_name in tqdm(task_names):
        # pure format-string instructions, no way to augment
        subdf = df[df.task_name == task_name]
        if task_name in ['open-domain_VQA', 'VQA'] or task_name in TEST_TASKS:
            continue
        for i, row in subdf.iterrows():       
            raw_inst = row['instructions']
            gen_insts = generator.generate(raw_inst)
            for gen_inst in gen_insts:
                new_row = dict(task_name=task_name, instructions=gen_inst, origin=i)
                new_df.loc[len(new_df)] = new_row

        print(f'Gen List length is {len(new_df)}')

    len_map = {instr: len(instr.split()) for instr in new_df['instructions']}
    new_df['length'] = new_df['instructions'].map(len_map)
    new_df.to_csv(eval('f"{}"'.format(MT_TGT_FILE)), sep='\t', index=False)

# def generate_new_instruction_multi_step(ckpt_dir: str, tokenizer_path: str, temperature: float = 0.6, top_p: float = 0.9, max_seq_len: int = 512, max_batch_size: int = 4, max_gen_len: Optional[int] = None, instr_dir: str = None):
#     last_df = pd.read_csv(RAW_FILE, sep='\t')
    
#     for num_iter in range(1):
#         print(f'Start iteration {i}, temperature is {temperature}')
#         task_names = last_df['task_name'].unique()

#         generator = Generator(
#             ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, temperature=temperature, top_p=top_p, max_seq_len=max_seq_len, max_batch_size=max_batch_size, max_gen_len=max_gen_len, instr_dir=instr_dir)

#         new_df = last_df.copy()
#         if num_iter == 0:
#             df = new_df[new_df['origin'] >= 0]
#             new_df['iteration'] = [0] * len(new_df)
#         else:
#             # use last iteration's results
#             df = new_df[new_df['iteration'] == i + 1]
            
#         for task_name in tqdm(task_names):
#             # pure format-string instructions, no need to augment
#             if task_name in ['open-domain_VQA', 'VQA'] or task_name in TEST_TASKS:
#                 continue
#             subdf = df[df.task_name == task_name]
            
#             for i, row in subdf.iterrows():       
#                 raw_inst = row['instructions']
                
#                 # instructions are composed of only placeholders, no need to proceed
#                 if len(re.replace('{.+?}', '', raw_inst).strip()) == 0:
#                     continue
                
#                 gen_insts = generator.generate(raw_inst)
#                 for gen_inst in gen_insts:
#                     new_row = dict(task_name=task_name, instructions=gen_inst, origin=i)
#                     new_df.loc[len(new_df)] = new_row

#                     print(len(gen_insts))
#             print(f'Gen List length is {len(new_df)}')

#         len_map = {instr: len(instr.split()) for instr in new_df['instructions']}
#         new_df['length'] = new_df['instructions'].map(len_map)
#         new_df.to_csv(f'./instruction_data/multistep/gen_instructions_temp{temperature}_iter{i+1}.tsv', sep='\t', index=False)
#         last_df = new_df

def filter_instructions():
    df = pd.read_csv(SRC_FILE, sep='\t')

    print(f'Start filtering... Input length is {len(df)}')

    preserve_origin = True
    raw_df, gen_df = df[df['origin'] == -1], df[df['origin'] >= 0]
    is_preserved_list = [preserve_origin] * len(raw_df)

    for _, row in gen_df.iterrows():
        # origin instruction and length
        origin, length = row['origin'], row['length']

        # instruction
        inst = row['instructions']

        if origin >= len(raw_df):
            oriorigin = gen_df.loc[origin, 'origin']  # origin's origin

        # filter conditions
        # rule 1: length does not exceeds twice of original length
        if origin < len(raw_df):
            origin_inst = raw_df.loc[origin, 'instructions']
            origin_len = raw_df.loc[origin, 'length']
        else:
            origin_inst = df.loc[oriorigin, 'instructions']
            origin_len = df.loc[origin, 'length']
            
        # rule 2: filter too short/long generations to circumvent hallu
        is_preserved = (0.0 * origin_len < length < 2.0 * origin_len)

        # rule 3: formatted strings are (sorted) equal
        is_preserved &= (sorted(re.findall('{.*?}', inst)) == sorted(re.findall('{.*?}', origin_inst)))   # no order enforcement
        
        # rule 4: the sentence does not have "'d, 'll"
        # is_preserved &= (len(re.findall("('d|'ll)", inst)) == 0)
        # is_preserved_list.append(is_preserved)

    if not preserve_origin:
        pass

    df = pd.concat([raw_df, gen_df]).reset_index(drop=True)
    df['is_preserved'] = is_preserved_list
    num_preserved = sum(is_preserved_list)
    print(f'End filtering... Result length is {num_preserved}')
    df.to_csv(GEN_TRG_FILE, sep='\t', index=False)

def build_with_new_instructions(data_frame, task, text=None, options=None, region=None, context=None, question=None, explanation=None, response=None, premise=None, hypothesis=None, answer=None, meta_data=None, target=None, use_natural=False, instruction_id=-1):
    image_token = "image" # this show only appear before the output token
    options_token = "\n\n[Options]:"
    region_token = "Regions: "
    split_token = "||||" # or 
    region_split_token = "||||" # or

    if options:
        random.shuffle(options)
        # define options string
        if use_natural  == 'use_natural':
            num_choices = list(range(1, len(options)+1)) # 1, 2, 3, ..
            num_choices_b = [f'({c})' for c in num_choices] # (1), (2), (3), ..
            lower_letter_choices = [f'({c})' for c in string.ascii_lowercase] # (a), (b), (c), ..
            upper_letter_choices = [f'({c})' for c in string.ascii_uppercase] # (A), (B), (C), ..
            op_choices = [f'Option {c}:' for c in num_choices] # Option 1:, Option 2:, ...
            choices_list = [('', num_choices, ', '), ('', num_choices_b, ', '), ('', lower_letter_choices, ', '), ('', upper_letter_choices, ', '), ('\n', op_choices, '\n')]
            tgt_choice = choices_list[instruction_id]
            
            options_str = [f'{tgt_choice[1][i]}. {option}' for i, option in enumerate(options)]
            options_str = tgt_choice[2].join(options_str)
            options_str = f'{options_token}{tgt_choice[0]} {options_str}'
        else:
            options_str = f'{options_token} {split_token.join(options)}'

    if task in ['visual_object_region', 'visual_subject_region']:
        region = region_split_token.join(meta_data['object_regions']['subject'])
    elif task in ['wikihow_next_step']:
        context = '\n'.join(context) if len(context)>0 else '\"nothing\"'
    elif task in ['wikihow_text_image_step_order', 'wikihow_image_text_step_order']:
        options = ['next','previous']
        random.shuffle(options)
    elif task in ['select_overlap_most_region', 'select_overlap_least_region', 'select_overlaped_region', 'select_nonoverlaped_region', 'if_region_overlap']:
        given_region = region_split_token.join(meta_data['object_regions']['given_region'])
    elif task in ['commonsense_VQA']:
        pass  
    elif task in ['visual_answer_justification']:
        region_info= ' '.join([ f"{k} is in {v[0]}."  for k, v in meta_data['object_regions'].items()])
    elif task == ['visual_dialog']:
        dial_history = [ f"{dial_turn['q']}, {dial_turn['a']};" for dial_turn in meta_data['dialog_hist']]
        if len(dial_history) > 0:
            task = 'visual_dialog_has_hist'
        else:
            task = 'visual_dialog_no_hist'
    
    instr_pool = data_frame[(data_frame['task_name'] == task) & data_frame['is_preserved']]['instructions'].values

    while True:
        maybe_formated_instruction = random.choice(instr_pool)
        # strip special tokens
        stripped_finst = maybe_formated_instruction.strip().strip('\")')
        has_enter = False
        if re.findall('\n', stripped_finst):
            has_enter = True
            stripped_finst = re.sub('\n', '[new_line]', stripped_finst)

        # remove redundant quotes
        if re.findall('"{text}"', stripped_finst) and re.findall('"', stripped_finst):
            stripped_finst = re.sub('\"','',stripped_finst)
        if re.findall('\'{text}\'', stripped_finst) and re.findall('\'', stripped_finst):
            stripped_finst = re.sub('\'','', stripped_finst)

        if re.findall('\'', stripped_finst) and re.findall('\"', stripped_finst):
            stripped_finst = re.sub('\'', '"', stripped_finst)

        # fix undefined bug
        method = 'method'

        try:
            instruction = eval('f"{}"'.format(stripped_finst))
            break
        except:
            instruction = eval("f'{}'".format(stripped_finst))
            break

    # post processing: re-add those EOFs
    if has_enter:
        instruction = re.sub('[new_line]', '\n', stripped_finst)
    return instruction, target

def build_with_new_instructions_sample(data_frame, task, text=None, options=None, region=None, context=None, question=None, explanation=None, response=None, premise=None, hypothesis=None, answer=None, meta_data=None, target=None, use_natural=False, instruction_id=-1):
    image_token = "image" # this show only appear before the output token
    options_token = "\n\n[Options]:"
    region_token = "Regions: "
    split_token = "||||" # or 
    region_split_token = "||||" # or

    if options:
        random.shuffle(options)
        # define options string
        if use_natural  == 'use_natural':
            num_choices = list(range(1, len(options)+1)) # 1, 2, 3, ..
            num_choices_b = [f'({c})' for c in num_choices] # (1), (2), (3), ..
            lower_letter_choices = [f'({c})' for c in string.ascii_lowercase] # (a), (b), (c), ..
            upper_letter_choices = [f'({c})' for c in string.ascii_uppercase] # (A), (B), (C), ..
            op_choices = [f'Option {c}:' for c in num_choices] # Option 1:, Option 2:, ...
            choices_list = [('', num_choices, ', '), ('', num_choices_b, ', '), ('', lower_letter_choices, ', '), ('', upper_letter_choices, ', '), ('\n', op_choices, '\n')]
            tgt_choice = choices_list[instruction_id]
            
            options_str = [f'{tgt_choice[1][i]}. {option}' for i, option in enumerate(options)]
            options_str = tgt_choice[2].join(options_str)
            options_str = f'{options_token}{tgt_choice[0]} {options_str}'
        else:
            options_str = f'{options_token} {split_token.join(options)}'

    if task in ['visual_object_region', 'visual_subject_region']:
        region = region_split_token.join(meta_data['object_regions']['subject'])
    elif task in ['wikihow_next_step']:
        context = '\n'.join(context) if len(context)>0 else '\"nothing\"'
    elif task in ['wikihow_text_image_step_order', 'wikihow_image_text_step_order']:
        options = ['next','previous']
        random.shuffle(options)
    elif task in ['select_overlap_most_region', 'select_overlap_least_region', 'select_overlaped_region', 'select_nonoverlaped_region', 'if_region_overlap']:
        given_region = region_split_token.join(meta_data['object_regions']['given_region'])
    elif task in ['commonsense_VQA']:
        pass  
    elif task in ['visual_answer_justification']:
        region_info= ' '.join([ f"{k} is in {v[0]}."  for k, v in meta_data['object_regions'].items()])
    elif task == ['visual_dialog']:
        dial_history = [ f"{dial_turn['q']}, {dial_turn['a']};" for dial_turn in meta_data['dialog_hist']]
        if len(dial_history) > 0:
            task = 'visual_dialog_has_hist'
        else:
            task = 'visual_dialog_no_hist'
    
    instr_pool = data_frame[(data_frame['task_name'] == task) & data_frame['is_preserved']]['instructions'].values
    raw_instr_pool = data_frame[(data_frame['task_name'] == task) & data_frame['origin'] == -1]


    import numpy as np
    if len(instr_pool) == len(raw_instr_pool):
        instr_prob = ([RAW_LOGIT/len(raw_instr_pool)] * len(raw_instr_pool))
    elif len(raw_instr_pool) == 0:
        instr_prob = [GEN_LOGIT/((len(instr_pool) - len(raw_instr_pool)))] * (len(instr_pool) - len(raw_instr_pool))
    else:
        instr_prob = ([RAW_LOGIT/len(raw_instr_pool)] * len(raw_instr_pool)) + ([GEN_LOGIT/((len(instr_pool) - len(raw_instr_pool)))] * (len(instr_pool) - len(raw_instr_pool)))
    instr_prob = instr_prob / np.sum(instr_prob)

    # maybe_formated_instruction = random.choice(instr_pool)
    maybe_formatted_instruction = np.random.choice(instr_pool, size=1, p=instr_prob)
    maybe_formatted_instruction = maybe_formatted_instruction.tolist()[0]

    # strip special tokens
    stripped_finst = maybe_formatted_instruction.strip().strip('\")')
    has_enter = False
    if re.findall('\n', stripped_finst):
        has_enter = True
        stripped_finst = re.sub('\n', '[new_line]', stripped_finst)

    # remove redundant quotes
    if re.findall('"{text}"', stripped_finst) and re.findall('"', stripped_finst):
        stripped_finst = re.sub('\"','',stripped_finst)
    if re.findall('\'{text}\'', stripped_finst) and re.findall('\'', stripped_finst):
        stripped_finst = re.sub('\'','', stripped_finst)

    if re.findall('\'', stripped_finst) and re.findall('\"', stripped_finst):
        stripped_finst = re.sub('\'', '"', stripped_finst)

    if task == 'wikihow_text_image_step_order':
        method = 'method'
        
    # eval formatted strings
    try:
        instruction = eval('f"{}"'.format(stripped_finst))
    except:
        instruction = eval("f'{}'".format(stripped_finst))
    
    # post processing: re-add those EOFs
    if has_enter:
        instruction = re.sub('[new_line]', '\n', stripped_finst)
        
    return instruction, target
    
if __name__ == '__main__':
    save_raw_instruction()

    # step 1: Generate new instruction
    fire.Fire(generate_new_instruction)
    # For multi-temp generation
    # fire.Fire(generate_new_instruction_multi_step)

    # step 2: Use heuristics to filter generated instruction
    filter_instructions()
