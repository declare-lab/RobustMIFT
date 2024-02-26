"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import torch
import numpy as np

from collections import OrderedDict
from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image


class InstrDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, csv_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, csv_path=csv_path)
        self.img_ids = {}
        self.vis_root = vis_root
    
    def __getitem__(self, index):
        ann = self.annotations[index]

        image_path = os.path.join(self.vis_root, ann["image_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        return {
            "image": torch.tensor(np.array(image)),
            "text_input": ann["prompt"],
            "text_output": ann["target"]
        }
    
    def collater(self, samples):
        return {
            "image": torch.stack([s['image'] for s in samples], axis=0),
            "text_input": [s['text_input'] for s in samples],
            "text_output": [s['text_output'] for s in samples],
        }

class InstrEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, csv_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        """
        super().__init__(vis_processor, text_processor, vis_root=vis_root, csv_path=csv_path)

    def __getitem__(self, index):
        ann = self.annotations[index]

        image_path = os.path.join(self.vis_root, ann["image_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        return {
            "image": np.array(image),
            "text_input": ann["prompt"],
            "text_output": ann["target_txt"],
            "instance_id": ann["unique_id"],
        }

class InstrEvalCLSDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, csv_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        """
        super().__init__(vis_processor, text_processor, vis_root=vis_root, csv_path=csv_path)

    def __getitem__(self, index):
        ann = self.annotations[index]

        image_path = os.path.join(self.vis_root, ann["image_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        return {
            "image": np.array(image),
            "prompt": ann["prompt"],
            "text_output": ann["target_txt"],
            "instance_id": ann["unique_id"],
            "options": ann["options"],
        }