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



class CommonsenseVQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, csv_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, csv_path=csv_path)
        self.img_ids = {}

class CommonsenseVQAEvalDataset(BaseDataset):
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
            "options": ann["options"]
        }