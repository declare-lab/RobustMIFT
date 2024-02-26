"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.tvqa_datasets import TVQADataset, TVQAEvalDataset
from lavis.datasets.datasets.visdial_datasets import VisDialDataset, VisDialEvalDataset
from lavis.datasets.datasets.commonsenseVQA_datasets import CommonsenseVQADataset, CommonsenseVQAEvalDataset
from lavis.datasets.datasets.instr_datasets import InstrDataset, InstrEvalDataset, InstrEvalCLSDataset

# training task
@registry.register_builder("instr_tuning")
class InstrTuningBuilder(BaseDatasetBuilder):
    train_dataset_cls = InstrDataset
    eval_dataset_cls = InstrEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instr/defaults_tuning.yaml",
        "eval":  "configs/datasets/instr/defaults_tuning.yaml"
    }

# generation task
@registry.register_builder("vte")
class VTEBuilder(BaseDatasetBuilder):
    train_dataset_cls = InstrDataset
    eval_dataset_cls = InstrEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instr/defaults_vte.yaml",
        "eval":  "configs/datasets/instr/defaults_vte.yaml"
    }
    
@registry.register_builder("tvqa")
class TVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = InstrDataset
    eval_dataset_cls = InstrEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instr/defaults_tvqa.yaml",
        "eval":  "configs/datasets/instr/defaults_tvqa.yaml"
    }

@registry.register_builder("visdial")
class VisDialBuilder(BaseDatasetBuilder):
    train_dataset_cls = VisDialDataset
    eval_dataset_cls = VisDialEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instr/defaults_visdial.yaml",
        "eval":  "configs/datasets/instr/defaults_visdial.yaml"
    }

@registry.register_builder("commonsensevqa")
class CommonsenseVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = CommonsenseVQADataset
    eval_dataset_cls = CommonsenseVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instr/defaults_commonsensevqa.yaml",
        "eval":  "configs/datasets/instr/defaults_commonsensevqa.yaml"
    }

# MC task
@registry.register_builder("dc")
class DCBuilder(BaseDatasetBuilder):
    train_dataset_cls = InstrDataset
    eval_dataset_cls = InstrEvalCLSDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instr/defaults_dc.yaml",
        "eval":  "configs/datasets/instr/defaults_dc.yaml"
    }

@registry.register_builder("gvqa")
class GVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = InstrDataset
    eval_dataset_cls = InstrEvalCLSDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instr/defaults_gvqa.yaml",
        "eval":  "configs/datasets/instr/defaults_gvqa.yaml"
    }

@registry.register_builder("visnli")
class VisNLIBuilder(BaseDatasetBuilder):
    train_dataset_cls = InstrDataset
    eval_dataset_cls = InstrEvalCLSDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instr/defaults_visnli.yaml",
        "eval":  "configs/datasets/instr/defaults_visnli.yaml"
    }

# classification task
@registry.register_builder("vsr")
class VSRBuilder(BaseDatasetBuilder):
    train_dataset_cls = InstrDataset
    eval_dataset_cls = InstrEvalCLSDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instr/defaults_vsr.yaml",
        "eval":  "configs/datasets/instr/defaults_vsr.yaml"
    }


@registry.register_builder("nlvrins")
class NLVRBuilder(BaseDatasetBuilder):
    train_dataset_cls = InstrDataset
    eval_dataset_cls = InstrEvalCLSDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instr/defaults_nlvr.yaml",
        "eval":  "configs/datasets/instr/defaults_nlvr.yaml"
    }

