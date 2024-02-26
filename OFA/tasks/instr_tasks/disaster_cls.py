# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import os
import json
import logging
from typing import Optional
from argparse import Namespace
from itertools import zip_longest
from collections import OrderedDict

import numpy as np
# evaluation scores
import sacrebleu
from torchmetrics.text.rouge import ROUGEScore

import string

from fairseq import metrics, utils
from fairseq.tasks import FairseqTask, register_task

from tasks.ofa_task import OFAConfig, OFATask
from data.instr_data.disasterclassification_dataset import DisasterClassifcationMCInstrDataset
from data.file_dataset import FileDataset
from utils.cider.pyciderevalcap.ciderD.ciderD import CiderD
from omegaconf import DictConfig

EVAL_BLEU_ORDER = 4
logger = logging.getLogger(__name__)


@dataclass
class DisasterClassificationConfig(OFAConfig):
    max_src_length: int = field(
        default=1024, metadata={"help": "the maximum src sequence length"}
    )
    max_tgt_length: int = field(
        default=30, metadata={"help": "the maximum target sequence length"}
    )
    # evaluation configs
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_cider: bool = field(
        default=False, metadata={"help": "evaluation with CIDEr scores"}
    )
    eval_rouge: bool = field(
        default=False, metadata={"help": "evaluation with ROUGE-L scores"}
    )
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args for BLUE or CIDEr scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    eval_cider_cached_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "path to cached cPickle file used to calculate CIDEr scores"},
    )

    scst: bool = field(
        default=False, metadata={"help": "Self-critical sequence training"}
    )
    scst_args: str = field(
        default='{}',
        metadata={
            "help": 'generation args for Self-critical sequence training, as JSON string'
        },
    )

@register_task("disaster_classification", dataclass=DisasterClassificationConfig)
class DisasterClassificationTask(OFATask):
    def __init__(self, cfg: DisasterClassificationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
    
    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task."""
        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(cfg.bpe_dir, "dict.txt")
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(cfg.bpe_dir, "dict.txt")
        )
        src_dict.add_symbol("<mask>")
        tgt_dict.add_symbol("<mask>")
        for i in range(cfg.code_dict_size):
            src_dict.add_symbol("<code_{}>".format(i))
            tgt_dict.add_symbol("<code_{}>".format(i))
        # quantization
        for i in range(cfg.num_bins):
            src_dict.add_symbol("<bin_{}>".format(i))
            tgt_dict.add_symbol("<bin_{}>".format(i))

        logger.info("source dictionary: {} types".format(len(src_dict)))
        logger.info("target dictionary: {} types".format(len(tgt_dict)))
        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0
        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = DisasterClassifcationMCInstrDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_bleu or self.cfg.eval_cider:
            gen_args = json.loads(self.cfg.eval_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
            if self.cfg.eval_cider:
                self.CiderD_scorer = CiderD(df=self.cfg.eval_cider_cached_tokens)
        if self.cfg.scst:
            scst_args = json.loads(self.cfg.scst_args)
            self.scst_generator = self.build_generator(
                [model], Namespace(**scst_args)
            )
        return model

    def _calculate_cider_scores(self, gen_res, gt_res):
        pass

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        if self.cfg.eval_bleu:
            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

        if self.cfg.eval_cider:
            def compute_cider(meters):
                cider = meters["_cider_score_sum"].sum / meters["_cider_cnt"].sum
                cider = cider if isinstance(cider, float) else cider.item()
                return round(cider, 3)

            if sum_logs("_cider_cnt") > 0:
                metrics.log_scalar("_cider_score_sum", sum_logs("_cider_score_sum"))
                metrics.log_scalar("_cider_cnt", sum_logs("_cider_cnt"))
                metrics.log_derived("cider", compute_cider)
        
    def _inference(self, generator, sample, model):
        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.bpe:
                s = self.bpe.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = [], []
        transtab = str.maketrans({key: None for key in string.punctuation})
        for i in range(len(gen_out)):
            decode_tokens = decode(gen_out[i][0]["tokens"])
            hyps.append(decode_tokens.translate(transtab).strip())
            refs.append(
                [
                    sent.translate(transtab).strip()
                    for sent in decode(
                        utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                        escape_unk=True,  # don't count <unk> as matches to the hypo
                    ).split('&&')
                ]
            )
        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + ' && '.join(refs[0]))

        return hyps, refs