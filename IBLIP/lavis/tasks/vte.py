"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os
import pandas as pd

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from torchmetrics.text.rouge import ROUGEScore

@registry.register_task("vte")
class VTETask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.targets = dict()

        self.rscore = ROUGEScore(rouge_keys='rougeL')
        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def build_datasets(self, cfg):
        # datasets = super().build_datasets(cfg)['vte']
        datasets = super().build_datasets(cfg)
        for _, dataset in datasets.items():
            for split in dataset.values():
                for row in split:
                    uid, target = row['instance_id'], row['text_output']
                    self.targets[uid] = target
        
        return datasets

    def valid_step(self, model, samples):
        gen_res = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )
        
        result = []
        unique_id = samples["instance_id"]
        for uid, gen_txt in zip(unique_id, gen_res):
            result.append({"uid": uid, 'gen_txt': gen_txt})
        return result

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
        )
        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=result_file
            )
        else:
            metrics = {"agg_metrics": 0.0}
        return metrics
    
    @main_process
    def _report_metrics(self, eval_result_file):
        # compute rouge
        eval_result = json.load(open(eval_result_file, 'r'))
        
        rouge = 0
        for res in eval_result:
            uid, gen_txt = res["uid"], res["gen_txt"]
            rouge += self.rscore(gen_txt, self.targets[uid])['rougeL_fmeasure'].item()
        rouge /= len(eval_result)

        metrics = dict()
        metrics["ROUGE-L"] = rouge
        logging.info("Overall ROUGE is: %.4f\n" % rouge)
        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")
        return res