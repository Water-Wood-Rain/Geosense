"""
Base-model profiling wrapper.

Inherits Qwen2_5_VL and adds per-sample latency logging.
Forces geometry encoder OFF so the base path is clean.

Model args example:
    --model_args "pretrained=...,latency_log=/path/base_pure2d.jsonl,workload=pure_2d"
"""

from typing import List, Optional, Union

import torch
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.registry import register_model
from lmms_eval.models.qwen2_5_vl import Qwen2_5_VL
from lmms_eval.models.profile_utils import (
    LatencyLogger,
    cuda_sync_time,
    _count_input_tokens,
)
from PIL import Image


@register_model("qwen2_5_vl_latency")
class Qwen2_5_VL_Latency(Qwen2_5_VL):

    def __init__(
        self,
        latency_log: str = "latency_base.jsonl",
        workload: str = "unset",
        **kwargs,
    ) -> None:
        # Force geometry off before parent loads the model
        # We patch config after AutoConfig.from_pretrained inside parent,
        # so we override via kwargs that parent will pass to AutoConfig logic.
        # The safest approach: let parent load, then assert no geometry was used.
        super().__init__(**kwargs)

        # Verify (or enforce) geometry is disabled
        cfg = self._config
        if getattr(cfg, "use_geometry_encoder", False) or getattr(cfg, "use_vggt_feature", False):
            raise RuntimeError(
                "[qwen2_5_vl_latency] Base model must have geometry encoder disabled. "
                "Set use_geometry_encoder=False in the model config."
            )

        self._latency_log = latency_log
        self._workload = workload
        self._logger: Optional[LatencyLogger] = None  # lazy-init on first generate_until

    def _get_logger(self) -> LatencyLogger:
        if self._logger is None:
            self._logger = LatencyLogger(
                path=self._latency_log,
                model="base",
                workload=self._workload,
                rank=self._rank,
            )
        return self._logger

    # ------------------------------------------------------------------
    # Override generate_until: identical to parent except timing + logging
    # ------------------------------------------------------------------
    def generate_until(self, requests):
        res = []
        logger = self._get_logger()

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            entries = []
            for i, ctx in enumerate(contexts):
                visual = visuals[i] if i < len(visuals) else None
                entry = {
                    "id": str(doc_id[i]),
                    "conversations": [],
                    "data_source": "lmms_eval",
                    "data_path": "",
                    "tag": "2d",
                }
                if visual is None:
                    human_val = ctx
                elif isinstance(visual, str) and visual.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")):
                    entry["video"] = visual
                    human_val = "<video>\n" + ctx
                elif isinstance(visual, Image.Image):
                    entry["image"] = [visual]
                    human_val = "<image>\n" + ctx
                elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
                    entry["image"] = list(visual)
                    human_val = ("<image>" * len(visual)) + "\n" + ctx
                else:
                    human_val = ctx
                entry["conversations"].append({"from": "human", "value": human_val})
                entries.append(entry)

            batch = self._batch_from_entries(entries)
            gen_kwargs = dict(all_gen_kwargs[0]) if all_gen_kwargs else {}

            # --- Latency measurement ---
            in_lens = _count_input_tokens(batch)
            t0 = cuda_sync_time()
            answers = self._generate_batch(batch, gen_kwargs)
            t1 = cuda_sync_time()
            total_ms = t1 - t0
            # Distribute latency evenly across samples in chunk (batch_size=1 in normal runs)
            n = len(answers)
            per_sample_ms = total_ms / max(n, 1)

            for i, (ans, ctx) in enumerate(zip(answers, contexts)):
                entry = entries[i]
                num_images = len(entry.get("image", []))
                num_frames = 1 if "video" in entry else 0
                logger.log(
                    sample_id=entry["id"],
                    task=task,
                    r1_latency_ms=per_sample_ms,
                    r2_latency_ms=0.0,
                    input_tokens=in_lens[i] if i < len(in_lens) else 0,
                    r1_output_tokens=0,  # token count not tracked here for simplicity
                    triggered=False,
                    used_vggt=False,
                    gate_label="base_disabled",
                    num_images=num_images,
                    num_frames=num_frames,
                )
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (ctx, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
