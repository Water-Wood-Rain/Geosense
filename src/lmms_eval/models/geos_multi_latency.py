"""
GeoSense (geos_multi) profiling wrapper.

Inherits geos_multi and adds per-sample R1/R2 split latency logging.
Key design: _generate_batch is overridden to time R1 and R2 separately.

Model args example:
    --model_args "pretrained=...,latency_log=/path/geosense_pure2d.jsonl,workload=pure_2d"
"""

import copy
from typing import Optional

from tqdm import tqdm
from PIL import Image

from lmms_eval import utils
from lmms_eval.api.registry import register_model
from lmms_eval.models.geos_multi import geos_multi, VGGT_TAG
from lmms_eval.models.profile_utils import (
    LatencyLogger,
    cuda_sync_time,
    _count_input_tokens,
)


@register_model("geos_multi_latency")
class geos_multi_latency(geos_multi):

    def __init__(
        self,
        latency_log: str = "latency_geosense.jsonl",
        workload: str = "unset",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._latency_log = latency_log
        self._workload = workload
        self._logger: Optional[LatencyLogger] = None

        # Scratch space: generate_until writes chunk metadata here so
        # _generate_batch can read it without changing the method signature.
        self._current_chunk_meta: list[dict] = []
        self._pending_latency_records: list[dict] = []

    def _get_logger(self) -> LatencyLogger:
        if self._logger is None:
            self._logger = LatencyLogger(
                path=self._latency_log,
                model="geosense",
                workload=self._workload,
                rank=self._rank,
            )
        return self._logger

    # ------------------------------------------------------------------
    # Override _generate_batch: R1 and R2 timed separately.
    # Logic is identical to geos_multi._generate_batch; only timing added.
    # ------------------------------------------------------------------
    def _generate_batch(self, batch: dict, gen_kwargs: dict | None = None) -> list[str]:
        gen_kwargs = {} if gen_kwargs is None else dict(gen_kwargs)
        gen_kwargs.setdefault("max_new_tokens", 4096)
        gen_kwargs.setdefault("temperature", 0)
        gen_kwargs.setdefault("top_p", None)
        gen_kwargs.setdefault("num_beams", 1)

        model_inputs = {k: v for k, v in batch.items() if k in self._allowed_input_keys and v is not None}
        device = self._target_device()
        model_inputs = self._to_device_tree(model_inputs, device)

        # --- Round 1 ---
        t_r1_start = cuda_sync_time()
        outputs = self.model.generate(
            **model_inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=bool(gen_kwargs["temperature"] > 0),
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            use_cache=self.use_cache,
        )
        t_r1_end = cuda_sync_time()
        r1_latency_ms = t_r1_end - t_r1_start

        in_lens = []
        am = batch.get("attention_mask", None)
        if am is not None and getattr(am, "dim", lambda: 0)() == 2:
            in_lens = am.sum(dim=1).tolist()
        if not in_lens:
            pad_id = self.tokenizer.pad_token_id
            for row in batch["input_ids"]:
                in_lens.append(int((row != pad_id).sum().item()) if pad_id is not None else int(row.numel()))

        trimmed = [out_ids[L:] for L, out_ids in zip(in_lens, outputs)]
        ans1 = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        final = list(ans1)

        need = [i for i, a in enumerate(ans1) if isinstance(a, str) and (VGGT_TAG in a)]

        # --- Build per-sample R1 records ---
        in_lens_list = _count_input_tokens(batch)
        metas = batch.get("meta_vg", None)
        if metas is None:
            metas = batch.get("meta", [{} for _ in range(len(final))])

        records = []
        for idx in range(len(final)):
            m = metas[idx] if idx < len(metas) else {}
            entry_meta = self._current_chunk_meta[idx] if idx < len(self._current_chunk_meta) else {}
            records.append({
                "sample_id": m.get("id", entry_meta.get("id", str(idx))),
                "r1_latency_ms": r1_latency_ms / max(len(final), 1),  # per-sample share
                "r2_latency_ms": 0.0,
                "input_tokens": in_lens_list[idx] if idx < len(in_lens_list) else 0,
                "triggered": idx in need,
                "num_images": len(entry_meta.get("image", [])),
                "num_frames": 1 if "video" in entry_meta else 0,
            })

        if not need:
            self._pending_latency_records = records
            return final

        # --- Round 2 ---
        second_entries, back_map = [], []
        for i in need:
            m = metas[i] if i < len(metas) else {}
            orig_conv = m.get("orig_conversations") or m.get("conversations") or []
            user0 = orig_conv[0].get("value", "") if (isinstance(orig_conv, list) and orig_conv) else ""
            hist = self._build_history_prefix(user0, ans1[i])
            q2 = self._inject_vggt_after_first_tag_block(user0)
            e2 = {
                "id": m.get("id"),
                "conversations": [{"from": "human", "value": hist + (q2 or "")}],
                "data_source": m.get("data_source", "lmms_eval"),
                "data_path": m.get("data_path", ""),
                "tag": m.get("tag", "2d"),
                "dataset_name": m.get("dataset_name", "lmms_eval"),
            }
            if "video" in m:
                e2["video"] = m["video"]
            if "image" in m:
                e2["image"] = m["image"]
            second_entries.append(e2)
            back_map.append(i)

        batch2 = self._batch_from_entries(second_entries)
        model_inputs2 = {k: v for k, v in batch2.items() if k in self._allowed_input_keys and v is not None}
        model_inputs2 = self._to_device_tree(model_inputs2, device)

        t_r2_start = cuda_sync_time()
        outputs2 = self.model.generate(
            **model_inputs2,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=bool(gen_kwargs["temperature"] > 0),
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            use_cache=self.use_cache,
        )
        t_r2_end = cuda_sync_time()
        r2_latency_ms_total = t_r2_end - t_r2_start

        in_lens2 = []
        am2 = batch2.get("attention_mask", None)
        if am2 is not None and getattr(am2, "dim", lambda: 0)() == 2:
            in_lens2 = am2.sum(dim=1).tolist()
        if not in_lens2:
            pad_id = self.tokenizer.pad_token_id
            for row in batch2["input_ids"]:
                in_lens2.append(int((row != pad_id).sum().item()) if pad_id is not None else int(row.numel()))

        trimmed2 = [out_ids[L:] for L, out_ids in zip(in_lens2, outputs2)]
        ans2 = self.processor.batch_decode(trimmed2, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        r2_per_triggered = r2_latency_ms_total / max(len(need), 1)
        for j, orig_i in enumerate(back_map):
            # Update record for this triggered sample
            records[orig_i]["r2_latency_ms"] = r2_per_triggered
            if j < len(ans2):
                if VGGT_TAG not in ans2[j]:
                    final[orig_i] = f"{VGGT_TAG} {ans2[j]}"
                else:
                    final[orig_i] = ans2[j]

        self._pending_latency_records = records
        return final

    # ------------------------------------------------------------------
    # Override generate_until: identical to parent except it saves chunk
    # metadata for _generate_batch and flushes records to logger after.
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
            task_name = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task_name][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            entries = []
            for i, ctx in enumerate(contexts):
                visual = visuals[i] if i < len(visuals) else None
                doc = self.task_dict[task_name][split][doc_id[i]]
                dataset_name = task_name
                if isinstance(doc, dict):
                    dataset_name = doc.get("dataset_name", task_name) or task_name
                entry = {
                    "id": str(doc_id[i]),
                    "conversations": [],
                    "data_source": "lmms_eval",
                    "data_path": "",
                    "tag": "2d",
                    "dataset_name": dataset_name,
                }
                if isinstance(doc, dict) and "tag" in doc:
                    entry["tag"] = doc.get("tag", "2d")

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

            # Pass chunk metadata to _generate_batch via instance variable
            self._current_chunk_meta = entries
            self._pending_latency_records = []

            batch = self._batch_from_entries(entries)
            gen_kwargs = dict(all_gen_kwargs[0]) if all_gen_kwargs else {}
            answers = self._generate_batch(batch, gen_kwargs)

            # Flush latency records
            for rec in self._pending_latency_records:
                triggered = rec["triggered"]
                logger.log(
                    sample_id=rec["sample_id"],
                    task=task_name,
                    r1_latency_ms=rec["r1_latency_ms"],
                    r2_latency_ms=rec["r2_latency_ms"],
                    input_tokens=rec["input_tokens"],
                    triggered=triggered,
                    used_vggt=triggered,
                    gate_label="triggered" if triggered else "no_trigger",
                    num_images=rec["num_images"],
                    num_frames=rec["num_frames"],
                )

            for ans, ctx in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (ctx, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
