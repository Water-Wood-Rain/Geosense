"""
Shared profiling utilities for latency experiments.

Usage pattern in a wrapper:
    from lmms_eval.models.profile_utils import cuda_sync_time, LatencyLogger

    logger = LatencyLogger(path="/path/to/output.jsonl", model="base", workload="pure_2d")
    t0 = cuda_sync_time()
    answers = self._generate_batch(batch, gen_kwargs)
    t1 = cuda_sync_time()
    logger.log(sample_id="0", r1_latency_ms=t1-t0, ...)
"""

import json
import os
import time
from typing import Optional

import torch


def cuda_sync_time() -> float:
    """
    Return current time in milliseconds after synchronizing the current CUDA device.
    Falls back to time.perf_counter if CUDA is not available.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() * 1000.0


def _count_output_tokens(
    outputs: "list[torch.Tensor]",
    in_lens: "list[int]",
) -> list[int]:
    """Count the number of generated tokens per sample (output_ids minus input_ids)."""
    result = []
    for out, in_len in zip(outputs, in_lens):
        result.append(max(0, int(out.shape[0]) - in_len))
    return result


def _count_input_tokens(batch: dict) -> list[int]:
    """Count the number of real (non-pad) input tokens per sample."""
    in_lens = []
    am = batch.get("attention_mask", None)
    if am is not None and hasattr(am, "dim") and am.dim() == 2:
        in_lens = am.sum(dim=1).tolist()
    if not in_lens:
        input_ids = batch.get("input_ids", None)
        if input_ids is not None:
            # try to infer from input_ids directly
            for row in input_ids:
                in_lens.append(int(row.numel()))
    return in_lens


class LatencyLogger:
    """
    Thread-safe(ish) JSONL writer for per-sample latency records.

    Each call to log() appends one line to the output file.
    Fields that are None are omitted from the record.
    """

    def __init__(
        self,
        path: str,
        model: str,
        workload: str,
        rank: int = 0,
    ) -> None:
        self.path = path
        self.model = model
        self.workload = workload
        self.rank = rank
        # Only rank-0 writes to avoid duplicate lines in multi-GPU runs.
        # Each rank writes to its own shard file; rank-0 is the canonical output.
        if rank == 0:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._f = open(path, "a", encoding="utf-8")
        else:
            self._f = None

    def log(
        self,
        *,
        sample_id: str,
        task: str = "",
        r1_latency_ms: float,
        r2_latency_ms: float = 0.0,
        input_tokens: int = 0,
        r1_output_tokens: int = 0,
        r2_output_tokens: int = 0,
        triggered: bool = False,
        used_vggt: bool = False,
        gate_label: str = "",
        num_images: int = 0,
        num_frames: int = 0,
    ) -> None:
        if self._f is None:
            return
        record = {
            "model": self.model,
            "workload": self.workload,
            "task": task,
            "sample_id": sample_id,
            "r1_latency_ms": round(r1_latency_ms, 3),
            "r2_latency_ms": round(r2_latency_ms, 3),
            "total_latency_ms": round(r1_latency_ms + r2_latency_ms, 3),
            "input_tokens": input_tokens,
            "r1_output_tokens": r1_output_tokens,
            "r2_output_tokens": r2_output_tokens,
            "triggered": triggered,
            "used_vggt": used_vggt,
            "gate_label": gate_label,
            "num_images": num_images,
            "num_frames": num_frames,
        }
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __del__(self) -> None:
        self.close()
