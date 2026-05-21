import os
import re
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.registry import register_model
from lmms_eval.models.geos_multi import VGGT_TAG, geos_multi


class TextMetaGateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@register_model("geos_multi_gate")
class geos_multi_gate(geos_multi):
    FEATURE_VERSION = "text_meta_v1"
    FEATURE_NAMES = [
        "has_image",
        "image_count",
        "has_video",
        "tag_block_count_clipped",
        "newline_count_clipped",
        "word_count_norm",
        "char_count_norm",
        "spatial_hits_norm",
        "count_hits_norm",
        "choice_hits_norm",
        "has_question_mark",
        "dataset_is_mme_or_vsi",
        "tag_is_2d",
        "tag_is_3d",
    ]

    SPATIAL_PATTERNS = [
        r"\bleft\b",
        r"\bright\b",
        r"\btop\b",
        r"\bbottom\b",
        r"\bmiddle\b",
        r"\bcenter\b",
        r"\bcentre\b",
        r"\bfront\b",
        r"\bbehind\b",
        r"\bbetween\b",
        r"\bunder\b",
        r"\bover\b",
        r"\babove\b",
        r"\bbelow\b",
        r"\bnear\b",
        r"\bnearest\b",
        r"\bfarthest\b",
        r"\bfurthest\b",
        r"\bdistance\b",
        r"\bposition\b",
        r"\blocation\b",
        r"\bdirection\b",
        r"\brelative\b",
        r"\balign(?:ed|ment)?\b",
        r"\boverlap\b",
        r"\bintersect\b",
        r"\binside\b",
        r"\boutside\b",
        r"\bcontain(?:s|ing)?\b",
        r"\bsurround(?:ing)?\b",
        r"\bclosest\b",
        r"\bcorner\b",
        r"左",
        r"右",
        r"上",
        r"下",
        r"中间",
        r"中心",
        r"前",
        r"后",
        r"之间",
        r"附近",
        r"距离",
        r"位置",
        r"方位",
        r"方向",
        r"重叠",
        r"相交",
        r"包含",
        r"内部",
        r"外部",
        r"周围",
        r"最近",
        r"最远",
        r"角落",
    ]

    COUNT_PATTERNS = [
        r"\bhow many\b",
        r"\bcount\b",
        r"多少",
        r"几个",
        r"计数",
    ]

    CHOICE_PATTERNS = [
        r"\b[A-D][\).]",
        r"\boption\b",
        r"\bchoices?\b",
        r"选项",
    ]

    def __init__(
        self,
        gate_ckpt: Optional[str] = None,
        gate_threshold: float = 0.5,
        gate_hidden_dim: int = 64,
        gate_mode: str = "text_meta",
        gate_enable_for_images_only: bool = True,
        gate_use_heuristic_fallback: bool = True,
        **kwargs,
    ) -> None:
        self.gate_ckpt = gate_ckpt
        self.gate_threshold = float(gate_threshold)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.gate_mode = gate_mode
        self.gate_enable_for_images_only = str(gate_enable_for_images_only).lower() not in {"false", "0", "no"}
        self.gate_use_heuristic_fallback = str(gate_use_heuristic_fallback).lower() not in {"false", "0", "no"}

        super().__init__(**kwargs)

        self._gate_input_dim = 14
        self.gate_model = TextMetaGateMLP(self._gate_input_dim, self.gate_hidden_dim)
        self.gate_model.eval()
        self._load_gate_checkpoint_if_needed()

    def _load_gate_checkpoint_if_needed(self) -> None:
        if not self.gate_ckpt:
            return
        if not os.path.exists(self.gate_ckpt):
            raise FileNotFoundError(f"gate checkpoint not found: {self.gate_ckpt}")

        checkpoint = torch.load(self.gate_ckpt, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        checkpoint_input_dim = checkpoint.get("input_dim")
        if checkpoint_input_dim is not None and int(checkpoint_input_dim) != self._gate_input_dim:
            raise ValueError(
                f"gate checkpoint input_dim mismatch: expected {self._gate_input_dim}, got {checkpoint_input_dim}"
            )

        checkpoint_hidden_dim = checkpoint.get("hidden_dim")
        if checkpoint_hidden_dim is not None and int(checkpoint_hidden_dim) != self.gate_hidden_dim:
            raise ValueError(
                f"gate checkpoint hidden_dim mismatch: expected {self.gate_hidden_dim}, got {checkpoint_hidden_dim}"
            )

        checkpoint_feature_version = checkpoint.get("feature_version")
        if checkpoint_feature_version is not None and checkpoint_feature_version != self.FEATURE_VERSION:
            raise ValueError(
                f"gate checkpoint feature_version mismatch: expected {self.FEATURE_VERSION}, got {checkpoint_feature_version}"
            )

        checkpoint_feature_names = checkpoint.get("feature_names")
        if checkpoint_feature_names is not None and list(checkpoint_feature_names) != self.FEATURE_NAMES:
            raise ValueError("gate checkpoint feature_names mismatch")

        self.gate_model.load_state_dict(state_dict, strict=True)
        print(
            f"[geos_multi_gate] loaded gate checkpoint={self.gate_ckpt} "
            f"input_dim={self._gate_input_dim} hidden_dim={self.gate_hidden_dim} feature_version={self.FEATURE_VERSION}"
        )

    def _extract_user_text(self, entry: dict) -> str:
        conversations = entry.get("conversations") or []
        if not conversations:
            return ""
        value = conversations[0].get("value", "") if isinstance(conversations[0], dict) else ""
        return value if isinstance(value, str) else ""

    def _count_matches(self, text: str, patterns: List[str]) -> int:
        total = 0
        lowered = text.lower()
        for pattern in patterns:
            total += len(re.findall(pattern, lowered, flags=re.IGNORECASE))
        return total

    def _entry_metadata(self, entry: dict) -> dict:
        text = self._extract_user_text(entry)
        has_image = "image" in entry and bool(entry.get("image"))
        image_count = len(entry.get("image", [])) if has_image else 0
        has_video = "video" in entry and bool(entry.get("video"))
        tag_block_count = len(re.findall(r"<[^<>]+>", text))
        newline_count = text.count("\n")
        word_count = len(text.split())
        char_count = len(text)
        spatial_hits = self._count_matches(text, self.SPATIAL_PATTERNS)
        count_hits = self._count_matches(text, self.COUNT_PATTERNS)
        choice_hits = self._count_matches(text, self.CHOICE_PATTERNS)
        has_question_mark = int("?" in text or "？" in text)
        dataset_name = str(entry.get("dataset_name", "") or "")
        tag = str(entry.get("tag", "") or "")

        return {
            "text": text,
            "has_image": has_image,
            "image_count": image_count,
            "has_video": has_video,
            "tag_block_count": tag_block_count,
            "newline_count": newline_count,
            "word_count": word_count,
            "char_count": char_count,
            "spatial_hits": spatial_hits,
            "count_hits": count_hits,
            "choice_hits": choice_hits,
            "has_question_mark": has_question_mark,
            "dataset_name": dataset_name,
            "tag": tag,
        }

    def _build_gate_feature_tensor(self, entries: List[dict]) -> torch.Tensor:
        feats = []
        for entry in entries:
            meta = self._entry_metadata(entry)
            feats.append([
                float(meta["has_image"]),
                float(meta["image_count"]),
                float(meta["has_video"]),
                min(float(meta["tag_block_count"]), 16.0),
                min(float(meta["newline_count"]), 32.0),
                min(float(meta["word_count"]), 512.0) / 512.0,
                min(float(meta["char_count"]), 4096.0) / 4096.0,
                min(float(meta["spatial_hits"]), 16.0) / 16.0,
                min(float(meta["count_hits"]), 8.0) / 8.0,
                min(float(meta["choice_hits"]), 8.0) / 8.0,
                float(meta["has_question_mark"]),
                float("mme" in meta["dataset_name"].lower() or "vsi" in meta["dataset_name"].lower()),
                float(meta["tag"].lower() == "2d"),
                float(meta["tag"].lower() == "3d"),
            ])
        return torch.tensor(feats, dtype=torch.float32)

    def _heuristic_force_vggt(self, entry: dict) -> bool:
        meta = self._entry_metadata(entry)
        if self.gate_enable_for_images_only and (not meta["has_image"] or meta["has_video"]):
            return False
        if meta["spatial_hits"] > 0:
            return True
        if meta["image_count"] > 1 and (meta["count_hits"] > 0 or meta["choice_hits"] > 0):
            return True
        return False

    def _predict_gate_labels(self, entries: List[dict]) -> List[str]:
        if not entries:
            return []

        if self.gate_ckpt:
            features = self._build_gate_feature_tensor(entries)
            with torch.no_grad():
                logits = self.gate_model(features)
                probs = torch.sigmoid(logits)
            labels = []
            for entry, prob in zip(entries, probs.tolist()):
                meta = self._entry_metadata(entry)
                if self.gate_enable_for_images_only and (not meta["has_image"] or meta["has_video"]):
                    labels.append("no_vggt")
                else:
                    labels.append("force_vggt" if prob >= self.gate_threshold else "no_vggt")
            return labels

        if not self.gate_use_heuristic_fallback:
            return ["no_vggt" for _ in entries]

        return ["force_vggt" if self._heuristic_force_vggt(entry) else "no_vggt" for entry in entries]

    def _apply_gate_to_entries(self, entries: List[dict], labels: List[str]) -> List[dict]:
        updated_entries = []
        for entry, label in zip(entries, labels):
            new_entry = dict(entry)
            new_entry["conversations"] = [dict(conv) for conv in entry.get("conversations", [])]

            if label != "force_vggt":
                updated_entries.append(new_entry)
                continue

            has_image = "image" in new_entry and bool(new_entry.get("image"))
            has_video = "video" in new_entry and bool(new_entry.get("video"))
            if self.gate_enable_for_images_only and (not has_image or has_video):
                updated_entries.append(new_entry)
                continue

            if new_entry["conversations"]:
                current_value = new_entry["conversations"][0].get("value", "")
                new_entry["conversations"][0]["value"] = self._inject_vggt_after_first_tag_block(current_value)
            updated_entries.append(new_entry)
        return updated_entries

    def _entry_uses_vggt(self, entry: dict) -> bool:
        conversations = entry.get("conversations", [])
        if not conversations:
            return False
        first_value = conversations[0].get("value", "")
        return isinstance(first_value, str) and (VGGT_TAG in first_value)

    def _log_gate_results(self, entries: List[dict], labels: List[str], answers: List[str]) -> None:
        if self.rank != 0:
            return

        total = len(entries)
        used_vggt = sum(1 for entry in entries if self._entry_uses_vggt(entry))
        print(f"[geos_multi_gate] batch_size={total} used_vggt={used_vggt} no_vggt={total - used_vggt}")

        for entry, label, answer in zip(entries, labels, answers):
            sample_id = entry.get("id", "")
            used = self._entry_uses_vggt(entry)
            print(
                f"[geos_multi_gate] sample_id={sample_id} gate_label={label} "
                f"use_vggt={used} answer={repr(answer)}"
            )

    def _generate_batch_single_round(self, batch: dict, gen_kwargs: dict | None = None) -> List[str]:
        gen_kwargs = {} if gen_kwargs is None else dict(gen_kwargs)
        gen_kwargs.setdefault("max_new_tokens", 4096)
        gen_kwargs.setdefault("temperature", 0)
        gen_kwargs.setdefault("top_p", None)
        gen_kwargs.setdefault("num_beams", 1)

        model_inputs = {k: v for k, v in batch.items() if k in self._allowed_input_keys and v is not None}
        device = self._target_device()
        model_inputs = self._to_device_tree(model_inputs, device)

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

        in_lens = []
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None and getattr(attention_mask, "dim", lambda: 0)() == 2:
            in_lens = attention_mask.sum(dim=1).tolist()
        if not in_lens:
            pad_id = self.tokenizer.pad_token_id
            for row in batch["input_ids"]:
                in_lens.append(int((row != pad_id).sum().item()) if pad_id is not None else int(row.numel()))

        trimmed = [out_ids[length:] for length, out_ids in zip(in_lens, outputs)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def generate_until(self, requests):
        res = []

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
                doc = self.task_dict[task][split][doc_id[i]]
                dataset_name = task
                if isinstance(doc, dict):
                    dataset_name = doc.get("dataset_name", task) or task
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

            labels = self._predict_gate_labels(entries)
            gated_entries = self._apply_gate_to_entries(entries, labels)

            batch = self._batch_from_entries(gated_entries)
            gen_kwargs = dict(all_gen_kwargs[0]) if all_gen_kwargs else {}
            answers = self._generate_batch_single_round(batch, gen_kwargs)
            self._log_gate_results(gated_entries, labels, answers)

            for ans, ctx in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (ctx, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
