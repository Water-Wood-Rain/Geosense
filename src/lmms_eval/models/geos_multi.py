import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union
import re
import copy
import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT
from qwen_vl.data.utils import load_and_preprocess_images

try:
    # from qwen_vl_utils import process_vision_info
    from qwen_vl_utils import extract_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")
# NEW: 直接使用现成的 DataArguments 和数据模块构造函数
from qwen_vl.train.argument import DataArguments  # NEW
from qwen_vl.data.data_qwen import make_supervised_data_module  # NEW
import types, copy  # NEW
import os
from qwen_vl.data import data_qwen as dq  # 复用：VGGT_TAG / _strip_history_media_tags

VGGT_TAG = dq.VGGT_TAG  # "<vggt>"
@register_model("geos_multi")
class geos_multi(lmms):

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        max_length: Optional[int] = None,
        add_frame_index: bool=False,
        stage: Optional[str] = "inference",
        geometry_encoder_type: Optional[str] = None,
        geometry_encoder_path: Optional[str] = None,
        use_geometry_encoder: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.add_frame_index = add_frame_index
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        config = AutoConfig.from_pretrained(pretrained)
        if use_geometry_encoder is not None:
            setattr(config, "use_geometry_encoder", use_geometry_encoder)
        if geometry_encoder_type is not None:
            setattr(config, "geometry_encoder_type", geometry_encoder_type)
        if geometry_encoder_path is not None:
            setattr(config, "geometry_encoder_path", geometry_encoder_path)

        if getattr(config, "use_geometry_encoder", False) or getattr(config, "use_vggt_feature", False):
            load_class = Qwen2_5_VLForConditionalGenerationWithVGGT
            
            eval_logger.info("Using Qwen2_5_VLForConditionalGenerationWithVGGT")
        else:
            load_class = Qwen2_5_VLForConditionalGeneration
            eval_logger.info("Using Qwen2_5_VLForConditionalGeneration")
        if use_flash_attention_2:
            self._model = load_class.from_pretrained(
                pretrained,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = load_class.from_pretrained(pretrained, config=config, torch_dtype="auto", device_map=self.device_map).eval()

        self.max_num_frames = max_num_frames
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels, padding_side="left")
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")

        if max_length is not None:
            eval_logger.warning(f"Setting max_length to {max_length}")
            setattr(self.processor.tokenizer, "model_max_length", max_length)
            setattr(self._tokenizer, "model_max_length", max_length)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        # NEW: 用 DataArguments 作为默认 data_args，并补齐 dataset 真实所需字段
        self._data_args = DataArguments()  # NEW: 直接实例化默认
        # 覆盖/补齐与当前模型/处理器一致的字段
        self._data_args.use_geometry_encoder=True
        self._data_args.geometry_encoder_type = getattr(config, "geometry_encoder_type", "vggt")
        self._data_args.video_max_frames = max_num_frames  # NEW
        self._data_args.image_processor=self.processor.image_processor

        # NEW: DataArguments 原定义里没有 stage / use_geometry_encoder / image_processor，这里按需补充
        setattr(self._data_args, "stage", stage)  # NEW
        


        data_module = make_supervised_data_module(tokenizer=self._tokenizer, data_args=self._data_args)
        self.train_dataset = data_module["train_dataset"]
        self.data_collator = data_module["data_collator"]

        self._allowed_input_keys = {
            "input_ids", "attention_mask", 
            "past_key_values", "inputs_embeds",
            "pixel_values", "pixel_values_videos",
            "image_grid_thw", "video_grid_thw",
            "rope_deltas", "cache_position", "second_per_grid_ts",
            "geometry_encoder_inputs", "boxes",
        }

        # import os
        # if os.getenv("Debug", "False")=="True":
        #     from remote_pdb import RemotePdb
        #     RemotePdb('127.0.0.1', 18457).set_trace()

        

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list
    def _target_device(self):
        # 与你“原始正常的代码”保持一致的判定
        return "cuda" if self.device_map == "auto" else self._device


    def _to_device_tree(self, obj, device):
        # 递归把 dict / list / tuple / Tensor 全搬到同一 device
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_device_tree(x, device) for x in obj)
        if isinstance(obj, dict):
            return {k: self._to_device_tree(v, device) for k, v in obj.items()}
        return obj

    def _batch_from_entries(self, entries: list[dict]) -> dict:
        assert self.train_dataset is not None and self.data_collator is not None, \
            "train_dataset / data_collator 还未初始化；请先用 make_supervised_data_module(...) 注入。"

        samples, metas = [], []
        for e in entries:
            samples.append(self.train_dataset.build_from_entry(e))

            metas.append({
                "id": e.get("id"),
                "data_source": e.get("data_source", "lmms_eval"),
                "data_path": e.get("data_path", ""),
                "tag": e.get("tag", "2d"),
                "dataset_name": e.get("dataset_name", "lmms_eval"),
                "orig_conversations": copy.deepcopy(e.get("conversations", [])),
                **({"video": e["video"]} if "video" in e else {}),
                **({"image": e["image"]} if "image" in e else {}),
            })

        batch = self.data_collator(samples)

        # collator 可能已经给了 batch["meta"]（stage 含 "generation" 时），但不保证包含我们要的字段；
        # 这里统一再挂一个 meta_vg，后续逻辑优先读 meta_vg。
        batch["meta_vg"] = metas
        return batch

    # -----------------------------
    # 2) helper：把 <vggt> 插到“首个连续 <...> tag 块”之后（与需求一致）
    #    注意：只对“当前问题”做，不对 history 做（history 必须全删 <...>）
    # -----------------------------
    _LEADING_TAG_BLOCK = re.compile(r"^(\s*(?:<[^<>]+>)+)(\s*)", flags=re.S)

    def _inject_vggt_after_first_tag_block(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.replace(VGGT_TAG, "")  # 避免重复注入
        m = self._LEADING_TAG_BLOCK.match(text)
        if m:
            tags, ws = m.group(1), m.group(2)
            rest = text[m.end():]
            return f"{tags}{VGGT_TAG}{ws}{rest}"
        # 没有任何 <...>，兜底：前置 <vggt>（让 preprocess_qwen_2_visual 能触发 vggt_use=True）
        return f"{VGGT_TAG}\n{text}"

    # -----------------------------
    # 3) helper：按 data_qwen.flatten_multiturn_to_singleturn_samples 的格式拼 history
    #    严格使用 dq._strip_history_media_tags：删除所有 <...>
    # -----------------------------
    def _build_history_prefix(self, user0: str, ans0: str) -> str:
        u = dq._strip_history_media_tags(user0).strip()
        a = dq._strip_history_media_tags(ans0).strip()
        lines = []
        if u:
            lines.append(f"User: {u}")
        if a:
            lines.append(f"Assistant: {a}")
        return ("Conversation history:\n" + "\n".join(lines) + "\n\n") if lines else ""

    # -----------------------------
    # 4) 两轮 generate（最多两轮）
    #    第一轮检测输出是否含 <vggt>：
    #      - 无：直接返回
    #      - 有：构造第二轮 entry：history + (问题首个 <...> 块后注入 <vggt>)，再走一遍 dataset pipeline
    # -----------------------------
    def _generate_batch(self, batch: dict, gen_kwargs: dict | None = None) -> list[str]:
        gen_kwargs = {} if gen_kwargs is None else dict(gen_kwargs)
        gen_kwargs.setdefault("max_new_tokens", 4096)
        gen_kwargs.setdefault("temperature", 0)
        gen_kwargs.setdefault("top_p", None)
        gen_kwargs.setdefault("num_beams", 1)

        # ===== 4.1 第 1 轮 =====
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

        # 用真实输入长度裁剪
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
        print(ans1)
        need = [i for i, a in enumerate(ans1) if isinstance(a, str) and (VGGT_TAG in a)]
        if not need:
            return final

        # ===== 4.2 构造第 2 轮 entries（仅对触发样本）=====
        metas = batch.get("meta_vg", None)
        if metas is None:
            # 兜底：如果没有 meta_vg，就尝试用 dataset/collator 产出的 meta
            metas = batch.get("meta", [{} for _ in range(len(final))])

        second_entries, back_map = [], []
        for i in need:
            m = metas[i] if i < len(metas) else {}

            # 原始第一轮 user 文本（source-of-truth：orig_conversations[0].value）
            orig_conv = m.get("orig_conversations") or m.get("conversations") or []
            user0 = orig_conv[0].get("value", "") if (isinstance(orig_conv, list) and orig_conv) else ""

            # history：第一轮 user + 第一轮模型输出（清除所有 <...>）
            hist = self._build_history_prefix(user0, ans1[i])

            # 第二轮问题：在“首个连续 <...> tag 块”后注入 <vggt>
            q2 = self._inject_vggt_after_first_tag_block(user0)

            e2 = {
                "id": m.get("id"),
                "conversations": [{"from": "human", "value": hist + (q2 or "")}],
                "data_source": m.get("data_source", "lmms_eval"),
                "data_path": m.get("data_path", ""),
                "tag": m.get("tag", "2d"),
                "dataset_name": m.get("dataset_name", "lmms_eval"),
            }
            if "video" in m: e2["video"] = m["video"]
            if "image" in m: e2["image"] = m["image"]

            second_entries.append(e2)
            back_map.append(i)

        # ===== 4.3 第 2 轮：再次完整复用 dataset pipeline =====
        batch2 = self._batch_from_entries(second_entries)

        model_inputs2 = {k: v for k, v in batch2.items() if k in self._allowed_input_keys and v is not None}
        model_inputs2 = self._to_device_tree(model_inputs2, device)

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
        print("=======================",ans2 )
        # 回填到原 batch 的对应样本
        for j, orig_i in enumerate(back_map):
            if j < len(ans2):
                if VGGT_TAG not in ans2[j]:
                    final[orig_i] = f"{VGGT_TAG} {ans2[j]}"
                else:
                    final[orig_i] = ans2[j]

        return final

    # ------------------------- generate_until: 不需要改动（继续复用） -------------------------
    def generate_until(self, requests: List["Instance"]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]; split = split[0]
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
                    "dataset_name": "lmms_eval",
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
            answers = self._generate_batch(batch, gen_kwargs)

            for ans, ctx in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (ctx, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res


    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")