#!/usr/bin/env python3
import os
import json
import torch
import traceback

LOG_PATH = os.environ.get("DEBUG_BAD_BATCH_LOG", "runtime_bad_batches.jsonl")
IMAGE_TOKEN_ID = 151655  # qwen image pad token

def jsonl_append(obj, path=LOG_PATH):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def patch_dataset_and_collator():
    from qwen_vl.data import data_qwen as dq

    orig_get_item = dq.LazySupervisedDataset._get_item
    orig_collate = dq.DataCollatorForSupervisedDataset.__call__

    def patched_get_item(self, i):                                                                                                                                                                      
        sample = orig_get_item(self, i)
        raw = self.list_data_dict[i]                                                                                                                                                                    
                
        merge_size = self.data_args.image_processor.merge_size

        img_features = 0
        num_frames = 0
        grid_list = sample.get("image_grid_thw", None)
        if grid_list is not None:
            num_frames = len(grid_list)
            for g in grid_list:
                img_features += int(torch.prod(g).item()) // (merge_size ** 2)

        first_human = ""
        conv = raw.get("conversations", [])
        if conv:
            try:
                first_human = conv[0].get("value", "")[:500]
            except Exception:
                pass

        sample["_debug_meta"] = {
            "dataset_index": i,
            "sample_id": raw.get("id"),
            "turn_id": raw.get("turn_id"),
            "dataset_name": raw.get("dataset_name"),
            "video": raw.get("video"),
            "num_turns": len(conv),
            "first_human_preview": first_human,
        }
        sample["_debug_orig_len"] = int(sample["input_ids"].shape[0])
        sample["_debug_orig_img_tokens"] = int((sample["input_ids"] == IMAGE_TOKEN_ID).sum().item())
        sample["_debug_img_features"] = img_features
        sample["_debug_num_frames"] = num_frames
        return sample

    def patched_collate(self, instances):
        max_len = self.tokenizer.model_max_length
        bad = []
        batch_debug = []

        cleaned = []
        for inst in instances:
            meta = inst.get("_debug_meta", {})
            input_ids = inst["input_ids"]

            orig_len = int(inst.get("_debug_orig_len", input_ids.shape[0]))
            orig_img_tokens = int(inst.get("_debug_orig_img_tokens", (input_ids == IMAGE_TOKEN_ID).sum().item()))
            img_features = int(inst.get("_debug_img_features", 0))
            num_frames = int(inst.get("_debug_num_frames", 0))                                                                                                                                          

            trunc_ids = input_ids[:max_len]                                                                                                                                                             
            trunc_img_tokens = int((trunc_ids == IMAGE_TOKEN_ID).sum().item())

            rec = {
                **meta,
                "orig_len": orig_len,
                "model_max_length": max_len,
                "orig_img_tokens": orig_img_tokens,
                "trunc_img_tokens": trunc_img_tokens,
                "img_features": img_features,
                "num_frames": num_frames,
            }
            batch_debug.append(rec)

            # 只关注有图像特征的样本
            if img_features > 0:
                if orig_img_tokens == 0:
                    rec["reason"] = "orig_tokens_zero"
                    bad.append(rec)
                elif trunc_img_tokens == 0 and orig_img_tokens > 0:
                    rec["reason"] = "trunc_to_zero"
                    bad.append(rec)
                elif 0 < trunc_img_tokens < orig_img_tokens:
                    rec["reason"] = "partial_trunc"
                    bad.append(rec)
                elif trunc_img_tokens != img_features:
                    rec["reason"] = "tokens_features_mismatch_before_forward"
                    bad.append(rec)

            inst2 = dict(inst)
            for k in list(inst2.keys()):
                if k.startswith("_debug_"):                                                                                                                                                             
                    inst2.pop(k, None)
            cleaned.append(inst2)                                                                                                                                                                       
                
        if bad:
            payload = {
                "event": "bad_batch_before_forward",
                "bad_count": len(bad),
                "bad": bad,
                "all_instances": batch_debug,
            }
            print("\n========== BAD BATCH DETECTED BEFORE MODEL FORWARD ==========")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            jsonl_append(payload)
            raise RuntimeError("debug stop before model forward")

        return orig_collate(self, cleaned)

    dq.LazySupervisedDataset._get_item = patched_get_item
    dq.DataCollatorForSupervisedDataset.__call__ = patched_collate


def main():
    patch_dataset_and_collator()

    from qwen_vl.train.train_qwen import train

    try:
        train(attn_implementation="flash_attention_2")
    except Exception as e:
        print("\n========== TRAIN EXIT ==========")
        print(repr(e))
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()