import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

SPAR = {
    "annotation_path": "data/train/spar_7m.jsonl",
    "data_path": "data/media",
    "tag": "3d"
}

SPAR_234K = {
    "annotation_path": "data/train/spar_234k.json",
    "data_path": "data/media",
    "tag": "3d"
}

SPAR_234K_LIM = {
    "annotation_path": "data/train/spar_234k_limit.json",
    "data_path": "data/media",
    "tag": "3d"
}

spar_tool_40k = {
    "annotation_path": "data/train/spar_tool_40k.json",
    "data_path": "data/media",
    "tag": "3d"
}

LLAVA_HOUND = {
    "annotation_path": "data/train/llava_hound_255k.json",
    "data_path": "data/media",
    "tag": "2d"
}

LLAVA_HOUND_64K = {
    "annotation_path": "data/train/llava_hound_64k.json",
    "data_path": "data/media",
    "tag": "2d"
}

llava_hound_sampleN = {
    "annotation_path": "data/train/llava_hound_sample10.json",
    "data_path": "data/media",
    "tag": "2d"
}
llava_hound_tool_10k = {
    "annotation_path": "data/train/llava_hound_tool_10k.json",
    "data_path": "data/media",
    "tag": "2d"
}
llava_hound_tool_10k_multiRoundTrain = {
    "annotation_path": "data/train/llava_hound_tool_10k_multiRoundTrain.json",
    "data_path": "data/media",
    "tag": "2d"
}
SCANNET_DET = {
    "annotation_path": "data/train/scannet_det_train_4frames.json",
    "data_path": "data/media",
    "tag": "3d"
}

SCANREFER = {
    "annotation_path": "data/train/scanrefer_train_32frames.json",
    "data_path": "data/media",
    "tag": "3d"
}

SCAN2CAP = {
    "annotation_path": "data/train/scan2cap_train_32frames.json",
    "data_path": "data/media",
    "tag": "3d"
}

vsi_20k  = {
    "annotation_path": "data/train/vsi_20k_alig.json",
    "data_path": "data/media/VSI-590K",
    "tag": "3d"
}
vsi_50k  = {
    "annotation_path": "data/train/vsi_50k_alig.json",
    "data_path": "data/media/VSI-590K",
    "tag": "3d"
}
vsi_590k  = {
    "annotation_path": "data/train/vsi_590k_alig.jsonl",
    "data_path": "data/media/VSI-590K",
    "tag": "3d"
}
vsi_stage3_1_small_lr_train  = {
    "annotation_path": "data/train/sft_tfft_json/processed/small_lr.json",
    "data_path": "data/media/VSI-590K",
    "tag": "3d"
}
vsi_review = {
    "annotation_path": "data/train/vsi_10k_sampled_inference.jsonl",
    "data_path": "data/media/VSI-590K",
    "tag": "3d"
}

mindcube_train_aug_cgmap_ffr  = {
    "annotation_path": "data/media/MindCube/prompts/training/qwen2.5vl/MindCube_train_aug_cgmap_ffr_out_qwen_sft.json",
    "data_path": "data/media/MindCube",
    "tag": "3d"
}
mindcube_train_raw_qa  = {
    "annotation_path": "data/media/MindCube/prompts/training/qwen2.5vl/MindCube_train_raw_qa_qwen_sft.json",
    "data_path": "data/media/MindCube",
    "tag": "3d"
}
vsi_tfft  = {
    "annotation_path": "data/train/sft_tfft_json/processed/large_lr_multiRound_v2.json",
    "data_path": "data/media/VSI-590K",
    "tag": "3d"
}

laava_vsi_tfft  = {
    "annotation_path": "data/train/llava_vsi_TF_and_FT_cotrain_shuffled.json",
    "data_path": "data/media/VSI-590K",
    "tag": "3d"
}

spatial_general  = {
    "annotation_path": "data/train/spatial_general_shuffle_v2.json",
    "data_path": "data/media/VSI-590K",
    "tag": "3d"
}

spatial_general_v3  = {
    "annotation_path": "data/train/spatial_general_shuffle_v3.json",
    "data_path": "data/media/VSI-590K",
    "tag": "3d"
}

ab_exp_rand100  = {
    "annotation_path": "data/ab_exp/rand100.json",
    "data_path": "data/media/VSI-590K",
    "tag": "3d"
}
ab_exp_100UseVggt  = {
    "annotation_path": "data/ab_exp/rand100_usevggt.json",
    "data_path": "data/media/VSI-590K",
    "tag": "3d"
}
ab_exp_okvqa  = {
    "annotation_path": "data/ab_exp/rand100_A-OKVQA.json",
    "data_path": "data/media/VSI-590K",
    "tag": "3d"
}
vsi_tfft_qwen  = {
    "annotation_path": "data/train/vsi_tfft_recon_singleturn.json",
    "data_path": "data/media/VSI-590K",
    "tag": "3d"
}
vsi_590k_qwen = {
    "annotation_path": "data/train/vsi_590k_alig_fixed.jsonl",
    "data_path": "data/media/VSI-590K",
    "tag": "3d"
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "spar": SPAR,
    "llava_hound": LLAVA_HOUND,
    "scannet_det": SCANNET_DET,
    "scanrefer": SCANREFER,
    "scan2cap": SCAN2CAP,
    "spar_234k": SPAR_234K,
    "llava_hound_64k": LLAVA_HOUND_64K,
    "llava_hound_sampleN":llava_hound_sampleN,
    "llava_hound_tool_10k":llava_hound_tool_10k,
    "spar_tool_40k":spar_tool_40k,
    "llava_hound_tool_10k_multiRoundTrain":llava_hound_tool_10k_multiRoundTrain,
    "vsi_20k": vsi_20k,
    "vsi_50k": vsi_50k,
    "vsi_590k": vsi_590k,
    "vsi_stage3-1_small_lr_train": vsi_stage3_1_small_lr_train,
    "mindcube_train_aug_cgmap_ffr":mindcube_train_aug_cgmap_ffr,
    "mindcube_train_raw_qa":mindcube_train_raw_qa,
    "vsi_tfft":vsi_tfft,
    "llava_vsi_cotrain":laava_vsi_tfft,
    "spatial_general":spatial_general,
    "mind_spatial_general":spatial_general_v3,
    "ab_exp_rand100":ab_exp_rand100,
    "ab_exp_100UseVggt":ab_exp_100UseVggt,
    "ab_exp_okvqa":ab_exp_okvqa,
    "spar_234l":SPAR_234K_LIM,
    "vsi_10kr":vsi_review,
    "vsi_tfft_qwen":vsi_tfft_qwen,
    "vsi_590k_qwen":vsi_590k_qwen
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config["dataset_name"] = dataset_name
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
