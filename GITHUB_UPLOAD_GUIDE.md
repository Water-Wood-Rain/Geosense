# VG-LLM GitHub Upload Guide

This guide is for preparing the VG-LLM source code for a GitHub repository. It is written so the useful parts can later be moved into `README.md`.

The release is organized around two public entrypoints:

- Training: `scripts/train/train_sr.sh`
- Evaluation: `scripts/evaluation/eval_geos_multi.sh`

Dataset names and JSON annotation paths should follow `src/qwen_vl/data/__init__.py`, starting from `SPAR`. Do not treat the first four placeholder entries, `cambrian_737k`, `mp_doc`, `clevr_mc`, and `videochatgpt`, as required release datasets.

## 1. Environment File

Use a simple `requirements.txt` for the public release. The project was developed locally in a conda environment named `geos`, but the public README should use the release environment name `geos`. Users should create `geos` and install dependencies from `requirements.txt`.

Recommended README setup:

```bash
conda create -n geos python=3.10
conda activate geos
pip install -r requirements.txt
pip install -e . --no-deps
```

Upload these environment/package files:

| Path | Upload | Notes |
| --- | --- | --- |
| `requirements.txt` | Yes | Main dependency file for the public `geos` conda environment. |
| `setup.py` | Yes | Keep it because `pip install -e . --no-deps` installs the local `src/` packages cleanly. Dependencies should be read from `requirements.txt`. |

Do not upload a full conda environment export unless needed. A full `conda env export` usually includes machine-specific build strings and extra packages.

## 2. Release Rename TODO: `geos` to `geos`

Before uploading the final source release, replace public/runtime string identifiers from `geos` to `geos` across every file that will be uploaded. This is not limited to bash scripts; include Python registries, config strings, README text, package metadata, and any string variables used by runtime lookup.

Run a text audit on the upload set before the final commit:

```bash
rg -n "geos|VG-LLM|VG_LLM" README.md setup.py requirements.txt scripts src
```

Required release-name replacements:

| Current string | Release string | Where to check |
| --- | --- | --- |
| `geos` | `geos` | Package metadata, script variables, README text, model args, logs/tags intended for release. |
| `geos_multi` | `geos_multi` | `lmms_eval` model registry, `--model` arguments, evaluation wrappers, task scripts. |
| `geos_multi_gate` | `geos_multi_gate` | Optional gate evaluation wrappers if uploaded. |
| `geos_multi_latency` | `geos_multi_latency` | Optional latency wrappers if uploaded. |
| `VG-LLM`, `VG_LLM` | `GEOS` | Public-facing docs, badges, headings, descriptions, and project metadata. |

Important files to inspect for string-level runtime names:

- `setup.py`: package name, description, and README metadata.
- `scripts/evaluation/eval_geos_multi.sh`: `--model`, stage/model tags, and output labels.
- `scripts/train/train_sr.sh`: environment-facing labels and output tags.
- `src/lmms_eval/models/__init__.py`: model registry keys and class lookup strings.
- `src/lmms_eval/models/geos_multi.py`: registered model name and any class/string identifiers.
- Any optional uploaded wrappers such as `geos_multi_gate.py`, `geos_multi_latency.py`, or latency scripts.

If file names are renamed, update imports and script paths consistently. For example, if `src/lmms_eval/models/geos_multi.py` becomes `src/lmms_eval/models/geos_multi.py`, then `AVAILABLE_MODELS`, `@register_model(...)`, class names, and `--model geos_multi` must all agree.

## 3. Source Code to Upload

Upload source code, lightweight configs, scripts, and documentation.

| Path | Upload | Why |
| --- | --- | --- |
| `README.md` | Yes | Main project documentation. |
| `GITHUB_UPLOAD_GUIDE.md` | Optional | Keep until its content is merged into `README.md`. |
| `requirements.txt` | Yes | Dependency list for the public `geos` environment. |
| `setup.py` | Yes | Editable install support. |
| `.gitignore`, `.gitattributes` | Yes | Git behavior and large-file filtering. |
| `assets/` | Yes | README figures and paper assets. |
| `src/qwen_vl/` | Yes | Training, data loading, model, and VGGT geometry encoder code. |
| `src/lmms_eval/` | Yes | Evaluation framework and `geos_multi` model wrapper. |
| `src/evaluate_vsi_bench/` | Recommended | VSI-Bench post-processing utilities if VSI results are documented. |
| `scripts/train/train_sr.sh` | Yes | Main training command. |
| `scripts/evaluation/eval_geos_multi.sh` | Yes | Main evaluation command. |
| `scripts/zero2_opt.json` | Yes | DeepSpeed config used by `train_sr.sh`. |
| `scripts/preprocess/` | Recommended | JSON annotation preparation utilities. |
| `scripts/data_generation/` | Optional | Include only if the README documents synthetic/VSI generation. |
| `scripts/latency/` | Optional | Include only if latency experiments are documented. |
| `demo.ipynb` | Optional | Include only if demo data/checkpoint instructions are complete. |

## 4. Dataset Upload Rule

For datasets, upload only JSON/JSONL annotation files that are needed for training/evaluation. Do not upload raw images, videos, frame folders, checkpoints, or local dataset symlinks.

Recommended public structure:

```text
data/
|-- train/
|   |-- spar_234k.json
|   |-- llava_hound_64k.json
|   |-- scannet_det_train_4frames.json
|   |-- scanrefer_train_32frames.json
|   |-- scan2cap_train_32frames.json
|   `-- vsi_590k_alig_fixed.jsonl
`-- README.md
```

The local `data` path in this workspace is a symlink to `/remote-home/share/_datasets/Spatial-MLLM`. Do not commit that symlink. For release, create a real `data/train/` folder containing only JSON/JSONL annotations, or keep annotations in a separate release folder and update the README paths consistently.

Recommended `.gitignore` pattern if JSON annotations are uploaded under `data/train/`:

```gitignore
/data/*
!/data/README.md
!/data/train/
!/data/train/**/*.json
!/data/train/**/*.jsonl
/data/media/
```

### Dataset JSONs to Document

Use these keys from `src/qwen_vl/data/__init__.py` after the first four placeholder datasets.

| Dataset key | JSON/JSONL annotation path | Source media path users must prepare |
| --- | --- | --- |
| `spar` | `data/train/spar_7m.jsonl` | `data/media` |
| `spar_234k` | `data/train/spar_234k.json` | `data/media` |
| `spar_234l` | `data/train/spar_234k_limit.json` | `data/media` |
| `spar_tool_40k` | `data/train/spar_tool_40k.json` | `data/media` |
| `llava_hound` | `data/train/llava_hound_255k.json` | `data/media` |
| `llava_hound_64k` | `data/train/llava_hound_64k.json` | `data/media` |
| `llava_hound_sampleN` | `data/train/llava_hound_sample10.json` | `data/media` |
| `llava_hound_tool_10k` | `data/train/llava_hound_tool_10k.json` | `data/media` |
| `llava_hound_tool_10k_multiRoundTrain` | `data/train/llava_hound_tool_10k_multiRoundTrain.json` | `data/media` |
| `scannet_det` | `data/train/scannet_det_train_4frames.json` | `data/media` |
| `scanrefer` | `data/train/scanrefer_train_32frames.json` | `data/media` |
| `scan2cap` | `data/train/scan2cap_train_32frames.json` | `data/media` |
| `vsi_20k` | `data/train/vsi_20k_alig.json` | User-provided VSI-590K media path |
| `vsi_50k` | `data/train/vsi_50k_alig.json` | User-provided VSI-590K media path |
| `vsi_590k` | `data/train/vsi_590k_alig.jsonl` | User-provided VSI-590K media path |
| `vsi_stage3-1_small_lr_train` | `data/train/sft_tfft_json/processed/small_lr.json` | User-provided VSI-590K media path |
| `vsi_10kr` | `data/train/vsi_10k_sampled_inference.jsonl` | User-provided VSI-590K media path |
| `vsi_tfft` | `data/train/sft_tfft_json/processed/large_lr_multiRound_v2.json` | User-provided VSI-590K media path |
| `llava_vsi_cotrain` | `data/train/llava_vsi_TF_and_FT_cotrain_shuffled.json` | User-provided VSI-590K media path |
| `vsi_tfft_qwen` | `data/train/vsi_tfft_recon_singleturn.json` | User-provided VSI-590K media path |
| `vsi_590k_qwen` | `data/train/vsi_590k_alig_fixed.jsonl` | User-provided VSI-590K media path |
| `mindcube_train_aug_cgmap_ffr` | MindCube JSON annotation | User-provided MindCube media path |
| `mindcube_train_raw_qa` | MindCube JSON annotation | User-provided MindCube media path |
| `spatial_general` | Spatial-general JSON annotation | User-provided VSI-590K media path |
| `mind_spatial_general` | Spatial-general JSON annotation | User-provided VSI-590K media path |
| `ab_exp_rand100` | Ablation JSON annotation | User-provided VSI-590K media path |
| `ab_exp_100UseVggt` | Ablation JSON annotation | User-provided VSI-590K media path |
| `ab_exp_okvqa` | Ablation JSON annotation | User-provided VSI-590K media path |

### Source Data Links for README

The README should explain that JSON annotations are included in this repo or downloaded separately, while raw media must be prepared by users from the original dataset pages.

| Data purpose | Hugging Face page | README instruction |
| --- | --- | --- |
| VG-LLM annotation files | https://huggingface.co/datasets/zd11024/VG-LLM-Data | Download annotation JSON/JSONL files and place them under `data/train/`. |
| SPAR source data | https://huggingface.co/datasets/jasonzhango/SPAR-7M | Prepare SPAR media under `data/media/spar/` or update `data_path` in `src/qwen_vl/data/__init__.py`. |
| LLaVA-Video annotations | https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K | Use the LLaVA-Hound split annotations if reproducing `llava_hound*` entries. |
| LLaVA/ShareGPTVideo raw videos | https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/tree/main/train_300k | Prepare extracted frames/videos under `data/media/llava_hound/`. |
| VSI training/eval data | https://huggingface.co/datasets/nyu-visionx/VSI-590K | Prepare the VSI-590K media locally and update VSI `data_path` entries if not using `data/media`. |
| VSI-Bench evaluation | https://huggingface.co/datasets/nyu-visionx/VSI-Bench | Required for `vsibench` evaluation. |
| CV-Bench evaluation | https://huggingface.co/datasets/nyu-visionx/CV-Bench | Required for `cvbench` evaluation. |

If a dataset path in `src/qwen_vl/data/__init__.py` still contains `/remote-home/...`, rewrite it to a relative path or document it as a user-provided local path before release.

## 5. Files and Folders Not to Upload

Keep heavy, local, private, or generated files out of GitHub.

| Path or pattern | Reason |
| --- | --- |
| `data` symlink | Local symlink; replace with JSON-only release folder if needed. |
| `data/media/` | Raw source images/videos/frames should not be committed. |
| `ckpt_saves/` | Local model checkpoints. |
| `train_output/`, `trainer_output/`, `test_output/` | Training outputs and checkpoints. |
| `logs/`, `outputs/`, `scripts/logs/` | Runtime logs and generated evaluation results. |
| `generatedData/`, `generatedData_old/`, `rdgeneratedData/`, `rgeneratedData/` | Generated samples and experiment outputs. |
| `wandb/` | Local experiment tracking cache. |
| `ab_exp_output/` | Ablation outputs. |
| `.claude/`, `.vscode/` | Local editor/agent settings. |
| `tmp/`, `tmp*` | Temporary files. |
| `nohup.out` | Local process log. |
| `runtime_bad_batches.jsonl`, `debug_truncation_result.json` | Runtime/debug artifacts. |
| `vggt_values.txt`, `vggt_vsibench_debug*.jsonl` | Local debugging outputs. |
| `__pycache__/`, `*.pyc`, `*.egg-info/` | Python/build artifacts. |
| `*.bak`, `*.backup*`, `*.backup_*` | Local backup copies. |

## 6. Step-by-step GitHub Upload

Use these steps after your GitHub repository has already been created in your account.

1. Enter the project folder.

```bash
cd /attached/remote-home-1/liurh/icml26/VG-LLM
```

2. Initialize Git locally if this folder is not already a Git repository.

```bash
git init
git branch -M main
```

3. Connect your GitHub repository. Replace the URL with your own repository URL.

```bash
git remote add origin https://github.com/<YOUR_ACCOUNT>/<YOUR_REPO>.git
```

If `origin` already exists but points to the wrong repository:

```bash
git remote set-url origin https://github.com/<YOUR_ACCOUNT>/<YOUR_REPO>.git
```

4. Check ignored files before staging.

```bash
git status --short --ignored
git check-ignore data/media ckpt_saves train_output logs outputs generatedData wandb
```

5. Stage the core release files.

```bash
git add README.md GITHUB_UPLOAD_GUIDE.md requirements.txt setup.py .gitignore .gitattributes assets
git add src/qwen_vl src/lmms_eval src/evaluate_vsi_bench
git add scripts/train/train_sr.sh scripts/evaluation/eval_geos_multi.sh scripts/zero2_opt.json scripts/preprocess
```

6. Stage dataset JSON annotations only, if you decide to include them in this GitHub repo.

```bash
git add data/README.md data/train
```

Before this command, make sure `data` is a real directory containing JSON/JSONL annotations, not the current local symlink.

7. Review staged files carefully.

```bash
git status --short
git diff --cached --stat
git diff --cached --name-only | grep -E '(ckpt_saves|train_output|data/media|wandb|logs|outputs|__pycache__|\.pyc|nohup.out)' || true
```

The last command should print nothing.

8. Commit and push.

```bash
git commit -m "Prepare VG-LLM source release"
git push -u origin main
```

9. After pushing, open the GitHub page and verify that the repository contains code, scripts, requirements, README, assets, and JSON annotations only. It should not contain checkpoints, raw media, logs, generated outputs, or local backup files.

## 7. Final Checks Before README Conversion

- `requirements.txt` installs inside `conda activate geos`.
- `scripts/train/train_sr.sh` does not expose private checkpoint paths; use `<PATH_TO_INITIAL_VG_LLM_CHECKPOINT>` or a public model path.
- `scripts/evaluation/eval_geos_multi.sh` does not expose private checkpoint paths; use `<PATH_TO_VG_LLM_CHECKPOINT>` or a public model path.
- Dataset JSON files are uploaded or clearly linked.
- Raw source data is documented with Hugging Face links instead of committed to Git.
- No `/remote-home/...` path remains in README instructions except as an example of a local user-provided path.
