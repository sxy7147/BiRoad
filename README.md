# BiRoAD: Learning Shared and Role-Adaptive Representations for Bimanual Manipulation

**CoRL 2026**

**[Project page](https://sxy7147.github.io/biroad-website/) · [Paper](https://arxiv.org/pdf/2609.23445) · [Video and real-world demonstrations](https://sxy7147.github.io/biroad-website/#overview)**

![BiRoAD overview: bimanual tasks, imbalanced role distributions, and role-adaptive feature decomposition.](fig/teaser.png)

This repository provides the **3D FlowMatch Actor (3DFA) implementation of BiRoAD**, with tools for data preprocessing, training, and online inference in RLBench. Pretrained checkpoints are available [below](#pretrained-checkpoints). Test data is available in the [evaluation section](#inference-and-online-evaluation).

## Installation

Use a Linux machine with an NVIDIA GPU. Run all commands from the repository root in **Bash**.

```bash
conda create -n biroad python=3.10 -y
conda activate biroad

# Install a CUDA-enabled PyTorch build and its matching torchvision version.
python -m pip install torch torchvision
python -m pip install numpy pillow einops tqdm "transformers<5" \
    "zarr<3" numcodecs diffusers kornia tensorboard natsort "imageio[ffmpeg]"
python -m pip install git+https://github.com/openai/CLIP.git
python -m pip install -e .
mkdir -p ../hf_cache
```

Use `zarr<3` for compatibility with the dataset loader. Install CLIP from the GitHub repository shown above. The training and evaluation scripts disable Hugging Face downloads by default (offline mode). Before the first run, set `HF_HUB_OFFLINE=0` and `TRANSFORMERS_OFFLINE=0` in the script you plan to use to allow pretrained model downloads.

### RLBench simulation environment

Training from Zarr data does not require RLBench. For data preprocessing and online evaluation, install the [BiRoad_Tasks](https://github.com/Tracywhy/BiRoad_Tasks) simulation environment, which includes all 16 benchmark task variants.

Install it into the same `biroad` environment created above. From this repository root:

```bash
conda activate biroad
git clone https://github.com/Tracywhy/BiRoad_Tasks.git ../BiRoad_Tasks
PYTHON_BIN=python bash ../BiRoad_Tasks/scripts/setup_environment.sh
source ../BiRoad_Tasks/scripts/env.sh
```

See the [task-suite installation guide](https://github.com/Tracywhy/BiRoad_Tasks#installation) for system requirements and headless rendering.

## Data preprocessing

### Benchmark tasks

The benchmark contains eight task families, each with original and role-swapped variants (16 variants in total). Ratios such as 95/5 and 50/50 refer to original / role-swapped demonstrations per task family. Task names containing `_symmetric` denote the role-swapped variants.

<details>
<summary>Full task list and task-ID mapping</summary>

The task order defines the task IDs used during data preprocessing and training. When using custom or existing Zarr datasets, match the training scripts' `all_tasks` order to the task order used during preprocessing.

```bash
all_tasks=(
    bimanual_handover_item_easy_random
    bimanual_handover_item_easy_symmetric_random
    bimanual_pick_laptop_random
    bimanual_pick_laptop_symmetric_random
    bimanual_pick_plate_random
    bimanual_pick_plate_symmetric_random
    bimanual_put_bottle_in_fridge
    bimanual_put_bottle_in_fridge_symmetric
    bimanual_put_item_in_drawer_random
    bimanual_put_item_in_drawer_symmetric_random
    bimanual_sweep_to_dustpan_random
    bimanual_sweep_to_dustpan_symmetric_random
    bimanual_take_tray_out_of_oven
    bimanual_take_tray_out_of_oven_symmetric
    coordinated_take_shoes_out_of_box
    coordinated_take_shoes_out_of_box_symmetric
)
```

</details>

### Preprocess raw demonstrations

Generate raw training and validation demonstrations using the [BiRoad_Tasks dataset generator](https://github.com/Tracywhy/BiRoad_Tasks#dataset-generation). The preprocessing script packages these demonstrations into Zarr datasets for training. Arrange disjoint training, validation, and test episodes as follows:

```text
data/
├── raw/train/<task>/all_variations/episodes/<episode>/
├── raw/val/<task>/all_variations/episodes/<episode>/
└── test_raw/<task>/all_variations/episodes/<episode>/
```

The default training setup uses 95/5 training data and 50/50 validation data. Run [run_peract2_role_adaptation_to_zarr.sh](data_processing/run_peract2_role_adaptation_to_zarr.sh) twice: first set `original_count=95` and `role_swapped_count=5` inside the script, then change both to `50` and run again:

```bash
bash data_processing/run_peract2_role_adaptation_to_zarr.sh
```

Each run preprocesses both training and validation splits with the selected counts, saving the datasets under `data/zarr_datasets/ratio_<original>_<swapped>/`. Skip this step if the preprocessed datasets are already available. Test episodes are used directly during evaluation and do not need this preprocessing step.

## Training

Use the [BiRoAD training script](scripts/rlbench/train_biroad_95_5.sh) or [3DFA baseline script](scripts/rlbench/train_3dfa_95_5.sh). Both train on the 95/5 split and validate on the 50/50 split. For example:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/rlbench/train_biroad_95_5.sh
```

Choose your GPUs with `CUDA_VISIBLE_DEVICES`. If your datasets are stored elsewhere, update the data paths inside the script.

Checkpoints are saved to `train_logs/<main_dir>/<run_log_dir>/`: `last.pth` is used to resume training automatically when present, and `best.pth` is used for inference. Change `main_dir` to start a separate experiment. Monitor with `tensorboard --logdir train_logs`.

## Pretrained checkpoints

Download a checkpoint (~422 MB each) below and save it under `checkpoints/`. Optional SHA-256 checksums are available in the [download folder](https://drive.google.com/drive/folders/1Dl1djKwliIzvr6e9Z4sc8M0vcxQOYGf1).

| Model | Ratio | Checkpoint |
| --- | --- | --- |
| BiRoAD | 95/5 | [biroad_95_5.pth](https://drive.google.com/file/d/1a_zH-hwVV9TWeAfjBtulbAcIrOJ7aFs-/view?usp=drive_link) |
| 3DFA | 95/5 | [3dfa_95_5.pth](https://drive.google.com/file/d/1pl9KqXjeb99AUqhtD-VbxeUUM4PAKrLv/view?usp=drive_link) |
| BiRoAD | 50/50 | [biroad_50_50.pth](https://drive.google.com/file/d/1YZ8Durh9STl9qLVq3x4UWZwaBVWFOGe_/view?usp=drive_link) |
| 3DFA | 50/50 | [3dfa_50_50.pth](https://drive.google.com/file/d/1kJL8a4WKBjtFF5-zqv6oXln2x0xALTsu/view?usp=sharing) |

## Inference and online evaluation

Download the [test dataset](https://drive.google.com/file/d/1hxFM-XP5-UU_mbfpmzjOP_PWP798p9AE/view?usp=drive_link) (~160 MB), containing 50 episodes for each of the 16 task variants. Optional checksums are available in the [download folder](https://drive.google.com/drive/folders/1KzHMAFeFOLEq5ZFY2RC0iXrCamUK565s).

Extract the ZIP and place its `test/` directory at `data/test_raw/`, so tasks are directly under `data/test_raw/<task>/all_variations/episodes/`.

Use the [BiRoAD evaluation script](online_evaluation_rlbench/scripts/eval_biroad.sh) or [3DFA baseline script](online_evaluation_rlbench/scripts/eval_3dfa.sh). Set `checkpoint` inside the script to your downloaded or trained model, for example:

```bash
checkpoint=./checkpoints/biroad_95_5.pth
```

With the [RLBench simulation environment](#rlbench-simulation-environment) installed and `xvfb-run` available, run:

```bash
CUDA_VISIBLE_DEVICES=0 N_WORKERS=1 bash online_evaluation_rlbench/scripts/eval_biroad.sh
```

To disable video recording, set `save_video=false` inside the evaluation script.

Results are saved under `eval_logs/<exp>/<checkpoint_alias>/seed<seed>/`, including per-task `eval.json`, videos when enabled, and an automatically generated `all_results.txt` summary.

## Citation

```bibtex
@inproceedings{shen2026biroad,
  title = {{BiRoAD}: Learning Shared and Role-Adaptive Representations for Bimanual Manipulation},
  author = {Shen, Yan and Liu, Yuchen and Jiang, Feng and Hu, Hangtian and Li, Xiaoqi and Chen, Shu and Wu, Ruihai and Dong, Hao},
  booktitle = {Proceedings of the Conference on Robot Learning (CoRL)},
  year = {2026}
}
```

## Acknowledgements and license

This implementation builds on [3D FlowMatch Actor](https://github.com/nickgkan/3d_flowmatch_actor) and its [paper](https://arxiv.org/abs/2508.11002), with components adapted from [3D Diffuser Actor](https://github.com/nickgkan/3d_diffuser_actor). The simulation setup builds on [PerAct2](https://github.com/markusgrotz/peract_bimanual) and [RLBench](https://github.com/markusgrotz/RLBench). We thank the authors for making their work available.

The repository retains the upstream [MIT license and copyright notice](LICENSE).
