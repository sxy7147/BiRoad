"""Main script for training and testing."""

import argparse
import os
from pathlib import Path
import sys

import torch

from datasets import fetch_dataset_class
from modeling.policy import fetch_model_class
from utils.common_utils import str2bool, str_none
from utils.trainers import fetch_train_tester


def str_float_tuple(value):
    if value is None or value.lower() == "none":
        return None
    return tuple(float(x) for x in value.split(","))


def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    # Tuples: (name, type, default)
    arguments = [
        # Dataset/loader arguments
        ('train_data_dir', Path, ''),
        ('eval_data_dir', Path, ''),
        ('train_instructions', Path, ''),
        ('val_instructions', Path, ''),
        ('dataset', str, "Peract2_3dfront_3dwrist"),
        ('num_workers', int, 4),
        ('batch_size', int, 64),
        ('batch_size_val', int, 64),
        ('chunk_size', int, 1),
        ('memory_limit', float, 8),  # cache limit in GB
        ('tasks', str, None, '*'),
        ('all_tasks', str, None, '*'),
        ('balanced_role_sampling', str2bool, False),
        # Logging arguments
        ('base_log_dir', Path, Path(__file__).parent / "train_logs"),
        ('exp_log_dir', Path, "exp"),
        ('run_log_dir', Path, "run"),
        # Training and testing arguments
        ('checkpoint', str_none, None),
        ('val_freq', int, 4000),
        ('interm_ckpt_freq', int, 1000000),
        ('num_best_checkpoints', int, 1),
        ('eval_only', str2bool, False),
        ('lr', float, 1e-4),
        ('backbone_lr', float, 1e-4),
        ('lr_scheduler', str, "constant"),
        ('wd', float, 5e-3),
        ('train_iters', int, 600000),
        ('use_compile', str2bool, False),
        ('use_ema', str2bool, False),
        ('lv2_batch_size', int, 1),
        # Model arguments: general policy type
        ('model_type', str, 'denoise3d'),
        ('bimanual', str2bool, True),
        ('keypose_only', str2bool, True),
        ('pre_tokenize', str2bool, True),
        ('custom_img_size', int, None),
        ('pcd_workspace_bounds', str_float_tuple, None),
        ('pcd_image_filter_radius', float, 0.0),
        ('pcd_image_filter_min_neighbors', int, 0),
        ('pcd_local_density_filter_radius', float, 0.0),
        ('pcd_local_density_filter_min_neighbors', int, 0),
        ('pcd_image_component_filter_min_pixels', int, 0),
        ('pcd_image_component_filter_iterations', int, 32),
        ('workspace_normalizer_buffer', float, 0.04),
        # Model arguments: encoder
        ('backbone', str, "clip"),
        ('finetune_backbone', str2bool, False),
        ('finetune_text_encoder', str2bool, False),
        ('fps_subsampling_factor', int, 5),
        # Model arguments: encoder and head
        ('embedding_dim', int, 120),  # divisible by num_attn_heads
        ('num_attn_heads', int, 8),
        ('num_vis_instr_attn_layers', int, 3),
        ('num_history', int, 1),
        # Model arguments: head
        ('num_shared_attn_layers', int, 4),
        ('relative_action', str2bool, False),
        ('rotation_format', str, 'quat_xyzw'),
        ('denoise_timesteps', int, 10),
        ('denoise_model', str, "rectified_flow"),
        ('data_aug_swap', str2bool, False),
        ('use_sym_anti_decomp', str2bool, False),
        ('sym_anti_direct_update', str2bool, False),
        ('sym_anti_decomp_placement', str, "post_self_attn"),
        ('sym_anti_module_type', str, "sym_anti"),
        # End of arguments
    ]
    for arg in arguments:
        if len(arg) == 4:
            parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2], nargs=arg[3])
        else:
            parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


def suppress_output_on_non_main():
    if int(os.environ.get("RANK", 0)) != 0:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # Arguments
    args = parse_arguments()
    print("Arguments:")
    print(args)
    print("-" * 100)

    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    tensorboard_root = os.environ.get("AIHC_TENSORBOARD_LOG_PATH")
    if tensorboard_root:
        args.tensorboard_log_dir = Path(tensorboard_root) / args.exp_log_dir / args.run_log_dir
    else:
        args.tensorboard_log_dir = log_dir
    args.tensorboard_log_dir.mkdir(exist_ok=True, parents=True)
    print("TensorBoard logging:", args.tensorboard_log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count:", torch.cuda.device_count())
    args.local_rank = int(os.environ["LOCAL_RANK"])
    suppress_output_on_non_main()

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Select dataset and model classes
    dataset_class = fetch_dataset_class(args.dataset)
    model_class = fetch_model_class(args.model_type)

    # Run
    TrainTester = fetch_train_tester(args.dataset)
    train_tester = TrainTester(args, dataset_class, model_class)
    train_tester.main()

    # Safe program termination
    if torch.distributed.is_initialized():
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
