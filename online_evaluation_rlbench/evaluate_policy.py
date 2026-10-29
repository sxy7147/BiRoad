"""Online evaluation script on RLBench."""

import os
import sys

# Since the server has a display (is not headless), we remove the 
# QT_QPA_PLATFORM overrides which were causing plugin search errors.
# We also keep the cv2 mock for Python-level imports just in case.
from unittest.mock import MagicMock
from importlib.machinery import ModuleSpec

mock_cv2 = MagicMock()
mock_cv2.__spec__ = ModuleSpec("cv2", None)
sys.modules["cv2"] = mock_cv2

import argparse
import random
from pathlib import Path
import json

# Add the root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np

from datasets import fetch_dataset_class
from modeling.policy import fetch_model_class
from utils.common_utils import str2bool, str_none, round_floats

from online_evaluation_rlbench.utils_with_bimanual_rlbench import RLBenchEnv, Actioner



def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    # Tuples: (name, type, default)
    arguments = [
        # Testing arguments
        ('checkpoint', str_none, None),
        ('task', str, "close_jar"),
        ('max_tries', int, 10),
        ('max_steps', int, 25),
        ('headless', str2bool, False),
        ('collision_checking', str2bool, False),
        ('seed', int, 0),
        # Dataset arguments
        ('data_dir', Path, Path(__file__).resolve().parent.parent / "data" / "test_raw"),
        ('eval_instructions', str_none, None),
        ('dataset', str, "Peract2_3dfront_3dwrist"),
        ('image_size', str, "256,256"),
        # Logging arguments
        ('output_file', Path, Path(__file__).parent / "eval.json"),
        ('num_demos', int, -1),
        ('from_episode_number', int, 0),
        ('save_video', str2bool, False),
        ('video_dir', Path, Path(__file__).parent / "videos"),
        ('video_resolution', str, "256,256"),
        ('video_camera', str_none, None),
        ('video_freq', int, 5),
        # Model arguments: general policy type
        ('model_type', str, 'denoise3d'),
        ('bimanual', str2bool, True),
        ('prediction_len', int, 1),
        # Model arguments: encoder
        ('backbone', str, "clip"),
        ('fps_subsampling_factor', int, 5),
        # Model arguments: encoder and head
        ('embedding_dim', int, 144),
        ('num_attn_heads', int, 9),
        ('num_vis_instr_attn_layers', int, 2),
        ('num_history', int, 0),
        # Model arguments: head
        ('num_shared_attn_layers', int, 4),
        ('relative_action', str2bool, False),
        ('rotation_format', str, 'quat_xyzw'),
        ('denoise_timesteps', int, 10),
        ('denoise_model', str, "rectified_flow"),
        ('use_sym_anti_decomp', str2bool, False),
        ('sym_anti_direct_update', str2bool, False),
        ('sym_anti_decomp_placement', str, "post_self_attn"),
        ('sym_anti_module_type', str, "sym_anti"),
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


def load_models(args):
    print("Loading model from", args.checkpoint, flush=True)

    model_class = fetch_model_class(args.model_type)
    model = model_class(
        backbone=args.backbone,
        num_vis_instr_attn_layers=args.num_vis_instr_attn_layers,
        fps_subsampling_factor=args.fps_subsampling_factor,
        embedding_dim=args.embedding_dim,
        num_attn_heads=args.num_attn_heads,
        nhist=args.num_history,
        nhand=2 if args.bimanual else 1,
        num_shared_attn_layers=args.num_shared_attn_layers,
        relative=args.relative_action,
        rotation_format=args.rotation_format,
        denoise_timesteps=args.denoise_timesteps,
        denoise_model=args.denoise_model,
        use_sym_anti_decomp=getattr(args, 'use_sym_anti_decomp', False),
        sym_anti_direct_update=getattr(args, 'sym_anti_direct_update', False),
        sym_anti_decomp_placement=getattr(args, 'sym_anti_decomp_placement', "post_self_attn"),
        sym_anti_module_type=getattr(args, 'sym_anti_module_type', "sym_anti"),
    )

    # Load model weights
    model_dict = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    
    model_dict_weight = {}
    for key in model_dict["weight"]:
        _key = key[7:]
        model_dict_weight[_key] = model_dict["weight"][key]
    model.load_state_dict(model_dict_weight, strict=False)
    
    model.eval()

    return model.cuda()


if __name__ == "__main__":
    # Arguments
    args = parse_arguments()
    print("Arguments:")
    print(args)
    print("-" * 100)

    # Save results here
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Create video directory if saving videos
    if args.save_video and args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)

    # Dataset class (for getting cameras and tasks/variations)
    dataset_class = fetch_dataset_class(args.dataset)

    # Load models
    model = load_models(args)
    # print(model.workspace_normalizer)

    # Evaluate - reload environment for each task (crashes otherwise)
    task_success_rates = {}
    for task_str in [args.task]:

        # Seeds - re-seed for each task
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # Load RLBench environment
        env = RLBenchEnv(
            data_path=args.data_dir,
            task_str=task_str,
            image_size=[int(x) for x in args.image_size.split(",")],
            apply_rgb=True,
            apply_pc=True,
            headless=bool(args.headless),
            apply_cameras=dataset_class.cameras,
            collision_checking=bool(args.collision_checking)
        )

        # Actioner (runs the policy online)
        actioner_kwargs = {
            "backbone": args.backbone,
            "task_str": args.task,
        }
        if args.bimanual:
            actioner_kwargs["eval_instructions"] = args.eval_instructions
        actioner = Actioner(model, **actioner_kwargs)

        # Evaluate
        var_success_rates = env.evaluate_task_on_multiple_variations(
            task_str,
            max_steps=args.max_steps,
            actioner=actioner,
            max_tries=args.max_tries,
            prediction_len=args.prediction_len,
            num_history=args.num_history,
            num_demos=int(args.num_demos),
            from_episode_number=int(args.from_episode_number),
            save_video=args.save_video,
            video_dir=args.video_dir,
            video_resolution=tuple(int(x) for x in args.video_resolution.split(",")),
            video_camera=args.video_camera,
            video_freq=args.video_freq
        )
        print()
        print(
            f"{task_str} variation success rates:",
            round_floats(var_success_rates)
        )
        print(
            f"{task_str} mean success rate:",
            round_floats(var_success_rates["mean"])
        )

        task_success_rates[task_str] = var_success_rates
        with open(args.output_file, "w") as f:
            json.dump(round_floats(task_success_rates), f, indent=4)
