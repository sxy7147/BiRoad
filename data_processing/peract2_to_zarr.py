import argparse
import json
import os
import pickle
import shutil

from numcodecs import Blosc
import zarr
import numpy as np
from PIL import Image
from tqdm import tqdm

from data_processing.rlbench_utils import (
    keypoint_discovery,
    image_to_float_array,
    store_instructions
)


# NCAM = 3
# NHAND = 2
# IM_SIZE = 256
DEPTH_SCALE = 2**24 - 1


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Tuples: (name, type, default, nargs)
    arguments = [
        ('root', str, './data/raw', '?'),
        ('tgt', str, './data/zarr_datasets/ratio_95_5/', '?'),
        ('split', str, ['train', 'val'], '*'),
        ('tasks', str, [
            'bimanual_handover_item_easy',
            'bimanual_handover_item_easy_symmetric',
        ], '*'),
        ('im_size', int, 256, '?'),
        ('ncam', int, 3, '?'),
        ('nhand', int, 2, '?'),
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2], nargs=arg[3])

    return parser.parse_args()


def all_tasks_main(
    split,
    tasks,
    im_size=256,
    ncam=3,
    nhand=2,
):
    # Check if the zarr already exists
    filename = f"{STORE_PATH}/{split}.zarr"
    if os.path.exists(filename):
        print(f"Zarr file {filename} already exists.")
        response = input(f"Do you want to overwrite {filename}? (y/n): ").lower()
        if response == 'y':
            print(f"Overwriting {filename}...")
            if os.path.isdir(filename):
                shutil.rmtree(filename)
            else:
                os.remove(filename)
        else:
            print("Skipping...")
            return None

    cameras = ["front", "wrist_left", "wrist_right"]
    if ncam == 1:
        cameras = ["front"]
    
    if isinstance(tasks, dict):
        task_list = list(tasks.keys())
    else:
        task_list = tasks
    
    task2id = {task: t for t, task in enumerate(task_list)}

    # Initialize zarr
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    with zarr.open_group(filename, mode="w") as zarr_file:

        def _create(field, shape, dtype):
            zarr_file.create_dataset(
                field,
                shape=(0,) + shape,
                chunks=(1,) + shape,
                compressor=compressor,
                dtype=dtype
            )

        _create("rgb", (ncam, 3, im_size, im_size), "uint8")
        _create("depth", (ncam, im_size, im_size), "float16")
        _create("proprioception", (3, nhand, 8), "float32")
        _create("action", (1, nhand, 8), "float32")
        _create("proprioception_joints", (1, nhand, 8), "float32")
        _create("action_joints", (1, nhand, 8), "float32")
        _create("extrinsics", (ncam, 4, 4), "float16")
        _create("intrinsics", (ncam, 3, 3), "float16")
        _create("task_id", (), "uint8")
        _create("variation", (), "uint8")

    # Loop through episodes
    if isinstance(tasks, dict):
        task_list = list(tasks.keys())
        task_counts = tasks
    else:
        task_list = tasks
        task_counts = {t: -1 for t in tasks}

    for task in task_list:
        print(task)
        task_folder = f'{ROOT}/{split}/{task}/all_variations/episodes'
        episodes = sorted(os.listdir(task_folder))
        
        # Limit episodes if count is specified
        count = task_counts.get(task, -1)
        if count > 0:
            episodes = episodes[:count]
            print(f"Limiting {task} to {len(episodes)} episodes")

        for ep in tqdm(episodes):
                # Read low-dim file from RLBench
                ld_file = f"{task_folder}/{ep}/low_dim_obs.pkl"
                with open(ld_file, 'rb') as f:
                    demo = pickle.load(f)

                # Keypose discovery
                key_frames = keypoint_discovery(demo)
                key_frames.insert(0, 0)

                # Loop through keyposes and store:
                # RGB (keyframes, cameras, 3, 256, 256)
                rgb = np.stack([
                    np.stack([
                        np.array(Image.open(
                            f"{task_folder}/{ep}/{cam}_rgb/rgb_{_num2id(k)}.png"
                        ))
                        for cam in cameras
                    ])
                    for k in key_frames[:-1]
                ])
                rgb = rgb.transpose(0, 1, 4, 2, 3)

                # Depth (keyframes, cameras, 256, 256)
                depth_list = []
                for k in key_frames[:-1]:
                    cam_d = []
                    for cam in cameras:
                        depth = image_to_float_array(Image.open(
                            f"{task_folder}/{ep}/{cam}_depth/depth_{_num2id(k)}.png"
                        ), DEPTH_SCALE)
                        near = demo[k].misc[f'{cam}_camera_near']
                        far = demo[k].misc[f'{cam}_camera_far']
                        depth = near + depth * (far - near)
                        cam_d.append(depth)
                    depth_list.append(np.stack(cam_d).astype(np.float16))
                depth = np.stack(depth_list)

                # Proprioception (keyframes, 3, 2, 8)
                states = np.stack([np.concatenate([
                    demo[k].left.gripper_pose, [demo[k].left.gripper_open],
                    demo[k].right.gripper_pose, [demo[k].right.gripper_open]
                ]) for k in key_frames]).astype(np.float32)
                # Store current eef pose as well as two previous ones
                prop = states[:-1]
                prop_1 = np.concatenate([prop[:1], prop[:-1]])
                prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])
                prop = np.concatenate([prop_2, prop_1, prop], 1)
                prop = prop.reshape(len(prop), 3, nhand, 8)

                # Action (keyframes, 1, 2, 8)
                actions = states[1:].reshape(len(states[1:]), 1, nhand, 8)

                # Proprioception in joints (keyframes, 3, 2, 8)
                states = np.stack([np.concatenate([
                    demo[k].left.joint_positions,
                    [demo[k].left.gripper_open],
                    demo[k].right.joint_positions,
                    [demo[k].right.gripper_open]
                ]) for k in key_frames]).astype(np.float32)
                prop_jnts = states[:-1].reshape(len(states[:-1]), 1, nhand, 8)

                # Action in joints (keyframes, 1, 2, 8)
                actions_jnts = states[1:].reshape(len(states[1:]), 1, nhand, 8)

                # Extrinsics (keyframes, cameras, 4, 4)
                extrinsics = np.stack([
                    np.stack([
                        demo[k].misc[f'{cam}_camera_extrinsics'].astype(np.float16)
                        for cam in cameras
                    ])
                    for k in key_frames[:-1]
                ])

                # Intrinsics (keyframes, cameras, 3, 3)
                intrinsics = np.stack([
                    np.stack([
                        demo[k].misc[f'{cam}_camera_intrinsics'].astype(np.float16)
                        for cam in cameras
                    ])
                    for k in key_frames[:-1]
                ])

                # Task id (keyframes,)
                task_id = np.array([task2id[task]] * len(key_frames[:-1]))
                task_id = task_id.astype(np.uint8)

                # Variation (keyframes,)
                with open(f"{task_folder}/{ep}/variation_number.pkl", 'rb') as f:
                    var_ = pickle.load(f)
                var_ = np.array([int(var_)] * len(key_frames[:-1]))
                var_ = var_.astype(np.uint8)

                # Write
                zarr_file['rgb'].append(rgb)
                zarr_file['depth'].append(depth)
                zarr_file['proprioception'].append(prop)
                zarr_file['action'].append(actions)
                zarr_file['proprioception_joints'].append(prop_jnts)
                zarr_file['action_joints'].append(actions_jnts)
                zarr_file['extrinsics'].append(extrinsics)
                zarr_file['intrinsics'].append(intrinsics)
                zarr_file['task_id'].append(task_id)
                zarr_file['variation'].append(var_)


def _num2id(int_):
    str_ = str(int_)
    return '0' * (4 - len(str_)) + str_


if __name__ == "__main__":
    args = parse_arguments()
    ROOT = args.root
    STORE_PATH = args.tgt
    splits = args.split
    tasks_raw = args.tasks
    im_size = args.im_size
    ncam = args.ncam
    nhand = args.nhand

    # Process tasks to support task:count format
    tasks = {}
    for task_str in tasks_raw:
        if ':' in task_str:
            name, count = task_str.split(':')
            tasks[name] = int(count)
        else:
            tasks[task_str] = -1

    # Create zarr data
    for split in splits:
        all_tasks_main(
            split,
            tasks,
            im_size=im_size,
            ncam=ncam,
            nhand=nhand,
        )
    # Store instructions as json (can be run independently)
    os.makedirs(STORE_PATH, exist_ok=True)
    instr_dict = store_instructions(ROOT, list(tasks.keys()), splits=splits)
    instr_path = os.path.join(STORE_PATH, 'instructions.json')
    print(f"Storing instructions to {instr_path}...")
    with open(instr_path, 'w') as fid:
        json.dump(instr_dict, fid)
