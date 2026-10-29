import random
import torch.nn.functional as F

from .base import BaseDataset


PERACT2_TASKS = [
    'bimanual_push_box',
    'bimanual_lift_ball',
    'bimanual_dual_push_buttons',
    'bimanual_pick_plate',
    'bimanual_put_item_in_drawer',
    'bimanual_put_bottle_in_fridge',
    'bimanual_handover_item',
    'bimanual_pick_laptop',
    'bimanual_straighten_rope',
    'bimanual_sweep_to_dustpan',
    'bimanual_lift_tray',
    'bimanual_handover_item_easy',
    'bimanual_take_tray_out_of_oven'
]


_EVAL_ROLE_SUFFIXES = (" with swapped hands", " using the other hand")
_PARTIAL_ROLE_LANGUAGE_MARKERS = (
    "using only the left arm, ",
    "using only the right arm, ",
    ", using only the left arm",
    ", using only the right arm",
    " with your left arm",
    " with your right arm",
    " using only the left arm",
    " using only the right arm",
)


def _strip_eval_role_suffix(instr):
    for suffix in _EVAL_ROLE_SUFFIXES:
        if instr.endswith(suffix):
            return instr[:-len(suffix)]
    return instr

def strip_partial_role_language_markers(instr):
    for marker in _PARTIAL_ROLE_LANGUAGE_MARKERS:
        instr = instr.replace(marker, "")
    return instr


class RLBenchDataset(BaseDataset):
    """RLBench dataset."""
    quat_format= 'xyzw'

    def __init__(
        self,
        root,
        instructions,
        copies=None,
        relative_action=False,
        mem_limit=8,
        chunk_size=4,
        tasks=None,
        all_tasks=None,
        img_size=None,
        data_aug_swap=False,
        data_aug_swap_prob=0.5,
    ):
        self.data_aug_swap = data_aug_swap
        self.data_aug_swap_prob = data_aug_swap_prob
        if not 0.0 <= self.data_aug_swap_prob <= 1.0:
            raise ValueError("data_aug_swap_prob must be in [0, 1]")
        self.swap_camera_inds = None
        
        if self.data_aug_swap and hasattr(self, 'cameras'):
            cam_list = list(self.cameras)
            indices = list(range(len(cam_list)))
            
            def swap_if_exist(name1, name2):
                if name1 in cam_list and name2 in cam_list:
                    i1 = cam_list.index(name1)
                    i2 = cam_list.index(name2)
                    indices[i1], indices[i2] = indices[i2], indices[i1]
            
            swap_if_exist('wrist_left', 'wrist_right')
            swap_if_exist('left_shoulder', 'right_shoulder')
            swap_if_exist('over_shoulder_left', 'over_shoulder_right')
            
            if indices != list(range(len(cam_list))):
                self.swap_camera_inds = indices
                print(
                    "Data Augmentation Swap enabled. "
                    f"Probability: {self.data_aug_swap_prob}. "
                    f"Camera swap indices: {self.swap_camera_inds}"
                )
        
        super().__init__(
            root=root,
            instructions=instructions,
            copies=copies,
            relative_action=relative_action,
            mem_limit=mem_limit,
            chunk_size=chunk_size,
            tasks=tasks,
            all_tasks=all_tasks,
            img_size=img_size,
        )

    def _get_task(self, idx):
        if self.all_tasks is None:
            return ["task"] * self.chunk_size
        return [
            self.all_tasks[int(tid)]
            for tid in self.annos['task_id'][idx:idx + self.chunk_size]
        ]

    def _get_instr(self, idx):
        if self.all_tasks is None:
            return ["instruction"] * self.chunk_size
        instrs = []
        for t, v in zip(
            self.annos['task_id'][idx:idx + self.chunk_size],
            self.annos['variation'][idx:idx + self.chunk_size]
        ):
            task_name = self.all_tasks[int(t)]
            instr = random.choice(self._instructions[task_name][str(int(v))])
            instr = _strip_eval_role_suffix(instr)
            if (
                task_name.endswith(("_random_left", "_random_right"))
            ):
                instr = strip_partial_role_language_markers(instr)
            instrs.append(instr)
        return instrs

    def _get_rgb2d(self, idx):
        if self.camera_inds2d is not None:
            rgb2d = self._get_attr_by_idx(idx, 'rgb', False)[:, self.camera_inds2d]
            if self.img_size is not None and (rgb2d.shape[-1] != self.img_size or rgb2d.shape[-2] != self.img_size):
                b, nc, c, h, w = rgb2d.shape
                rgb2d = F.interpolate(
                    rgb2d.flatten(0, 1).float(), (self.img_size, self.img_size),
                    mode='bilinear', antialias=True
                ).to(rgb2d.dtype).reshape(b, nc, c, self.img_size, self.img_size)
            return rgb2d
        return None

    def _get_extrinsics(self, idx):
        return self._get_attr_by_idx(idx, 'extrinsics', True)

    def _get_intrinsics(self, idx):
        intrinsics = self._get_attr_by_idx(idx, 'intrinsics', True)
        if self.img_size is not None:
            # We need the original image size to calculate the scaling factor
            # We can get it from the zarr metadata or from a loaded sample
            # Since we already have the intrinsics, we can assume the images 
            # were loaded via _get_rgb or _get_depth which might have resized them.
            # However, those methods don't store the original size.
            
            # Let's peek at the original shape from the annos
            orig_h, orig_w = self.annos['depth'].shape[-2:]
            if orig_h != self.img_size or orig_w != self.img_size:
                scale_h = self.img_size / orig_h
                scale_w = self.img_size / orig_w
                intrinsics = intrinsics.clone()
                intrinsics[..., 0, 0] *= scale_w
                intrinsics[..., 0, 2] *= scale_w
                intrinsics[..., 1, 1] *= scale_h
                intrinsics[..., 1, 2] *= scale_h
        return intrinsics

    def _apply_swap(self, sample):
        if not self.data_aug_swap:
            return sample
        if random.random() >= self.data_aug_swap_prob:
            return sample
            
        # Swap proprioception and action if bimanual (nhand=2)
        # Proprio: (chunk, nhist, nhand, 8)
        if "proprioception" in sample:
            p = sample["proprioception"]
            if p.ndim >= 3 and p.shape[-2] == 2:
                sample["proprioception"] = p.flip(-2)
                
        # Action: (chunk, T, nhand, 8)
        if "action" in sample:
            a = sample["action"]
            if a.ndim >= 3 and a.shape[-2] == 2:
                sample["action"] = a.flip(-2)
                
        # Swap cameras
        if self.swap_camera_inds is not None:
            inds = self.swap_camera_inds
            for key in ["rgb", "depth", "pcd", "extrinsics", "intrinsics", "rgb2d"]:
                if key in sample and sample[key] is not None:
                    # Assumes dim 1 is n_cam
                    # Check if dim 1 matches length of indices
                    if sample[key].shape[1] == len(inds):
                        sample[key] = sample[key][:, inds]

        return sample

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam, H, W) float16
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
            task_id: (N,) uint8
            variation: (N,) uint8
            extrinsics: (N, n_cam, 4, 4) float
            intrinsics: (N, n_cam, 3, 3) float
        }
        """
        # First detect which copy we fall into
        idx = idx % len(self.indices)
        # and then get the original index from our filtered list
        idx = self.indices[idx]
        ret = {
            "task": self._get_task(idx),  # [str]
            "instr": self._get_instr(idx),  # [str]
            "rgb": self._get_rgb(idx),  # tensor(n_cam3d, 3, H, W)
            "depth": self._get_depth(idx),  # tensor(n_cam3d, H, W)
            "rgb2d": self._get_rgb2d(idx),  # tensor(n_cam2d, 3, H, W)
            "proprioception": self._get_proprioception(idx),  # tensor(1, 8)
            "action": self._get_action(idx),  # tensor(T, 8)
            "extrinsics": self._get_extrinsics(idx),  # tensor(n_cam3d, 4, 4)
            "intrinsics": self._get_intrinsics(idx)  # tensor(n_cam3d, 3, 3)
        }
        return self._apply_swap(ret)



class Peract2Dataset(RLBenchDataset):
    """RLBench dataset under Peract2 setup."""
    tasks = PERACT2_TASKS
    cameras = ("front", "wrist_left", "wrist_right")
    camera_inds = None
    train_copies = 10
    camera_inds2d = None

