import json
import torch.nn.functional as F

from torch.utils.data import Dataset

from .utils import to_tensor, read_zarr_with_cache, to_relative_action


class BaseDataset(Dataset):
    """Base dataset."""

    def __init__(
        self,
        root,  # the directory path of the dataset
        instructions,  # path to instruction file
        copies=None,  # copy the dataset for less loader restarts
        relative_action=False,  # whether to return relative actions
        mem_limit=8,  # cache limit per dataset class in GigaBytes
        chunk_size=4,  # chunk size for zarr
        tasks=None,  # tasks list
        all_tasks=None,  # all tasks list (for ID mapping)
        img_size=None,  # custom image size
    ):
        super().__init__()
        self.copies = self.train_copies if copies is None else copies
        self._relative_action = relative_action
        self.chunk_size = chunk_size
        self.img_size = img_size

        if all_tasks is not None:
            self.all_tasks = all_tasks
        else:
            self.all_tasks = getattr(self, 'tasks', None)

        if tasks is not None:
            self.tasks = tasks
        else:
            self.tasks = self.all_tasks

        # Load instructions
        self._instructions = self._load_instructions(instructions)

        # Load all annotations lazily
        self.annos = read_zarr_with_cache(root, mem_gb=mem_limit)
        # Sanity check
        len_ = len(self.annos['action'])
        for key in self.annos:
            assert len(self.annos[key]) == len_, f'length mismatch in {key}'
        print(f"Found {len(self.annos['action'])} samples")

        # Filter indices if tasks are specified
        self.indices = self._filter_indices()

    def _filter_indices(self):
        if self.tasks is not None and len(self.tasks) == 0:
            raise ValueError(
                "tasks must not be empty. Use tasks=None to select all tasks."
            )
        # Default: use all samples
        all_indices = list(range(0, len(self.annos['action']), self.chunk_size))
        
        if self.tasks is None or self.all_tasks is None:
            return all_indices
        
        print('self.tasks: ', self.tasks)
        print('self.all_tasks: ', self.all_tasks)
            
        # If tasks is the same as all_tasks, no filtering needed
        if set(self.tasks) == set(self.all_tasks):
            return all_indices
            
        # Map task names to IDs
        task_to_id = {task: i for i, task in enumerate(self.all_tasks)}
        print('task_to_id: ', task_to_id)
        target_ids = [task_to_id[t] for t in self.tasks if t in task_to_id]
        print('target_ids: ', target_ids)
        
        if not target_ids:
            raise ValueError(
                f"None of the requested tasks match all_tasks: {self.tasks}. "
                "Check task names and the all_tasks mapping."
            )
            
        # Filter indices where the task_id is in target_ids
        import numpy as np
        task_ids = np.array(self.annos['task_id'])
        # task_id is stored per sample, but we only need to check the first sample of each chunk
        filtered_indices = [
            i for i in all_indices 
            if task_ids[i] in target_ids
        ]
        
        print(f"Filtered dataset from {len(all_indices)} to {len(filtered_indices)} chunks for tasks: {self.tasks}")
        return filtered_indices

    def _load_instructions(self, instruction_file):
        return json.load(open(instruction_file))

    def _get_attr_by_idx(self, idx, attr, filter_cam=False):
        t = to_tensor(self.annos[attr][idx:idx + self.chunk_size])
        if filter_cam and self.camera_inds is not None:
            t = t[:, self.camera_inds]
        return t

    def _get_task(self, idx):
        return ["task"] * self.chunk_size

    def _get_instr(self, idx):
        return ["instruction"] * self.chunk_size

    def _get_rgb(self, idx, key='rgb'):
        rgb = self._get_attr_by_idx(idx, key, True)
        if self.img_size is not None and (rgb.shape[-1] != self.img_size or rgb.shape[-2] != self.img_size):
            b, nc, c, h, w = rgb.shape
            rgb = F.interpolate(
                rgb.flatten(0, 1).float(), (self.img_size, self.img_size),
                mode='bilinear', antialias=True
            ).to(rgb.dtype).reshape(b, nc, c, self.img_size, self.img_size)
        return rgb

    def _get_depth(self, idx, key='depth'):
        depth = self._get_attr_by_idx(idx, key, True)
        if self.img_size is not None and (depth.shape[-1] != self.img_size or depth.shape[-2] != self.img_size):
            b, nc, h, w = depth.shape
            depth = F.interpolate(
                depth.flatten(0, 1).unsqueeze(1).float(), (self.img_size, self.img_size),
                mode='bilinear', antialias=True
            ).to(depth.dtype).reshape(b, nc, self.img_size, self.img_size)
        return depth

    def _get_proprioception(self, idx):
        return self._get_attr_by_idx(idx, 'proprioception', False)

    def _get_action(self, idx):
        if self._relative_action:
            if 'rel_action' in self.annos:
                return self._get_attr_by_idx(idx, 'rel_action', False)
            else:
                action = self._get_attr_by_idx(idx, 'action', False)
                prop = self._get_proprioception(idx)[[-1]]
                action = to_relative_action(action, prop, self.quat_format)
        else:
            action = self._get_attr_by_idx(idx, 'action', False)
        return action

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam, H, W) float16
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
        }
        In addition self.annos may contain fields for task/instruction ids
        """
        # First detect which copy we fall into
        idx = idx % len(self.indices)
        # and then get the original index from our filtered list
        idx = self.indices[idx]
        
        return {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),  # [str]
            "rgb": self._get_rgb(idx),  # tensor(n_cam, 3, H, W)
            "depth": self._get_depth(idx),  # tensor(n_cam, H, W)
            "proprioception": self._get_proprioception(idx),  # tensor(1, 8)
            "action": self._get_action(idx)  # tensor(T, 8)
        }

    def __len__(self):
        return self.copies * len(self.indices)
