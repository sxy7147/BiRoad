import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


@dataclass(frozen=True)
class RoleSamplingStats:
    majority_count: int
    minority_count: int
    majority_weight: float
    minority_weight: float


def build_balanced_role_weights(dataset, minority_keyword="symmetric"):
    """Build per-item weights with equal mass for majority/minority roles."""
    if not hasattr(dataset, "indices"):
        raise ValueError("Balanced role sampling requires dataset.indices")
    if not hasattr(dataset, "annos") or "task_id" not in dataset.annos:
        raise ValueError("Balanced role sampling requires zarr field 'task_id'")
    if not getattr(dataset, "all_tasks", None):
        raise ValueError("Balanced role sampling requires dataset.all_tasks")

    task_ids = np.asarray(dataset.annos["task_id"])
    base_labels = []
    for sample_index in dataset.indices:
        task_id = int(task_ids[int(sample_index)])
        if task_id < 0 or task_id >= len(dataset.all_tasks):
            raise ValueError(
                f"task_id {task_id} is outside all_tasks with "
                f"{len(dataset.all_tasks)} entries"
            )
        task_name = dataset.all_tasks[task_id]
        base_labels.append(minority_keyword in task_name)

    if not base_labels or len(dataset) % len(base_labels) != 0:
        raise ValueError(
            "Balanced role sampling requires dataset length to be an integer "
            "multiple of its filtered indices"
        )

    repeats = len(dataset) // len(base_labels)
    minority_mask = torch.tensor(base_labels * repeats, dtype=torch.bool)
    minority_count = int(minority_mask.sum().item())
    majority_count = len(minority_mask) - minority_count
    if majority_count == 0 or minority_count == 0:
        raise ValueError(
            "Balanced role sampling requires both majority and minority samples; "
            f"found majority={majority_count}, minority={minority_count}"
        )

    majority_weight = 0.5 / majority_count
    minority_weight = 0.5 / minority_count
    weights = torch.full(
        (len(minority_mask),), majority_weight, dtype=torch.double
    )
    weights[minority_mask] = minority_weight
    stats = RoleSamplingStats(
        majority_count=majority_count,
        minority_count=minority_count,
        majority_weight=majority_weight,
        minority_weight=minority_weight,
    )
    return weights, stats


class DistributedWeightedSampler(Sampler):
    """Draw one global weighted sample stream, then shard it across ranks."""

    def __init__(
        self,
        weights,
        num_replicas=None,
        rank=None,
        seed=0,
        drop_last=False,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        if rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"rank must be in [0, {num_replicas - 1}], got {rank}"
            )

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if self.weights.ndim != 1 or len(self.weights) == 0:
            raise ValueError("weights must be a non-empty 1D sequence")
        if not torch.isfinite(self.weights).all() or (self.weights < 0).any():
            raise ValueError("weights must be finite and non-negative")
        if float(self.weights.sum()) <= 0:
            raise ValueError("weights must have a positive sum")

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        if drop_last and len(self.weights) % num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.weights) - num_replicas) / num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.weights) / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def build_train_sampler(
    dataset,
    balanced_role_sampling=False,
    num_replicas=None,
    rank=None,
):
    if not balanced_role_sampling:
        return DistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            drop_last=True,
        )

    weights, stats = build_balanced_role_weights(dataset)
    sampler = DistributedWeightedSampler(
        weights,
        num_replicas=num_replicas,
        rank=rank,
        drop_last=True,
    )
    sampler.role_sampling_stats = stats
    return sampler
