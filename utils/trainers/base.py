from copy import deepcopy
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from modeling.encoder.text import fetch_tokenizers
from ..common_utils import count_parameters
from ..depth2cloud import fetch_depth2cloud
from ..data_preprocessors import fetch_data_preprocessor
from ..ema import EMA
from ..schedulers import fetch_scheduler
from ..samplers import build_train_sampler
from .utils import compute_metrics


class SingleProcessModule(nn.Module):
    """DDP-compatible wrapper for single-process training."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class BaseTrainTester:
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args, dataset_cls, model_cls):
        """Initialize."""
        self.args = args
        self.dataset_cls = dataset_cls
        self.model_cls = model_cls

        self.preprocessor = fetch_data_preprocessor(self.args.dataset, self.args.custom_img_size)(
            self.args.keypose_only,
            self.args.num_history,
            custom_imsize=self.args.custom_img_size,
            depth2cloud=fetch_depth2cloud(self.args.dataset, self.args.custom_img_size),
            workspace_bounds=getattr(self.args, "pcd_workspace_bounds", None),
            image_filter_radius=getattr(self.args, "pcd_image_filter_radius", 0.0),
            image_filter_min_neighbors=getattr(self.args, "pcd_image_filter_min_neighbors", 0),
            local_density_filter_radius=getattr(self.args, "pcd_local_density_filter_radius", 0.0),
            local_density_filter_min_neighbors=getattr(
                self.args,
                "pcd_local_density_filter_min_neighbors",
                0,
            ),
            image_component_filter_min_pixels=getattr(
                self.args,
                "pcd_image_component_filter_min_pixels",
                0,
            ),
            image_component_filter_iterations=getattr(
                self.args,
                "pcd_image_component_filter_iterations",
                32,
            ),
        )

        if dist.get_rank() == 0 and not self.args.eval_only:
            self.writer = SummaryWriter(log_dir=args.tensorboard_log_dir, flush_secs=10)
            self.writer.add_text("run/log_dir", str(args.log_dir), 0)
            self.writer.flush()

    def get_datasets(self):
        """Initialize datasets."""
        # Initialize datasets with arguments
        train_dataset = self.dataset_cls(
            root=self.args.train_data_dir,
            instructions=self.args.train_instructions,
            relative_action=self.args.relative_action,
            mem_limit=self.args.memory_limit,
            chunk_size=self.args.chunk_size,
            tasks=self.args.tasks,
            all_tasks=getattr(self.args, 'all_tasks', None),
            img_size=self.args.custom_img_size,
            data_aug_swap=self.args.data_aug_swap,
        )
        val_dataset = self.dataset_cls(
            root=self.args.eval_data_dir,
            instructions=self.args.val_instructions,
            copies=1,
            relative_action=self.args.relative_action,
            mem_limit=0.1,
            chunk_size=self.args.chunk_size,
            tasks=self.args.tasks,
            all_tasks=getattr(self.args, 'all_tasks', None),
            img_size=self.args.custom_img_size,
            data_aug_swap=False,
        )
        return train_dataset, val_dataset

    def get_loaders(self):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Datasets
        train_dataset, val_dataset = self.get_datasets()
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)
        balanced_role_sampling = getattr(
            self.args, "balanced_role_sampling", False
        )
        train_sampler = build_train_sampler(
            train_dataset,
            balanced_role_sampling=balanced_role_sampling,
        )
        if balanced_role_sampling and dist.get_rank() == 0:
            stats = train_sampler.role_sampling_stats
            print(
                "Balanced role sampling enabled: "
                f"majority={stats.majority_count}, "
                f"minority={stats.minority_count}, "
                f"majority_weight={stats.majority_weight:.8g}, "
                f"minority_weight={stats.minority_weight:.8g}, "
                "expected_distribution=50/50, "
                f"samples_per_rank={len(train_sampler)}"
            )
        worker_kwargs = {
            "num_workers": self.args.num_workers,
            "prefetch_factor": 4 if self.args.num_workers > 0 else None,
            "persistent_workers": self.args.num_workers > 0
        }
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size // self.args.chunk_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            collate_fn=base_collate_fn,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            generator=g,
            **worker_kwargs
        )
        # No sampler for val!
        if dist.get_rank() == 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size_val // self.args.chunk_size,
                shuffle=False,
                collate_fn=base_collate_fn,
                pin_memory=True,
                sampler=None,
                drop_last=False,
                **worker_kwargs
            )
        else:
            val_loader = None
        return train_loader, val_loader, train_sampler

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        _model = self.model_cls(
            backbone=self.args.backbone,
            finetune_backbone=self.args.finetune_backbone,
            finetune_text_encoder=self.args.finetune_text_encoder,
            num_vis_instr_attn_layers=self.args.num_vis_instr_attn_layers,
            fps_subsampling_factor=self.args.fps_subsampling_factor,
            embedding_dim=self.args.embedding_dim,
            num_attn_heads=self.args.num_attn_heads,
            nhist=self.args.num_history,
            nhand=2 if self.args.bimanual else 1,
            num_shared_attn_layers=self.args.num_shared_attn_layers,
            relative=self.args.relative_action,
            rotation_format=self.args.rotation_format,
            denoise_timesteps=self.args.denoise_timesteps,
            denoise_model=self.args.denoise_model,
            lv2_batch_size=self.args.lv2_batch_size,
            use_sym_anti_decomp=getattr(self.args, 'use_sym_anti_decomp', False),
            sym_anti_direct_update=getattr(self.args, 'sym_anti_direct_update', False),
            sym_anti_decomp_placement=getattr(self.args, 'sym_anti_decomp_placement', "post_self_attn"),
            sym_anti_module_type=getattr(self.args, 'sym_anti_module_type', "sym_anti"),
        )

        # Print basic modules' parameters
        if dist.get_rank() == 0:
            count_parameters(_model)

        # Useful for some models to ensure parameters are contiguous
        for name, param in _model.named_parameters():
            if param.requires_grad and param.ndim > 1 and not param.is_contiguous():
                print(f"Fixing layout for: {name}")
                param.data = param.contiguous()

        return _model

    @torch.no_grad()
    def get_workspace_normalizer(self, dataset, ndims=3):
        print("Computing workspace normalizer...")

        # Loop and compute action min-max
        min_, max_ = torch.ones(ndims) * 10000, -torch.ones(ndims) * 10000
        for idx in tqdm(dataset.indices):
            action = dataset._get_action(int(idx))[..., :ndims].reshape([-1, ndims])
            min_ = torch.min(min_, action.min(0).values)
            max_ = torch.max(max_, action.max(0).values)

        min_ = min_ - self.args.workspace_normalizer_buffer
        max_ = max_ + self.args.workspace_normalizer_buffer

        return nn.Parameter(torch.stack([min_, max_]), requires_grad=False)

    def get_optimizer(self, model):
        """Initialize optimizer."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": self.args.lr},
            {"params": [], "weight_decay": self.args.wd, "lr": self.args.lr}
        ]
        if self.args.finetune_backbone:
            optimizer_grouped_parameters.append({
                "params": [], "weight_decay": self.args.wd,
                "lr": self.args.backbone_lr
            })

        # Collect names of all norm parameters
        norm_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LocalResponseNorm,
            torch.nn.RMSNorm
        )
        norm_param_names = set()
        for module_name, module in model.named_modules():
            if isinstance(module, norm_types):
                for param_name, _ in module.named_parameters(recurse=False):
                    norm_param_names.add(f"{module_name}.{param_name}")

        # Now split parameters based on name
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in norm_param_names or name.endswith(".bias"):
                optimizer_grouped_parameters[0]["params"].append(param)
            elif self.args.finetune_backbone and 'backbone' in name:
                optimizer_grouped_parameters[2]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.95)
        )
        return optimizer

    def main(self):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, val_loader, train_sampler = self.get_loaders()

        # Get model
        model = self.get_model()
        self.tokenizer = fetch_tokenizers(self.args.backbone)
        if not os.path.exists(self.args.checkpoint):
            normalizer_ndims = model.workspace_normalizer.size(-1)
            normalizer = self.get_workspace_normalizer(
                train_loader.dataset,
                ndims=normalizer_ndims,
            )
            with torch.no_grad():
                model.workspace_normalizer.copy_(normalizer)
            if dist.get_world_size() > 1:
                dist.barrier(device_ids=[torch.cuda.current_device()])

        # Get optimizer
        optimizer = self.get_optimizer(model)
        lr_scheduler = fetch_scheduler(
            self.args.lr_scheduler, optimizer, self.args.train_iters
        )
        scaler = torch.GradScaler()

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda()
        # make sure to compile before DDP!
        if self.args.use_compile:
            model.compute_loss = torch.compile(model.compute_loss, fullgraph=True)
        if dist.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[self.args.local_rank],
                broadcast_buffers=False, find_unused_parameters=True
            )
        else:
            model = SingleProcessModule(model)

        # Initialize EMA copy
        ema_model = deepcopy(model)
        self.ema = EMA()

        # Check for a checkpoint
        start_iter, best_loss, best_checkpoints = 0, None, []
        if self.args.checkpoint:
            start_iter, best_loss, best_checkpoints = self.load_checkpoint(
                model, ema_model, optimizer
            )
        print(model.module.workspace_normalizer)

        # Eval only
        if self.args.eval_only:
            if dist.get_rank() == 0:
                print("Test evaluation.......")
                model.eval()
                self.evaluate_nsteps(
                    ema_model if self.args.use_ema else model,
                    val_loader, step_id=-1,
                    val_iters=-1
                )
            if dist.get_world_size() > 1:
                dist.barrier(device_ids=[torch.cuda.current_device()])
            return ema_model if self.args.use_ema else model

        # Step the lr scheduler to the current step
        for _ in range(start_iter):
            lr_scheduler.step()

        # Step the sampler to the currect "epoch"
        samples_per_epoch = len(train_loader)
        epoch = start_iter // samples_per_epoch + 1
        train_sampler.set_epoch(epoch)  # ensures new batches are sampled

        # Training loop
        model.train()
        iter_loader = iter(train_loader)
        for step_id in trange(start_iter, self.args.train_iters):
            try:
                sample = next(iter_loader)
            except StopIteration:
                # when the iterator is exhausted, we need to reset it
                # and increment the epoch
                epoch += 1
                train_sampler.set_epoch(epoch)
                iter_loader = iter(train_loader)
                sample = next(iter_loader)

            train_loss = self.train_one_step(
                model, optimizer, scaler, lr_scheduler, sample, step_id
            )
            if dist.get_rank() == 0 and (step_id == start_iter or (step_id + 1) % 100 == 0):
                self.writer.add_scalar("train/loss", train_loss, step_id)
                self.writer.flush()
            self.ema.step(model, ema_model, self.args.use_ema, step_id)

            if (step_id + 1) % self.args.val_freq == 0 and dist.get_rank() == 0:
                print("Train evaluation.......")
                model.eval()
                self.evaluate_nsteps(
                    ema_model if self.args.use_ema else model,
                    train_loader, step_id,
                    val_iters=10,
                    split='train'
                )
                print("Test evaluation.......")
                new_loss = self.evaluate_nsteps(
                    ema_model if self.args.use_ema else model,
                    val_loader, step_id,
                    val_iters=1250
                )
                # save model
                best_loss = self.save_checkpoint(
                    model, ema_model, optimizer, step_id,
                    new_loss, best_loss, best_checkpoints
                )
                model.train()
            if dist.get_world_size() > 1:
                dist.barrier(device_ids=[torch.cuda.current_device()])

        return ema_model if self.args.use_ema else model

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        pass  # implement in children

    def _model_forward(self, model, sample, training=True):
        batch = self.prepare_batch(
            sample, augment=training
        )
        action, action_mask, rgbs, rgb2d, pcds, instr, prop = batch[:7]
        
        if self.args.pre_tokenize:
            instr = self.tokenizer(instr).cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                action, action_mask, rgbs, rgb2d, pcds, instr, prop,
                run_inference=not training,
            )
        return out  # loss if training, else action

    def train_one_step(self, model, optimizer, scaler, lr_scheduler, sample, step_id):
        """Run a single training step."""
        optimizer.zero_grad()

        # Forward pass
        loss = self._model_forward(model, sample)

        # Backward pass
        scaler.scale(loss).backward()

        # Clip gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        # Update
        scaler.step(optimizer)
        scaler.update()

        # Step the lr scheduler
        lr_scheduler.step()

        return loss.detach().item()

    @torch.inference_mode()
    def evaluate_nsteps(self, model, loader, step_id, val_iters, split='val'):
        """Run a given number of evaluation steps."""
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in tqdm(enumerate(loader)):
            if i == val_iters:
                break

            pred_action = self._model_forward(model, sample, training=False)
            gt_action = sample["action"].cuda(non_blocking=True)
            if self.args.relative_action:
                pred_action = relative_to_absolute(
                    pred_action[:, :, 0],
                    sample["proprioception"].cuda(non_blocking=True)[:, :, 0]
                )
                gt_action = relative_to_absolute(
                    gt_action[:, :, 0],
                    sample["proprioception"].cuda(non_blocking=True)[:, :, 0]
                )

            losses, losses_B = compute_metrics(pred_action, gt_action)

            # Gather global statistics
            for n, l in losses.items():
                key = f"{split}-losses/mean/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            # Gather per-task statistics
            tasks = np.array(sample["task"])
            for n, l in losses_B.items():
                for task in np.unique(tasks):
                    key = f"{split}-loss/{task}/{n}"
                    l_task = l[tasks == task].mean()
                    if key not in values:
                        values[key] = torch.Tensor([]).to(device)
                    values[key] = torch.cat([values[key], l_task.unsqueeze(0)])

        # Log all statistics
        values = {k: v.mean().item() for k, v in values.items()}
        if dist.get_rank() == 0:
            if step_id > -1:
                for key, val in values.items():
                    self.writer.add_scalar(key, val, step_id)

            # Also log to terminal
            print(f"Step {step_id}:")
            for key, value in values.items():
                print(f"{key}: {value:.03f}")

        return -values[f'{split}-losses/mean/traj_pos_acc_001']

    def load_checkpoint(self, model, ema_model, optimizer):
        """Load from checkpoint."""
        print("=> trying checkpoint '{}'".format(self.args.checkpoint))
        if not os.path.exists(self.args.checkpoint):
            print('Warning: checkpoint was not found, starting from scratch')
            print('The main process will compute workspace bounds')
            return 0, None, []

        model_dict = torch.load(
            self.args.checkpoint,
            map_location="cpu",
            weights_only=True
        )
        # Load weights flexibly
        msn, unxpct = model.load_state_dict(model_dict["weight"], strict=False)
        if msn:
            print(f"Missing keys (not found in checkpoint): {len(msn)}")
            print(msn)
        if unxpct:
            print(f"Unexpected keys (ignored): {len(unxpct)}")
            print(unxpct)
        if not msn and not unxpct:
            print("All keys matched successfully!")
        # EMA weights
        if model_dict.get("ema_weight") is not None:
            ema_model.load_state_dict(model_dict["ema_weight"], strict=True)
        # Useful for resuming training
        if 'optimizer' in model_dict and not self.args.eval_only:
            optimizer.load_state_dict(model_dict["optimizer"])
        start_iter = model_dict.get("iter", 0)
        best_loss = model_dict.get("best_loss", None)
        best_checkpoints = model_dict.get("best_checkpoints", [])
        if not best_checkpoints and best_loss is not None:
            best_path = self.args.log_dir / "best.pth"
            if best_path.exists():
                best_checkpoints = [{
                    "loss": float(best_loss),
                    "iter": int(start_iter),
                    "path": "best.pth"
                }]

        print("=> loaded successfully '{}' (step {})".format(
            self.args.checkpoint, model_dict.get("iter", 0)
        ))
        del model_dict
        torch.cuda.empty_cache()
        return start_iter, best_loss, best_checkpoints

    def save_checkpoint(self, model, ema_model, optimizer,
                        step_id, new_loss, best_loss, best_checkpoints):
        """Save checkpoint if requested."""
        model_state = model.state_dict()
        ema_state = ema_model.state_dict() if self.args.use_ema else None
        num_best_checkpoints = max(
            1, int(getattr(self.args, "num_best_checkpoints", 1))
        )
        next_iter = step_id + 1

        def make_payload(include_optimizer=False):
            payload = {
                "weight": model_state,
                "ema_weight": ema_state,
                "iter": next_iter,
                "best_loss": best_loss,
                "best_checkpoints": best_checkpoints
            }
            if include_optimizer:
                payload["optimizer"] = optimizer.state_dict()
            return payload

        # Best checkpoint
        candidate = {
            "loss": float(new_loss),
            "iter": next_iter,
            "path": None,
            "is_current": True
        }
        worst_best = (
            max(ckpt["loss"] for ckpt in best_checkpoints)
            if best_checkpoints else None
        )
        should_save_best = (
            len(best_checkpoints) < num_best_checkpoints
            or candidate["loss"] <= worst_best
        )
        if should_save_best:
            existing = [
                ckpt for ckpt in best_checkpoints
                if (self.args.log_dir / ckpt["path"]).exists()
            ]
            existing.append(candidate)
            top_checkpoints = sorted(
                existing, key=lambda ckpt: (ckpt["loss"], ckpt["iter"])
            )[:num_best_checkpoints]

            best_loss = top_checkpoints[0]["loss"]
            new_best_checkpoints = [
                {
                    "loss": ckpt["loss"],
                    "iter": ckpt["iter"],
                    "path": f"best_top{rank}.pth"
                }
                for rank, ckpt in enumerate(top_checkpoints, start=1)
            ]
            best_checkpoints[:] = new_best_checkpoints

            tmp_paths = []
            for rank, ckpt in enumerate(top_checkpoints, start=1):
                tmp_path = self.args.log_dir / f".best_top{rank}.tmp.pth"
                if ckpt.get("is_current", False):
                    torch.save(make_payload(), tmp_path)
                else:
                    payload = torch.load(
                        self.args.log_dir / ckpt["path"],
                        map_location="cpu",
                        weights_only=True
                    )
                    payload["best_loss"] = best_loss
                    payload["best_checkpoints"] = new_best_checkpoints
                    torch.save(payload, tmp_path)
                tmp_paths.append(tmp_path)

            for stale_path in self.args.log_dir.glob("best_top*.pth"):
                stale_path.unlink()

            for rank, tmp_path in enumerate(tmp_paths, start=1):
                dst_path = self.args.log_dir / f"best_top{rank}.pth"
                tmp_path.replace(dst_path)

            best_path = self.args.log_dir / "best.pth"
            shutil.copy2(self.args.log_dir / best_checkpoints[0]["path"], best_path)
            print(f"Best checkpoint saved to: {best_path}")
            print(f"Best top-{len(best_checkpoints)} checkpoints: {best_checkpoints}")

        # Last checkpoint (always saved)
        ckpt_path = self.args.log_dir / "last.pth"
        torch.save(make_payload(include_optimizer=True), ckpt_path)
        print(f"Last checkpoint saved to: {ckpt_path}")

        # Save intermediate checkpoints
        if (step_id + 1) % self.args.interm_ckpt_freq == 0:
            ckpt_path = self.args.log_dir / f"interm_iter_{step_id + 1}.pth"
            torch.save(make_payload(), ckpt_path)
            print(f"Intermediate checkpoint saved to: {ckpt_path}")

        return best_loss


def base_collate_fn(batch):
    """Custom collate_fn, measured to be faster than default."""
    _dict = {}

    # Values for these come as lists
    list_keys = ["task", "instr"]
    for key in list_keys:
        if key not in batch[0].keys():
            continue
        _dict[key] = []
        for item in batch:
            _dict[key].extend(item[key])

    # Treat rest as tensors
    _dict.update({
        k_: (
            torch.cat([item[k_] for item in batch])
            if batch[0][k_] is not None else None
        )
        for k_ in batch[0].keys() if k_ not in list_keys
    })

    return _dict


def relative_to_absolute(action, proprio):
    # action (B, T, 8), proprio (B, 1, 7)
    pos = proprio[..., :3] + action[..., :3].cumsum(1)

    orn = proprio[..., 3:6] + action[..., 3:6].cumsum(1)
    orn = (orn + torch.pi) % (2 * torch.pi) - torch.pi

    return torch.cat([pos, orn, action[..., 6:]], -1)
