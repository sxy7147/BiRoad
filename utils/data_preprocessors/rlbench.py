from kornia import augmentation as K
import torch
from torch.nn import functional as F

from .base import DataPreprocessor
from .workspace import (
    pcd_image_component_mask,
    pcd_image_noise_mask,
    pcd_local_density_mask,
    pcd_workspace_mask,
)


class RLBenchDataPreprocessor(DataPreprocessor):

    def __init__(self, keypose_only=False, num_history=1,
                 orig_imsize=256, custom_imsize=None, depth2cloud=None,
                 workspace_bounds=None, image_filter_radius=0.0,
                 image_filter_min_neighbors=0, local_density_filter_radius=0.0,
                 local_density_filter_min_neighbors=0, image_component_filter_min_pixels=0,
                 image_component_filter_iterations=32):
        super().__init__(
            keypose_only=keypose_only,
            num_history=num_history,
            custom_imsize=custom_imsize,
            depth2cloud=depth2cloud
        )
        self.workspace_bounds = workspace_bounds
        self.image_filter_radius = image_filter_radius
        self.image_filter_min_neighbors = image_filter_min_neighbors
        self.local_density_filter_radius = local_density_filter_radius
        self.local_density_filter_min_neighbors = local_density_filter_min_neighbors
        self.image_component_filter_min_pixels = image_component_filter_min_pixels
        self.image_component_filter_iterations = image_component_filter_iterations
        self.aug = K.AugmentationSequential(
            K.RandomAffine(
                degrees=0,
                translate=0.0,
                scale=(0.75, 1.25),
                padding_mode="reflection",
                p=0.8
            ),
            K.RandomResizedCrop(
                size=(orig_imsize, orig_imsize),
                scale=(0.95, 1.05),
                p=0.1
            )
        ).cuda()

    def process_obs(self, rgbs, rgb2d, depth, extrinsics, intrinsics,
                    augment=False):
        """
        RGBs of shape (B, ncam, 3, h_i, w_i),
        depths of shape (B, ncam, h_i, w_i).
        Assume the 3d cameras go before 2d cameras.
        """
        # Get point cloud from depth
        pcds = self.depth2cloud(
            depth.cuda(non_blocking=True).to(torch.bfloat16),
            extrinsics.cuda(non_blocking=True).to(torch.bfloat16),
            intrinsics.cuda(non_blocking=True).to(torch.bfloat16)
        )

        # Handle non-wrist cameras, which may require augmentations
        if augment:
            b, nc, _, h, w = rgbs.shape
            # Augment in half precision
            obs = torch.cat((
                rgbs.cuda(non_blocking=True).half() / 255,
                pcds[:, :rgbs.size(1)].half()
            ), 2)  # (B, ncam, 6, H, W)
            obs = obs.reshape(-1, 6, h, w)
            obs = self.aug(obs)
            # Convert to full precision
            rgb_3d = obs[:, :3].reshape(b, nc, 3, h, w).float()
            pcd_3d = obs[:, 3:].reshape(b, nc, 3, h, w).float()
        else:
            # Simply convert to full precision
            rgb_3d = rgbs.cuda(non_blocking=True).float() / 255
            pcd_3d = pcds[:, :rgb_3d.size(1)].float()
        workspace_mask = pcd_workspace_mask(pcd_3d, self.workspace_bounds)
        if workspace_mask is not None:
            pcd_3d = pcd_3d * workspace_mask[:, :, None].to(dtype=pcd_3d.dtype)
            rgb_3d = rgb_3d * workspace_mask[:, :, None].to(dtype=rgb_3d.dtype)
        if self.image_filter_radius > 0.0 and self.image_filter_min_neighbors > 0:
            image_filter_mask = pcd_image_noise_mask(
                pcd_3d,
                radius=self.image_filter_radius,
                min_neighbors=self.image_filter_min_neighbors,
            )
            pcd_3d = pcd_3d * image_filter_mask[:, :, None].to(dtype=pcd_3d.dtype)
            rgb_3d = rgb_3d * image_filter_mask[:, :, None].to(dtype=rgb_3d.dtype)
        if self.local_density_filter_radius > 0.0 and self.local_density_filter_min_neighbors > 0:
            density_filter_mask = pcd_local_density_mask(
                pcd_3d,
                radius=self.local_density_filter_radius,
                min_neighbors=self.local_density_filter_min_neighbors,
            )
            pcd_3d = pcd_3d * density_filter_mask[:, :, None].to(dtype=pcd_3d.dtype)
            rgb_3d = rgb_3d * density_filter_mask[:, :, None].to(dtype=rgb_3d.dtype)
        if self.image_component_filter_min_pixels > 1:
            component_filter_mask = pcd_image_component_mask(
                pcd_3d,
                min_pixels=self.image_component_filter_min_pixels,
                num_iterations=self.image_component_filter_iterations,
            )
            pcd_3d = pcd_3d * component_filter_mask[:, :, None].to(dtype=pcd_3d.dtype)
            rgb_3d = rgb_3d * component_filter_mask[:, :, None].to(dtype=rgb_3d.dtype)
        if self.custom_imsize is not None and self.custom_imsize != rgb_3d.size(-1):
            b, nc, _, _, _ = rgb_3d.shape
            rgb_3d = F.interpolate(
                rgb_3d.flatten(0, 1), (self.custom_imsize, self.custom_imsize),
                mode='bilinear', antialias=True
            ).reshape(b, nc, -1, self.custom_imsize, self.custom_imsize)

        # Handle wrist cameras, no augmentations
        rgb_2d = None
        if rgb2d is not None:
            rgb_2d = rgb2d.cuda(non_blocking=True).float() / 255
            if self.custom_imsize is not None and self.custom_imsize != rgb_2d.size(-1):
                b, nc, _, _, _ = rgb_2d.shape
                rgb_2d = F.interpolate(
                    rgb_2d.flatten(0, 1), (self.custom_imsize, self.custom_imsize),
                    mode='bilinear', antialias=True
                ).reshape(b, nc, -1, self.custom_imsize, self.custom_imsize)

        # Concatenate
        if rgb_2d is not None:
            rgbs = torch.cat((rgb_3d, rgb_2d), 1)
        else:
            rgbs = rgb_3d
        if pcd_3d.size(1) < pcds.size(1):
            pcds = torch.cat((pcd_3d, pcds[:, :pcd_3d.size(1)].float()))
        else:
            pcds = pcd_3d
        return rgbs, pcds
