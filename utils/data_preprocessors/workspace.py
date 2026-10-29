import torch
from torch.nn import functional as F


def pcd_workspace_mask(pcds, workspace_bounds):
    if workspace_bounds is None:
        return None
    if len(workspace_bounds) != 6:
        raise ValueError(
            "workspace_bounds must be (xmin,xmax,ymin,ymax,zmin,zmax), "
            f"got {workspace_bounds}"
        )
    xmin, xmax, ymin, ymax, zmin, zmax = workspace_bounds
    x = pcds[:, :, 0]
    y = pcds[:, :, 1]
    z = pcds[:, :, 2]
    mask = (
        (x >= xmin) & (x <= xmax)
        & (y >= ymin) & (y <= ymax)
        & (z >= zmin) & (z <= zmax)
    )
    return mask


def pcd_valid_mask(pcds):
    if pcds.ndim != 5 or pcds.size(2) != 3:
        raise ValueError(f"pcds must have shape (B, N, 3, H, W), got {tuple(pcds.shape)}")
    return torch.isfinite(pcds).all(dim=2) & (torch.linalg.norm(pcds, dim=2) > 1e-8)


def pcd_image_noise_mask(pcds, radius=0.0, min_neighbors=0):
    """Keep points supported by nearby pixels in image space and 3D space.

    This is intentionally conservative: with min_neighbors=1 it removes isolated
    flying pixels while preserving one-pixel-wide continuous gripper/object edges.
    """
    valid = pcd_valid_mask(pcds)
    if radius is None or radius <= 0.0 or min_neighbors is None or min_neighbors <= 0:
        return valid

    _, _, _, h, w = pcds.shape
    padded_pcds = pcds.new_zeros((*pcds.shape[:3], h + 2, w + 2))
    padded_valid = torch.zeros(
        (*valid.shape[:2], h + 2, w + 2),
        dtype=torch.bool,
        device=valid.device,
    )
    padded_pcds[:, :, :, 1:-1, 1:-1] = pcds
    padded_valid[:, :, 1:-1, 1:-1] = valid

    neighbor_count = torch.zeros_like(valid, dtype=torch.int16)
    offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )
    for dy, dx in offsets:
        y0 = 1 + dy
        x0 = 1 + dx
        neighbor_pcds = padded_pcds[:, :, :, y0:y0 + h, x0:x0 + w]
        neighbor_valid = padded_valid[:, :, y0:y0 + h, x0:x0 + w]
        neighbor_dist = torch.linalg.norm(neighbor_pcds - pcds, dim=2)
        close_neighbor = valid & neighbor_valid & (neighbor_dist <= radius)
        neighbor_count += close_neighbor.to(dtype=neighbor_count.dtype)

    return valid & (neighbor_count >= int(min_neighbors))


def pcd_local_density_mask(pcds, radius=0.0, min_neighbors=0, image_window_radius=2):
    """Conservative image-grid approximation of statistical outlier filtering.

    It counts close 3D neighbors inside a small image-space window. This keeps
    connected one-pixel-wide structures while removing tiny isolated clusters.
    """
    valid = pcd_valid_mask(pcds)
    if radius is None or radius <= 0.0 or min_neighbors is None or min_neighbors <= 0:
        return valid
    if image_window_radius < 1:
        raise ValueError(f"image_window_radius must be >= 1, got {image_window_radius}")

    _, _, _, h, w = pcds.shape
    pad = int(image_window_radius)
    padded_pcds = pcds.new_zeros((*pcds.shape[:3], h + 2 * pad, w + 2 * pad))
    padded_valid = torch.zeros(
        (*valid.shape[:2], h + 2 * pad, w + 2 * pad),
        dtype=torch.bool,
        device=valid.device,
    )
    padded_pcds[:, :, :, pad:pad + h, pad:pad + w] = pcds
    padded_valid[:, :, pad:pad + h, pad:pad + w] = valid

    neighbor_count = torch.zeros_like(valid, dtype=torch.int16)
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            if dy == 0 and dx == 0:
                continue
            y0 = pad + dy
            x0 = pad + dx
            neighbor_pcds = padded_pcds[:, :, :, y0:y0 + h, x0:x0 + w]
            neighbor_valid = padded_valid[:, :, y0:y0 + h, x0:x0 + w]
            neighbor_dist = torch.linalg.norm(neighbor_pcds - pcds, dim=2)
            close_neighbor = valid & neighbor_valid & (neighbor_dist <= radius)
            neighbor_count += close_neighbor.to(dtype=neighbor_count.dtype)

    return valid & (neighbor_count >= int(min_neighbors))


def pcd_image_component_mask(pcds, min_pixels=0, num_iterations=32):
    """Remove small image-space connected components from valid point masks.

    The label propagation is intentionally bounded. It targets small floating
    blobs while keeping long thin structures if their local propagated component
    has enough pixels.
    """
    valid = pcd_valid_mask(pcds)
    if min_pixels is None or min_pixels <= 1:
        return valid
    if num_iterations < 1:
        raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")

    b, ncam, _, h, w = pcds.shape
    num_samples = b * ncam
    valid_flat = valid.reshape(num_samples, 1, h, w)
    labels = torch.arange(
        1,
        h * w + 1,
        dtype=torch.float32,
        device=pcds.device,
    ).reshape(1, 1, h, w)
    labels = labels.expand(num_samples, 1, h, w) * valid_flat.to(dtype=torch.float32)

    for _ in range(int(num_iterations)):
        propagated = F.max_pool2d(labels, kernel_size=3, stride=1, padding=1)
        labels = torch.where(valid_flat, propagated, labels.new_zeros(()))

    labels_long = labels.round().to(dtype=torch.int64).reshape(num_samples, h, w)
    valid_long = valid_flat.reshape(num_samples, h, w)
    label_stride = h * w + 1
    sample_offsets = (
        torch.arange(num_samples, dtype=torch.int64, device=pcds.device)
        .reshape(num_samples, 1, 1)
        * label_stride
    )
    global_labels = torch.where(
        valid_long,
        labels_long + sample_offsets,
        torch.zeros((), dtype=torch.int64, device=pcds.device),
    )
    counts = torch.bincount(
        global_labels.reshape(-1),
        minlength=num_samples * label_stride,
    )
    component_sizes = counts[global_labels]
    keep = valid_long & (component_sizes >= int(min_pixels))
    return keep.reshape(b, ncam, h, w)
