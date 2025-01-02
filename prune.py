import os
from tqdm import tqdm
import torch
import math
import numpy as np
from PIL import Image

from kernels import render

exp_name = "toucan"
image_dir = os.path.join(exp_name, "images")
os.makedirs(image_dir, exist_ok=True)

N = 1000
H, W = 256, 256
lr = 1e-3
num_iters = 200

device = torch.device("cuda:0")
means = torch.rand((N, 2), device=device)
scales = torch.randn((N, 2), device=device)
rotations = torch.rand((N,), device=device) * math.pi
colors = torch.randn((N, 3), device=device)
opacities = torch.ones((N,), device=device)

target_image_path = "./imgs/toucan_256.png"
target_image = np.array(Image.open(target_image_path).resize((H, W))).astype(np.float32)
target_image = torch.tensor(target_image, device=device) / 255.0

# -- Hyperparameters for pruning / densification --
PRUNE_THRESHOLD = 0.05  # Remove Gaussians whose effective opacity is below this
MAX_NEW_GAUSSIANS_PER_ITER = 100  # How many new Gaussians to add in densification
SCALE_NEW_GAUSSIAN = 2  # A typical scale for newly added Gaussians
# -------------------------------------------------


def prune_and_densify(
    means,
    scales,
    rotations,
    colors,
    opacities,
    out_image,
    target_image,
    device,
    prune_threshold=PRUNE_THRESHOLD,
    max_new=MAX_NEW_GAUSSIANS_PER_ITER,
    scale_new=SCALE_NEW_GAUSSIAN,
):
    """
    Prunes Gaussians that are low-opacity and densifies regions of high error.
    """

    # ------------------------------------------------
    # 1) Pruning step:
    #    Remove Gaussians that have very low opacity.
    # ------------------------------------------------
    keep_mask = (opacities > prune_threshold)
    means = means[keep_mask]
    scales = scales[keep_mask]
    rotations = rotations[keep_mask]
    colors = colors[keep_mask]
    opacities = opacities[keep_mask]

    # ------------------------------------------------
    # 2) Densification step:
    #    Identify the largest-error pixels and add new Gaussians there.
    #    We'll do a simple approach: pick top K error locations,
    #    place new Gaussians there with random color/opacity near the target.
    # ------------------------------------------------
    # Compute the per-pixel error map (L2 or L1). We'll use L1 for simplicity.
    # out_image: (H, W, 3)
    # target_image: (H, W, 3)
    error_map = (target_image - out_image).abs().mean(dim=-1)  # shape (H, W)

    # Flatten the error_map for easy top-k indexing
    flat_error = error_map.view(-1)
    # Get top-k indices
    k = min(max_new, flat_error.numel())
    topk_values, topk_indices = torch.topk(flat_error, k)

    # Convert these top-k indices back to (row, col)
    rows = topk_indices // W
    cols = topk_indices % W

    # Create new Gaussians at these high-error locations
    # - Means will be normalized to [0,1], so we do row/H, col/W
    # - Colors from the target image at that pixel
    # - Scales set to a fixed or random small range
    # - Opacity set to 1 (or something near 1)
    new_means = torch.stack([cols.float() / W, rows.float() / H], dim=-1)
    new_colors = target_image[rows, cols, :]  # directly from the target
    new_scales = torch.ones((k, 2), device=device) * scale_new
    new_rotations = torch.zeros((k,), device=device)
    new_opacities = torch.ones((k,), device=device)

    # Concatenate them to existing sets
    means = torch.cat([means, new_means.to(device)], dim=0)
    scales = torch.cat([scales, new_scales], dim=0)
    rotations = torch.cat([rotations, new_rotations], dim=0)
    colors = torch.cat([colors, new_colors], dim=0)
    opacities = torch.cat([opacities, new_opacities], dim=0)

    return means, scales, rotations, colors, opacities


# Main loop
image_paths = []
for i in tqdm(range(num_iters)):
    out_image, grad_means, grad_scales, grad_rotations, grad_colors, grad_opacities = render(
        means, scales, rotations, colors, opacities,
        target_image,
        device, H, W,
        sigma_factor=3.0,
        TILE_SIZE=16,
        MAX_SPLATS_PER_TILE=300,
    )

    # Save the current rendered image
    out_image_np = (out_image.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
    image_path = os.path.join(image_dir, f"frame_{i:04d}.png")
    Image.fromarray(out_image_np).save(image_path)
    image_paths.append(image_path)

    # Gradient descent step
    colors = colors - lr * grad_colors
    means = (means - lr * grad_means).clamp(0, 1)
    scales = (scales - lr * grad_scales)
    rotations = (rotations - lr * grad_rotations).clamp(0, math.pi)
    opacities = (opacities - lr * grad_opacities).clamp(0, 1)

    # Adaptive strategy: prune and then densify
    if i % 50 == 0:
        means, scales, rotations, colors, opacities = prune_and_densify(
            means,
            scales,
            rotations,
            colors,
            opacities,
            out_image.detach(),
            target_image,
            device,
            prune_threshold=PRUNE_THRESHOLD,
            max_new=MAX_NEW_GAUSSIANS_PER_ITER,
            scale_new=SCALE_NEW_GAUSSIAN,
        )


# Finally, create a GIF animation from all saved frames
animation_path = os.path.join(exp_name, "animation.gif")
frames = [Image.open(img) for img in image_paths]
frames[0].save(
    animation_path,
    save_all=True,
    append_images=frames[1:],
    duration=num_iters // 5,
    loop=0
)
