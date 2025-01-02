import torch

from .preprocess import preprocess_kernel
from .render import (
    render_forward_kernel,
    render_backward_kernel
)

def render(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    target_image: torch.Tensor,
    device: torch.device,
    H: int, W: int,
    sigma_factor: float = 3.0,
    TILE_SIZE: int = 16,
    MAX_SPLATS_PER_TILE: int = 100,  
) -> torch.Tensor:

    N = means.shape[0]
    num_tiles_x = (W + TILE_SIZE - 1) // TILE_SIZE
    num_tiles_y = (H + TILE_SIZE - 1) // TILE_SIZE
    num_tiles = num_tiles_x * num_tiles_y
    
    tile_splat_counts = torch.zeros(num_tiles, dtype=torch.int32, device=device)
    tile_splat_indices = torch.full((num_tiles, MAX_SPLATS_PER_TILE), -1, dtype=torch.int32, device=device)
    
    # Step 1: Assign each Gaussian to relevant 16x16 tiles
    grid = lambda meta: (N,)
    preprocess_kernel[grid](
        N,
        means,
        scales,
        rotations,
        H,
        W,
        tile_splat_counts,
        tile_splat_indices,
        sigma_factor,
        TILE_SIZE,
        MAX_SPLATS_PER_TILE,
    )
    
    out_image = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    
    # Step 2: For each tile blend the splats which are assigned to it
    raster_grid = (num_tiles_x, num_tiles_y, TILE_SIZE * TILE_SIZE)
    render_forward_kernel[raster_grid](
        tile_splat_counts,
        tile_splat_indices,
        means, scales, rotations,
        colors, opacities,
        out_image,
        H,
        W,
        TILE_SIZE,
        MAX_SPLATS_PER_TILE,
        num_warps=1,
        num_stages=1
    )

    # Allocate grad buffers
    grad_means = torch.zeros((N, 2), dtype=torch.float32, device=device)
    grad_scales = torch.zeros((N, 2), dtype=torch.float32, device=device)
    grad_rotations = torch.zeros((N,), dtype=torch.float32, device=device)
    grad_colors = torch.zeros((N, 3), dtype=torch.float32, device=device)
    grad_opacities = torch.zeros((N,), dtype=torch.float32, device=device)
    target_image = target_image.to(dtype=torch.float32, device=device)

    grad_out_image = (out_image > target_image).float() * 2.0 - 1.0

    # Launch the backward kernel
    raster_grid = (num_tiles_x, num_tiles_y, TILE_SIZE * TILE_SIZE)
    render_backward_kernel[raster_grid](
        tile_splat_counts,
        tile_splat_indices,
        means,
        scales,
        rotations,
        colors,
        opacities,
        grad_out_image,
        grad_means,
        grad_scales,
        grad_rotations,
        grad_colors,
        grad_opacities,
        H, W,
        TILE_SIZE,
        MAX_SPLATS_PER_TILE,
        num_warps=1,
        num_stages=1
    )
    
    return out_image, grad_means, grad_scales, grad_rotations, grad_colors, grad_opacities