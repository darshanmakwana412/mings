import triton
import triton.language as tl

@triton.jit
def preprocess_kernel(
    N,           # number of gaussians
    means,       # (N, 2) float32
    scales,      # (N, 2) float32
    rotations,   # (N) float32, optional (angle in radians)
    H: tl.constexpr,
    W: tl.constexpr,
    tile_splat_counts,    # (num_tiles) int32
    tile_splat_indices,   # (num_tiles, MAX_SPLATS_PER_TILE) int32
    sigma_factor: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    MAX_SPLATS_PER_TILE: tl.constexpr,
):
    """
    We launch N threads in 1D. Each thread processes exactly one Gaussian.
    """
    gauss_id = tl.program_id(0)

    if gauss_id >= N:
        return

    # Load the Gaussian parameters
    mean_x = tl.load(means + gauss_id * 2 + 0) * W
    mean_y = tl.load(means + gauss_id * 2 + 1) * H
    sigma_x = tl.load(scales + gauss_id * 2 + 0)
    sigma_y = tl.load(scales + gauss_id * 2 + 1)
    rotation = tl.load(rotations + gauss_id)  # if used

    # -----------------------------------------------------------------------
    # 1. Compute bounding box of the 3-sigma ellipse
    #    For simplicity, we approximate bounding box in screen coords.
    # -----------------------------------------------------------------------
    # If ignoring rotation:
    #   bounding box: [mean_x - 3*sigma_x, mean_x + 3*sigma_x]
    #                 [mean_y - 3*sigma_y, mean_y + 3*sigma_y]
    # If applying rotation, you might transform corners by the rotation matrix
    # for a more accurate bounding box. For brevity, we omit that detail here.
    # -----------------------------------------------------------------------
    max_scale = tl.maximum(sigma_x, sigma_y)
    min_x = mean_x - sigma_factor * max_scale
    max_x = mean_x + sigma_factor * max_scale
    min_y = mean_y - sigma_factor * max_scale
    max_y = mean_y + sigma_factor * max_scale

    # Clip to image bounds
    min_x = tl.maximum(0.0, min_x)
    min_y = tl.maximum(0.0, min_y)
    max_x = tl.minimum(float(W), max_x)
    max_y = tl.minimum(float(H), max_y)

    # Convert bounding box to tile coordinates
    # tile_x = pixel_x // TILE_SIZE
    tile_min_x = tl.floor(min_x / TILE_SIZE)
    tile_min_y = tl.floor(min_y / TILE_SIZE)
    tile_max_x = tl.floor(max_x / TILE_SIZE)
    tile_max_y = tl.floor(max_y / TILE_SIZE)

    # integer tile range
    tile_min_x_i = tl.maximum(tl.cast(tile_min_x, tl.int32), 0)
    tile_min_y_i = tl.maximum(tl.cast(tile_min_y, tl.int32), 0)

    # Because we did a clamp above, we also ensure we don't exceed the tile dims
    num_tiles_x = (W + TILE_SIZE - 1) // TILE_SIZE
    num_tiles_y = (H + TILE_SIZE - 1) // TILE_SIZE

    tile_max_x_i = tl.minimum(tl.cast(tile_max_x, tl.int32), num_tiles_x - 1)
    tile_max_y_i = tl.minimum(tl.cast(tile_max_y, tl.int32), num_tiles_y - 1)

    # -----------------------------------------------------------------------
    # 2. Iterate over all tiles that the bounding box overlaps
    #    and add gauss_id to tile_splat_indices
    # -----------------------------------------------------------------------
    # We'll do a nested loop. In Triton, you should be mindful of looping, but
    # N is presumably large. If this is too slow, you'd consider different data
    # structures or CPU-based assignment. For demonstration, it’s okay.
    # -----------------------------------------------------------------------
    for ty in range(tile_min_y_i, tile_max_y_i + 1):  # clamp to some max to avoid infinite loops
        for tx in range(tile_min_x_i, tile_max_x_i + 1):
            # tile index in a row-major sense
            tile_id = ty * num_tiles_x + tx
            # get offset via atomicAdd
            offset = tl.atomic_add(tile_splat_counts + tile_id, 1)
            if offset < MAX_SPLATS_PER_TILE:
                tl.store(tile_splat_indices + tile_id * MAX_SPLATS_PER_TILE + offset, gauss_id)
            # else we ignore this splat if we overflow