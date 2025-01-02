import triton
import triton.language as tl

from .utils import (
    activation_forward,
)

@triton.jit
def preprocess_kernel(
    N, means, scales, rotations, H: tl.constexpr, W: tl.constexpr,
    tile_splat_counts, tile_splat_indices,
    sigma_factor: tl.constexpr, TILE_SIZE: tl.constexpr,
    MAX_SPLATS_PER_TILE: tl.constexpr,
    scale_act: tl.constexpr = 1,
    SCALE_MUL: tl.constexpr = 5,
    SCALE_ADD: tl.constexpr = 1
):
    gid = tl.program_id(0)
    if gid >= N: return

    mx = tl.load(means + gid*2 + 0) * W
    my = tl.load(means + gid*2 + 1) * H
    raw_sx = tl.load(scales + gid*2 + 0)
    raw_sy = tl.load(scales + gid*2 + 1)
    sx = activation_forward(raw_sx, scale_act) * SCALE_MUL + SCALE_ADD
    sy = activation_forward(raw_sy, scale_act) * SCALE_MUL + SCALE_ADD
    rot = tl.load(rotations + gid)  # not used in this bounding box calc

    s = tl.maximum(sx, sy)
    minx = tl.maximum(0.0, mx - sigma_factor*s)
    maxx = tl.minimum(float(W), mx + sigma_factor*s)
    miny = tl.maximum(0.0, my - sigma_factor*s)
    maxy = tl.minimum(float(H), my + sigma_factor*s)

    tx0 = tl.floor(minx / TILE_SIZE)
    ty0 = tl.floor(miny / TILE_SIZE)
    tx1 = tl.floor(maxx / TILE_SIZE)
    ty1 = tl.floor(maxy / TILE_SIZE)

    ix0 = tl.maximum(tl.cast(tx0, tl.int32), 0)
    iy0 = tl.maximum(tl.cast(ty0, tl.int32), 0)
    ntx = (W + TILE_SIZE - 1) // TILE_SIZE
    nty = (H + TILE_SIZE - 1) // TILE_SIZE
    ix1 = tl.minimum(tl.cast(tx1, tl.int32), ntx - 1)
    iy1 = tl.minimum(tl.cast(ty1, tl.int32), nty - 1)

    for yy in range(iy0, iy1 + 1):
        for xx in range(ix0, ix1 + 1):
            tile_id = yy * ntx + xx
            off = tl.atomic_add(tile_splat_counts + tile_id, 1)
            if off < MAX_SPLATS_PER_TILE:
                tl.store(tile_splat_indices + tile_id*MAX_SPLATS_PER_TILE + off, gid)
