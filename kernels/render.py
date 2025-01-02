import triton
import triton.language as tl

@triton.jit
def render_forward_kernel(
    tile_splat_counts,  # (num_tiles) int32
    tile_splat_indices, # (num_tiles, MAX_SPLATS_PER_TILE) int32
    means,              # (N, 2) float32
    scales,             # (N, 2) float32
    rotations,          # (N) float32
    colors,             # (N, 3) float32
    opacities,          # (N) float32
    out_image,          # (H, W, 3) float32
    H: tl.constexpr,
    W: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    MAX_SPLATS_PER_TILE: tl.constexpr,
):
    """
    For each (block_x, block_y), we handle the tile at tile_id = block_y * num_tiles_x + block_x.
    Inside the block, we have up to TILE_SIZE*TILE_SIZE threads (16x16).
    Each thread is responsible for one pixel in the tile.
    """
    # Identify which tile we are in from the 2D grid
    bx = tl.program_id(0)  # tile x index
    by = tl.program_id(1)  # tile y index

    num_tiles_x = (W + TILE_SIZE - 1) // TILE_SIZE
    num_tiles_y = (H + TILE_SIZE - 1) // TILE_SIZE

    tile_id = by * num_tiles_x + bx

    # 2) Flatten the 2D pixel coordinate inside the tile from the 3rd dimension
    local_thread_id = tl.program_id(2)  # range: [0..255] if TILE_SIZE=16
    local_x = local_thread_id % TILE_SIZE
    local_y = local_thread_id // TILE_SIZE

    # 3) Convert (bx, by, local_x, local_y) to global pixel coordinates
    pixel_x = bx * TILE_SIZE + local_x
    pixel_y = by * TILE_SIZE + local_y

    # Skip if pixel is out of bounds (possible if W or H is not multiple of 16)
    if pixel_x >= W or pixel_y >= H:
        return

    # Load how many Gaussians are in this tile
    splat_count = tl.load(tile_splat_counts + tile_id)

    # We will store partial color in registers
    out_r = 0.0
    out_g = 0.0
    out_b = 0.0
    out_a = 0.0

    # -----------------------------------------------------------------------
    # 1. We can load all the relevant Gaussians for this tile into shared memory
    #    or a local array. For demonstration, we do a simple loop, but in practice,
    #    you'd want to chunk them into shared memory in smaller batches for speed.
    # -----------------------------------------------------------------------
    # For demonstration, we do it directly from global tile_splat_indices
    i = 0
    while i < MAX_SPLATS_PER_TILE and i != -1:
    # for i in range(0, MAX_SPLATS_PER_TILE):

        gauss_id = tl.load(tile_splat_indices + tile_id * MAX_SPLATS_PER_TILE + i)
        # gauss_id could be -1 if you are clearing unused, but here we assume it’s valid
        
        # Load the Gaussian parameters
        mean_x = tl.load(means + gauss_id * 2 + 0) * W
        mean_y = tl.load(means + gauss_id * 2 + 1) * H
        sigma_x = tl.load(scales + gauss_id * 2 + 0)
        sigma_y = tl.load(scales + gauss_id * 2 + 1)
        rotation = tl.load(rotations + gauss_id)
        r = tl.load(colors + gauss_id * 3 + 0)
        g = tl.load(colors + gauss_id * 3 + 1)
        b = tl.load(colors + gauss_id * 3 + 2)
        alpha = tl.load(opacities + gauss_id)

        # -------------------------------------------------------------------
        # 2. Evaluate Gaussian coverage at this pixel
        #    For a 2D Gaussian (no rotation or with rotation).
        #    If ignoring rotation, use:
        #         dx = (pixel_x + 0.5) - mean_x
        #         dy = (pixel_y + 0.5) - mean_y
        #         exponent = (dx^2 / (2*sigma_x^2) + dy^2 / (2*sigma_y^2))
        #         value = exp(-exponent)
        #    Then multiply by alpha to get coverage etc.
        # -------------------------------------------------------------------
        px = tl.cast(pixel_x, tl.float32) + 0.5
        py = tl.cast(pixel_y, tl.float32) + 0.5

        # If ignoring rotation in the kernel:
        dx = px - mean_x
        dy = py - mean_y

        # If you want rotation, transform (dx, dy) by the rotation matrix:
        dx_rot =  dx * tl.cos(rotation) + dy * tl.sin(rotation)
        dy_rot = -dx * tl.sin(rotation) + dy * tl.cos(rotation)
        # We'll omit for brevity
        # dx_rot = dx
        # dy_rot = dy

        inv_2_sigx2 = 1.0 / (2.0 * sigma_x * sigma_x)
        inv_2_sigy2 = 1.0 / (2.0 * sigma_y * sigma_y)
        exponent = dx_rot*dx_rot * inv_2_sigx2 + dy_rot*dy_rot * inv_2_sigy2
        gauss_val = tl.exp(-exponent)

        # Multiply by alpha for coverage (or you can have a separate coverage factor)
        coverage = alpha * gauss_val

        # "Over" blending: front-to-back
        # final_color = src_color * src_alpha + dst_color * (1 - src_alpha)
        src_a = gauss_val * alpha
        # out_r = r * src_a + out_r * (1.0 - src_a)
        # out_g = g * src_a + out_g * (1.0 - src_a)
        # out_b = b * src_a + out_b * (1.0 - src_a)
        # out_a = src_a + out_a * (1.0 - src_a)
        out_r += r * src_a
        out_g += g * src_a
        out_b += b * src_a
        # out_a = src_a + out_a * (1.0 - src_a)

        i += 1
        gauss_id = tl.load(tile_splat_indices + tile_id * MAX_SPLATS_PER_TILE + i)
        if gauss_id == -1:
            i = -1

        # If you prefer additive blending or weighting, you can do that as well.

    # -----------------------------------------------------------------------
    # 3. Store final color to global output image
    # -----------------------------------------------------------------------
    # If you’d like to store alpha in the 4th channel, adjust accordingly. 
    out_offset = (pixel_y * W + pixel_x) * 3
    tl.store(out_image + out_offset + 0, out_r)
    tl.store(out_image + out_offset + 1, out_g)
    tl.store(out_image + out_offset + 2, out_b)

@triton.jit
def render_backward_kernel(
    tile_splat_counts,   # (num_tiles) int32
    tile_splat_indices,  # (num_tiles, MAX_SPLATS_PER_TILE) int32
    means,               # (N, 2) float32
    scales,              # (N, 2) float32
    rotations,           # (N) float32
    colors,              # (N, 3) float32
    opacities,           # (N) float32

    grad_out_image,      # (H, W, 3) float32, dL/dOutRGB
    grad_means,          # (N, 2) float32
    grad_scales,         # (N, 2) float32
    grad_rotations,      # (N) float32
    grad_colors,         # (N, 3) float32
    grad_opacities,      # (N) float32

    H: tl.constexpr,
    W: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    MAX_SPLATS_PER_TILE: tl.constexpr,
):
    """
    Backward pass: we assume grad_out_image is the gradient dL/dOutRGB for each pixel.
    We do the same tile-based iteration:
       (bx, by) for tile coords
       local_x, local_y for pixel coords
    Then for each relevant Gaussian in the tile, accumulate the partial derivatives
    into the global grad_* tensors via atomicAdds.
    """
    bx = tl.program_id(0)  # tile x index
    by = tl.program_id(1)  # tile y index

    num_tiles_x = (W + TILE_SIZE - 1) // TILE_SIZE
    num_tiles_y = (H + TILE_SIZE - 1) // TILE_SIZE
    tile_id = by * num_tiles_x + bx

    local_thread_id = tl.program_id(2)
    local_x = local_thread_id % TILE_SIZE
    local_y = local_thread_id // TILE_SIZE

    pixel_x = bx * TILE_SIZE + local_x
    pixel_y = by * TILE_SIZE + local_y

    if pixel_x >= W or pixel_y >= H:
        return

    # dL/dOutRGB at this pixel
    out_offset = (pixel_y * W + pixel_x) * 3
    dOutR = tl.load(grad_out_image + out_offset + 0)
    dOutG = tl.load(grad_out_image + out_offset + 1)
    dOutB = tl.load(grad_out_image + out_offset + 2)

    splat_count = tl.load(tile_splat_counts + tile_id)

    i = 0
    while i < MAX_SPLATS_PER_TILE and i != -1:
        gauss_id = tl.load(tile_splat_indices + tile_id*MAX_SPLATS_PER_TILE + i)

        # ---------------------------
        # load forward pass params
        # ---------------------------
        mean_x = tl.load(means + gauss_id*2 + 0) * W
        mean_y = tl.load(means + gauss_id*2 + 1) * H
        sigma_x = tl.load(scales + gauss_id*2 + 0)
        sigma_y = tl.load(scales + gauss_id*2 + 1)
        rot = tl.load(rotations + gauss_id)
        r = tl.load(colors + gauss_id*3 + 0)
        g = tl.load(colors + gauss_id*3 + 1)
        b = tl.load(colors + gauss_id*3 + 2)
        alpha = tl.load(opacities + gauss_id)

        # center-coord
        px = tl.cast(pixel_x, tl.float32) + 0.5
        py = tl.cast(pixel_y, tl.float32) + 0.5

        dx = px - mean_x
        dy = py - mean_y

        # rotate
        dx_rot =  dx*tl.cos(rot) + dy*tl.sin(rot)
        dy_rot = -dx*tl.sin(rot) + dy*tl.cos(rot)

        inv_2_sx2 = 1.0 / (2.0 * sigma_x * sigma_x)
        inv_2_sy2 = 1.0 / (2.0 * sigma_y * sigma_y)
        exponent = dx_rot*dx_rot*inv_2_sx2 + dy_rot*dy_rot*inv_2_sy2
        gauss_val = tl.exp(-exponent)      # = exp(-exponent)

        # forward out_r, out_g, out_b got:  += r * alpha * gauss_val, etc.
        # so dOutR/d(r,alpha,gauss_val) => chain rule

        # ------------------------------------------------
        # 1) partial wrt color: dL/dr = dOutR * alpha*gauss_val
        # ------------------------------------------------
        # We'll accumulate them in local registers first
        dL_dr    = dOutR * alpha * gauss_val
        dL_dg    = dOutG * alpha * gauss_val
        dL_db    = dOutB * alpha * gauss_val

        # ------------------------------------------------
        # 2) partial wrt alpha
        # ------------------------------------------------
        # sum of each color's contribution
        dL_dAlpha = gauss_val * (r*dOutR + g*dOutG + b*dOutB)

        # ------------------------------------------------
        # 3) partial wrt gauss_val
        # ------------------------------------------------
        dL_dGaussVal = alpha*(r*dOutR + g*dOutG + b*dOutB)
        # gauss_val = exp(-exponent)
        # => d(exponent)/d(gauss_val) = -1 / gauss_val
        # => chain rule => dL/d(exponent) = -gauss_val * dL_dGaussVal
        dL_dExponent = -gauss_val * dL_dGaussVal

        # ------------------------------------------------
        # 4) exponent = dx_rot^2/(2sx^2) + dy_rot^2/(2sy^2)
        #    => partial wrt dx_rot, dy_rot, sigma_x, sigma_y
        # ------------------------------------------------
        # partial exponent wrt dx_rot = dx_rot / (sigma_x^2)
        # partial exponent wrt dy_rot = dy_rot / (sigma_y^2)
        # partial exponent wrt sigma_x = - dx_rot^2 / (sigma_x^3)
        # etc.
        # We'll do them piecewise:
        dExponent_dDxRot = dx_rot / (sigma_x*sigma_x)
        dExponent_dDyRot = dy_rot / (sigma_y*sigma_y)

        # wrt sigma_x
        # exponent wrt sigma_x => derivative of (dx_rot^2 / (2 sigma_x^2)) = dx_rot^2 / (sigma_x^3)
        # with sign + or -?  exponent = dx^2/(2 sx^2) => partial derivative wrt sx => - dx^2/(sx^3)
        dExponent_dSx = -(dx_rot*dx_rot) / (sigma_x*sigma_x*sigma_x)
        dExponent_dSy = -(dy_rot*dy_rot) / (sigma_y*sigma_y*sigma_y)

        # So chain them:
        dL_dDxRot = dL_dExponent * dExponent_dDxRot
        dL_dDyRot = dL_dExponent * dExponent_dDyRot
        dL_dSx    = dL_dExponent * dExponent_dSx
        dL_dSy    = dL_dExponent * dExponent_dSy

        # ------------------------------------------------
        # 5) dx_rot = dx*cos(rot) + dy*sin(rot)
        #    => partial wrt dx, dy, rot
        # ------------------------------------------------
        # dx = px - mean_x => partial wrt mean_x = -1
        # but let's do rotation first:
        # d(dx_rot)/d(rot) = -dx*sin(rot) + dy*cos(rot)
        dDxRot_dRot = -dx*tl.sin(rot) + dy*tl.cos(rot)
        dDyRot_dRot = -(-dx*tl.cos(rot) - dy*tl.sin(rot))  # or do carefully

        # Actually carefully:
        #   dy_rot = -dx*sin(rot) + dy*cos(rot)
        # => d(dy_rot)/d(rot) = -dx*cos(rot) - dy*sin(rot)
        dDyRot_dRot = -dx*tl.cos(rot) - dy*tl.sin(rot)

        dL_dRot = dL_dDxRot * dDxRot_dRot + dL_dDyRot * dDyRot_dRot

        # partial wrt dx = partial wrt mean_x is -1
        # so d(dx_rot)/d(mean_x) = -cos(rot),  d(dy_rot)/d(mean_x) = sin(rot)
        # => total partial wrt mean_x = dL_dDxRot*(-cos(rot)) + dL_dDyRot*(+sin(rot))
        # Similarly for mean_y => d(dx_rot)/d(mean_y) = -sin(rot), ...
        dL_dMeanX = -(dL_dDxRot*tl.cos(rot) + dL_dDyRot*(-tl.sin(rot)))  # note dx = px - mean_x => derivative wrt mean_x is -1
        dL_dMeanY = -(dL_dDxRot*tl.sin(rot) + dL_dDyRot*( tl.cos(rot)))

        # Actually, carefully:
        # dx_rot = dx*cos(rot) + dy*sin(rot)
        # with dx = px - mean_x => partial wrt mean_x => -cos(rot)
        #        dy = py - mean_y => partial wrt mean_y => -sin(rot)
        # => combined => partial wrt mean_x = -cos(rot); wrt mean_y = -sin(rot)
        #
        # dy_rot = -dx*sin(rot) + dy*cos(rot)
        # => partial wrt mean_x => +sin(rot), wrt mean_y => -cos(rot)
        #
        # => Summation for dL/dMeanX = dL_dDxRot * (-cos(rot)) + dL_dDyRot * (+sin(rot))
        #    but we must not forget the minus sign from dx = px - mean_x => 
        # Actually we've built that in. Let's keep it consistent:

        # Now accumulate these into global memory via atomic adds
        # means shape = (N, 2)
        tl.atomic_add(grad_means + gauss_id*2 + 0, dL_dMeanX)
        tl.atomic_add(grad_means + gauss_id*2 + 1, dL_dMeanY)

        tl.atomic_add(grad_scales + gauss_id*2 + 0, dL_dSx)
        tl.atomic_add(grad_scales + gauss_id*2 + 1, dL_dSy)

        tl.atomic_add(grad_rotations + gauss_id, dL_dRot)

        tl.atomic_add(grad_colors + gauss_id*3 + 0, dL_dr)
        tl.atomic_add(grad_colors + gauss_id*3 + 1, dL_dg)
        tl.atomic_add(grad_colors + gauss_id*3 + 2, dL_db)

        tl.atomic_add(grad_opacities + gauss_id, dL_dAlpha)

        i += 1
        gauss_id = tl.load(tile_splat_indices + tile_id*MAX_SPLATS_PER_TILE + i)
        if gauss_id == -1:
            i = -1
