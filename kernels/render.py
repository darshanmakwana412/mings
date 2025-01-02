import triton
import triton.language as tl

from .utils import (
    activation_forward,
    activation_backward
)

@triton.jit
def render_forward_kernel(
    tile_splat_counts, tile_splat_indices,
    means, scales, rotations, colors, opacities, out_image,
    H: tl.constexpr, W: tl.constexpr,
    TILE_SIZE: tl.constexpr, MAX_SPLATS_PER_TILE: tl.constexpr,
    scale_act: tl.constexpr=1,  # 0=linear,1=sigmoid,2=tanh
    color_act: tl.constexpr=1,   # same idea
    SCALE_MUL: tl.constexpr = 5,
    SCALE_ADD: tl.constexpr = 1
):
    bx = tl.program_id(0)
    by = tl.program_id(1)
    tix = (W + TILE_SIZE - 1) // TILE_SIZE
    tile_id = by * tix + bx
    lid = tl.program_id(2)
    lx = lid % TILE_SIZE
    ly = lid // TILE_SIZE
    px = bx*TILE_SIZE + lx
    py = by*TILE_SIZE + ly
    if px >= W or py >= H: return

    r_out, g_out, b_out = 0.0, 0.0, 0.0
    i = 0
    while i < MAX_SPLATS_PER_TILE and i != -1:
        gid = tl.load(tile_splat_indices + tile_id*MAX_SPLATS_PER_TILE + i)
        mx = tl.load(means + gid*2 + 0) * W
        my = tl.load(means + gid*2 + 1) * H

        sx = tl.load(scales + gid*2 + 0)
        sy = tl.load(scales + gid*2 + 1)
        # Apply activation to scales
        sx = activation_forward(sx, scale_act) * SCALE_MUL + SCALE_ADD
        sy = activation_forward(sy, scale_act) * SCALE_MUL + SCALE_ADD

        rot = tl.load(rotations + gid)

        rr = tl.load(colors + gid*3 + 0)
        gg = tl.load(colors + gid*3 + 1)
        bb = tl.load(colors + gid*3 + 2)
        # Apply activation to colors
        rr = activation_forward(rr, color_act)
        gg = activation_forward(gg, color_act)
        bb = activation_forward(bb, color_act)

        a = tl.load(opacities + gid)

        fx = tl.cast(px, tl.float32) + 0.5
        fy = tl.cast(py, tl.float32) + 0.5
        dx = fx - mx
        dy = fy - my
        dxr = dx*tl.cos(rot) + dy*tl.sin(rot)
        dyr = -dx*tl.sin(rot) + dy*tl.cos(rot)
        e = (dxr*dxr)/(2.0*sx*sx) + (dyr*dyr)/(2.0*sy*sy)
        val = tl.exp(-e)
        alpha = a * val

        r_out += rr * alpha
        g_out += gg * alpha
        b_out += bb * alpha

        i += 1
        if i < MAX_SPLATS_PER_TILE:
            gid2 = tl.load(tile_splat_indices + tile_id*MAX_SPLATS_PER_TILE + i)
            if gid2 == -1:
                i = -1

    off = (py*W + px)*3
    tl.store(out_image + off + 0, r_out)
    tl.store(out_image + off + 1, g_out)
    tl.store(out_image + off + 2, b_out)


@triton.jit
def render_backward_kernel(
    tile_splat_counts, tile_splat_indices,
    means, scales, rotations, colors, opacities,
    grad_out_image,
    grad_means, grad_scales, grad_rotations,
    grad_colors, grad_opacities,
    H: tl.constexpr, W: tl.constexpr,
    TILE_SIZE: tl.constexpr, MAX_SPLATS_PER_TILE: tl.constexpr,
    scale_act: tl.constexpr=1,  # same flag as forward
    color_act: tl.constexpr=1,
    SCALE_MUL: tl.constexpr = 5,
    SCALE_ADD: tl.constexpr = 1
):
    bx = tl.program_id(0)
    by = tl.program_id(1)
    tix = (W + TILE_SIZE - 1) // TILE_SIZE
    tile_id = by * tix + bx
    lid = tl.program_id(2)
    lx = lid % TILE_SIZE
    ly = lid // TILE_SIZE
    px = bx*TILE_SIZE + lx
    py = by*TILE_SIZE + ly
    if px >= W or py >= H: return

    off = (py*W + px)*3
    dR = tl.load(grad_out_image + off + 0)
    dG = tl.load(grad_out_image + off + 1)
    dB = tl.load(grad_out_image + off + 2)

    i = 0
    while i < MAX_SPLATS_PER_TILE and i != -1:
        gid = tl.load(tile_splat_indices + tile_id*MAX_SPLATS_PER_TILE + i)

        mx = tl.load(means + gid*2 + 0) * W
        my = tl.load(means + gid*2 + 1) * H

        raw_sx = tl.load(scales + gid*2 + 0)
        raw_sy = tl.load(scales + gid*2 + 1)
        sx = activation_forward(raw_sx, scale_act) * SCALE_MUL + SCALE_ADD
        sy = activation_forward(raw_sy, scale_act) * SCALE_MUL + SCALE_ADD

        rot = tl.load(rotations + gid)

        raw_rr = tl.load(colors + gid*3 + 0)
        raw_gg = tl.load(colors + gid*3 + 1)
        raw_bb = tl.load(colors + gid*3 + 2)
        rr = activation_forward(raw_rr, color_act)  # forward color
        gg = activation_forward(raw_gg, color_act)
        bb = activation_forward(raw_bb, color_act)

        a  = tl.load(opacities + gid)

        fx = tl.cast(px, tl.float32) + 0.5
        fy = tl.cast(py, tl.float32) + 0.5
        dx = fx - mx
        dy = fy - my
        dxr = dx*tl.cos(rot) + dy*tl.sin(rot)
        dyr = -dx*tl.sin(rot) + dy*tl.cos(rot)
        e = (dxr*dxr)/(2*sx*sx) + (dyr*dyr)/(2*sy*sy)
        val = tl.exp(-e)

        # Forward produced r_out += rr*(a*val), etc.
        dLdr = dR*a*val
        dLdg = dG*a*val
        dLdb = dB*a*val
        dLa  = val*(rr*dR + gg*dG + bb*dB)
        dLv  = a*(rr*dR + gg*dG + bb*dB)  # wrt val
        dLe  = -val*dLv  # derivative wrt exponent

        dExDxr = dxr/(sx*sx)
        dExDyr = dyr/(sy*sy)
        dExDsx = -(dxr*dxr)/(sx*sx*sx)
        dExDsy = -(dyr*dyr)/(sy*sy*sy)
        dLdDxr = dLe*dExDxr
        dLdDyr = dLe*dExDyr
        dLdSx  = dLe*dExDsx
        dLdSy  = dLe*dExDsy

        dDxRot = -dx*tl.sin(rot) + dy*tl.cos(rot)
        dDyRot = -dx*tl.cos(rot) - dy*tl.sin(rot)
        dLdRot = dLdDxr*dDxRot + dLdDyr*dDyRot

        dMx = -(dLdDxr*tl.cos(rot) + dLdDyr*(-tl.sin(rot)))
        dMy = -(dLdDxr*tl.sin(rot) + dLdDyr*tl.cos(rot))

        # ---- Activation derivative for scales & colors ----
        # dLdRawSx = dLdSx * activation_backward(sx, 1, scale_act) 
        # but we need partial wrt the *activated* scale => chain rule
        dLdRawSx = activation_backward(sx, dLdSx, scale_act) * SCALE_MUL
        dLdRawSy = activation_backward(sy, dLdSy, scale_act) * SCALE_MUL

        dLdRawRr = activation_backward(rr, dLdr, color_act)
        dLdRawGg = activation_backward(gg, dLdg, color_act)
        dLdRawBb = activation_backward(bb, dLdb, color_act)

        # ---------------------------------------------------
        # Atomic adds to global grads
        # ---------------------------------------------------
        tl.atomic_add(grad_means + gid*2 + 0, dMx)
        tl.atomic_add(grad_means + gid*2 + 1, dMy)

        tl.atomic_add(grad_scales + gid*2 + 0, dLdRawSx)
        tl.atomic_add(grad_scales + gid*2 + 1, dLdRawSy)

        tl.atomic_add(grad_rotations + gid, dLdRot)

        tl.atomic_add(grad_colors + gid*3 + 0, dLdRawRr)
        tl.atomic_add(grad_colors + gid*3 + 1, dLdRawGg)
        tl.atomic_add(grad_colors + gid*3 + 2, dLdRawBb)

        tl.atomic_add(grad_opacities + gid, dLa)

        i += 1
        if i < MAX_SPLATS_PER_TILE:
            gid2 = tl.load(tile_splat_indices + tile_id*MAX_SPLATS_PER_TILE + i)
            if gid2 == -1:
                i = -1