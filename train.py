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

    # Update parameters
    colors = colors - lr * grad_colors
    means = (means - lr * grad_means).clamp(0, 1)
    scales = (scales - lr * grad_scales)
    rotations = (rotations - lr * grad_rotations).clamp(0, math.pi)
    # opacities = (opacities - lr * grad_opacities).clamp(0, 1)

animation_path = os.path.join(exp_name, "animation.gif")
frames = [Image.open(img) for img in image_paths]
frames[0].save(animation_path, save_all=True, append_images=frames[1:], duration=num_iters // 5, loop=0)