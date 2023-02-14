import torch
import svox2
from svox2.utils import eval_sh_bases
import os
from os import path
import argparse
import numpy as np
from tqdm import tqdm

import imageio
from util.dataset import datasets
from util.util import viridis_cmap
from util import config_util
from typing import Union, Optional


def render_unw_image(grid: svox2.SparseGrid, camera: svox2.Camera, device=Union[torch.device, str], sigma_thresh: Optional[float] = None, batch_size: int = 5000) -> torch.Tensor:
    grid.to(device)
    rays = cam.gen_rays()
    dirs = rays.dirs / torch.norm(rays.dirs, dim=-1, keepdim=True)
    # viewdirs = dirs
    # gsz = grid._grid_size()
    # dirs = dirs * (grid._scaling * gsz).to(device=dirs.device)
    # delta_scale = 1.0 / dirs.norm(dim=1)
    # dirs *= delta_scale.unsqueeze(-1)
    sh_mult = eval_sh_bases(grid.basis_dim, dirs)
    all_depths = []
    for batch_start in range(0, camera.height * camera.width, batch_size):
        depths = grid.volume_render_depth(rays[batch_start: batch_start+ batch_size], sigma_thresh)
        all_depths.append(depths)
    all_depths = torch.cat(all_depths, dim=0)
    
    pts = rays.origins + rays.dirs * all_depths[..., None]
    samples_sigma, samples_rgb = grid.sample(pts, use_kernel=True)
    rgb_sh = samples_rgb.reshape(-1, 3, grid.basis_dim)
    rgb = torch.clamp_min(torch.sum(sh_mult.unsqueeze(-2) * rgb_sh, dim=-1) + 0.5, 0.0)
        

    print(f'samples_rgb shape: {samples_rgb.shape}')
    return rgb.view(camera.height, camera.width, -1)

def gen_rays(camera: svox2.Camera)-> svox2.Rays:
    pass

parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)

config_util.define_common_args(parser)

parser.add_argument("--train", action='store_true',
                    default=False, help='render train set')
parser.add_argument('--use_sigma_thresh', action='store_true',
                    default=False, help='Use sigma thresh to render depth image')

args = parser.parse_args()
device = 'cuda:0'

render_dir = path.join(path.dirname(args.ckpt), 'play_renderer')

dset = datasets[args.dataset_type](
    args.data_dir, split='test_train' if args.train else 'test', **config_util.build_data_options(args))

grid = svox2.SparseGrid.load(args.ckpt, device=device)

config_util.setup_render_opts(grid.opt, args)

print('Writing to', render_dir)
os.makedirs(render_dir, exist_ok=True)

# NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
# other backends will manually generate rays per frame (slow)
with torch.no_grad():
    n_images = dset.n_images
    n_images_gen = 0
    c2ws = dset.c2w.to(device=device)
    frames = []

    for img_id in tqdm(range(0, n_images)):
        dset_h, dset_w = dset.get_image_size(img_id)
        im_size = dset_h * dset_w

        cam = svox2.Camera(c2ws[img_id],
                           dset.intrins.get('fx', img_id),
                           dset.intrins.get('fy', img_id),
                           dset.intrins.get('cx', img_id),
                           dset.intrins.get('cy', img_id),
                           dset_w, dset_h, ndc_coeffs=dset.ndc_coeffs)

        im_gt = dset.gt[img_id].numpy()

        # # Render RGB image
        im = grid.volume_render_image(cam, use_kernel=True)
        # im.clamp_(0.0, 1.0)
        # im = im.cpu().numpy()
        # im = np.concatenate([im_gt, im], axis=1)
        # img_path = path.join(render_dir, f'{img_id:04d}.png')
        # im = (im * 255).astype(np.uint8)
        # imageio.imwrite(img_path, im)

        # # Render depth image
        im_depth = grid.volume_render_depth_image(cam)
        im_depth = viridis_cmap(im_depth.cpu())
        im_depth = (im_depth * 255).astype(np.uint8)
        img_depth_path = path.join(render_dir, f'{img_id:04d}_depth.png')
        imageio.imwrite(img_depth_path, im_depth)

        # Render unwcolor image
        im_unw = render_unw_image(grid, cam, device, args.sigma_thresh)
        im_unw.clamp_(0.0, 1.0)
        im_unw = im_unw.cpu().numpy()
        im_unw = (im_unw * 255).astype(np.uint8)
        im_unw_path = path.join(render_dir, f'{img_id:04d}_unw.png')
        imageio.imwrite(im_unw_path, im_unw)
        n_images_gen += 1
