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

import matplotlib.pyplot as plt


def volume_render_given_depth(grid: svox2.SparseGrid,
                              rays: svox2.Rays,
                              depths: torch.Tensor,
                              render_backscatter: bool = False):
    origins = grid.world2grid(rays.origins)
    dirs = rays.dirs / torch.norm(rays.dirs, dim=-1, keepdim=True)
    viewdirs = dirs
    B = dirs.size(0)
    assert rays.origins.size(0) == B
    gsz = grid._grid_size()
    dirs = dirs * (grid._scaling * gsz).to(device=dirs.device)
    delta_scale = 1.0 / dirs.norm(dim=1)
    dirs *= delta_scale.unsqueeze(-1)

    sh_mult = eval_sh_bases(grid.basis_dim, viewdirs)

    pts = grid.world2grid(rays.origins + depths[..., None] * viewdirs)
    dist = (pts - origins).norm(
        dim=-1)  # distance in the grid space from the depths values

    gsz_cu = gsz.to(device=dirs.device)
    invdirs = 1.0 / dirs
    t1 = (-0.5 - origins) * invdirs
    t2 = (gsz_cu - 0.5 - origins) * invdirs

    t = torch.min(t1, t2)
    t[dirs == 0] = -1e9
    t = torch.max(t, dim=-1).values.clamp_min_(grid.opt.near_clip)
    tmax = torch.max(t1, t2)
    tmax[dirs == 0] = 1e9
    tmax = torch.min(tmax, dim=-1).values
    if render_backscatter:
        # Here we march along the ray starting from near clip and stop at the termination point
        tmax = torch.min(tmax, dist)
    else:
        # Here we march along the ray starting from the termination point til the ray end.
        t = torch.max(t, dist)

    log_light_intensity = torch.zeros(B, device=origins.device)
    out_rgb = torch.zeros((B, 3), device=origins.device)
    good_indices = torch.arange(B, device=origins.device)
    origins_ini = origins
    dirs_ini = dirs
    mask = t <= tmax
    good_indices = good_indices[mask]
    origins = origins[mask]
    dirs = dirs[mask]
    #  invdirs = invdirs[mask]
    del invdirs
    t = t[mask]
    sh_mult = sh_mult[mask]
    tmax = tmax[mask]

    while good_indices.numel() > 0:
        pos = origins + t[:, None] * dirs
        pos = pos.clamp_min_(0.0)
        pos[:, 0] = torch.clamp_max(pos[:, 0], gsz_cu[0] - 1)
        pos[:, 1] = torch.clamp_max(pos[:, 1], gsz_cu[1] - 1)
        pos[:, 2] = torch.clamp_max(pos[:, 2], gsz_cu[2] - 1)
        #  print('pym', pos, log_light_intensity)
        l = pos.to(torch.long)
        l.clamp_min_(0)
        l[:, 0] = torch.clamp_max(l[:, 0], gsz_cu[0] - 2)
        l[:, 1] = torch.clamp_max(l[:, 1], gsz_cu[1] - 2)
        l[:, 2] = torch.clamp_max(l[:, 2], gsz_cu[2] - 2)
        pos -= l
        # BEGIN CRAZY TRILERP
        lx, ly, lz = l.unbind(-1)
        links000 = grid.links[lx, ly, lz]
        links001 = grid.links[lx, ly, lz + 1]
        links010 = grid.links[lx, ly + 1, lz]
        links011 = grid.links[lx, ly + 1, lz + 1]
        links100 = grid.links[lx + 1, ly, lz]
        links101 = grid.links[lx + 1, ly, lz + 1]
        links110 = grid.links[lx + 1, ly + 1, lz]
        links111 = grid.links[lx + 1, ly + 1, lz + 1]
        sigma000, rgb000 = grid._fetch_links(links000)
        sigma001, rgb001 = grid._fetch_links(links001)
        sigma010, rgb010 = grid._fetch_links(links010)
        sigma011, rgb011 = grid._fetch_links(links011)
        sigma100, rgb100 = grid._fetch_links(links100)
        sigma101, rgb101 = grid._fetch_links(links101)
        sigma110, rgb110 = grid._fetch_links(links110)
        sigma111, rgb111 = grid._fetch_links(links111)
        wa, wb = 1.0 - pos, pos
        c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
        c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
        c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
        c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
        c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
        c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
        sigma = c0 * wa[:, :1] + c1 * wb[:, :1]
        c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
        c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
        c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
        c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
        c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
        c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
        rgb = c0 * wa[:, :1] + c1 * wb[:, :1]
        # END CRAZY TRILERP

        log_att = (-grid.opt.step_size * torch.relu(sigma[..., 0]) *
                   delta_scale[good_indices])
        weight = torch.exp(
            log_light_intensity[good_indices]) * (1.0 - torch.exp(log_att))
        # [B', 3, n_sh_coeffs]
        rgb_sh = rgb.reshape(-1, 3, grid.basis_dim)
        rgb = torch.clamp_min(
            torch.sum(sh_mult.unsqueeze(-2) * rgb_sh, dim=-1) + 0.5,
            0.0,
        )  # [B', 3]
        rgb = weight[:, None] * rgb[:, :3]
        out_rgb[good_indices] += rgb
        log_light_intensity[good_indices] += log_att
        t += grid.opt.step_size
        mask = t <= tmax
        good_indices = good_indices[mask]
        origins = origins[mask]
        dirs = dirs[mask]
        #  invdirs = invdirs[mask]
        t = t[mask]
        sh_mult = sh_mult[mask]
        tmax = tmax[mask]

    if grid.opt.background_brightness:
        out_rgb += (torch.exp(
            log_light_intensity.unsqueeze(-1) *
            grid.opt.background_brightness))
    return out_rgb


def render_unw_image(grid: svox2.SparseGrid,
                     camera: svox2.Camera,
                     device=Union[torch.device, str],
                     sigma_thresh: Optional[float] = None,
                     batch_size: int = 5000,
                     render_backscatter: bool = False) -> torch.Tensor:
    grid.to(device)
    # Manully generate rays for now
    rays = cam.gen_rays()

    depths = []
    for batch_start in range(0, rays.origins.shape[0], batch_size):
        depths.append(
            grid.volume_render_depth(rays[batch_start:batch_start +
                                          batch_size],
                                     sigma_thresh=sigma_thresh))
    depths = torch.cat(depths, dim=0)

    all_rgb_out = []
    for batch_start in range(0, rays.origins.shape[0], batch_size):
        rgb_out_part = volume_render_given_depth(
            grid, rays[batch_start:batch_start + batch_size],
            depths[batch_start:batch_start + batch_size], render_backscatter)
        all_rgb_out.append(rgb_out_part)

    all_rgb_out = torch.cat(all_rgb_out, dim=0)
    return all_rgb_out.view(camera.height, camera.width, -1)


parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)

config_util.define_common_args(parser)

parser.add_argument("--train",
                    action='store_true',
                    default=False,
                    help='render train set')
parser.add_argument('--use_sigma_thresh',
                    action='store_true',
                    default=False,
                    help='Use sigma thresh to render depth image')
parser.add_argument('--render_unw',
                    action='store_true',
                    default=False,
                    help='render unw image')
parser.add_argument('--render_backscatter',
                    action='store_true',
                    default=False,
                    help='render only backscatter')
args = parser.parse_args()
device = 'cuda:0'

render_dir = path.join(path.dirname(args.ckpt), 'play_renderer')

dset = datasets[args.dataset_type](
    args.data_dir,
    split='test_train' if args.train else 'test',
    **config_util.build_data_options(args))

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
                           dset_w,
                           dset_h,
                           ndc_coeffs=dset.ndc_coeffs)

        im_gt = dset.gt[img_id].numpy()

        # Render RGB image
        im = grid.volume_render_image(cam, use_kernel=True)
        im.clamp_(0.0, 1.0)
        im = im.cpu().numpy()
        im = np.concatenate([im_gt, im], axis=1)
        img_path = path.join(render_dir, f'{img_id:04d}.png')
        im = (im * 255).astype(np.uint8)
        imageio.imwrite(img_path, im)

        # Render depth image
        im_depth = grid.volume_render_depth_image(
            cam, args.sigma_thresh if args.use_sigma_thresh else None)
        im_depth = viridis_cmap(im_depth.cpu())
        im_depth = (im_depth * 255).astype(np.uint8)
        img_depth_path = path.join(render_dir, f'{img_id:04d}_depth.png')
        imageio.imwrite(img_depth_path, im_depth)

        if args.render_unw:
            # Render unwcolor image
            im_unw = render_unw_image(
                grid, cam, device,
                args.sigma_thresh if args.use_sigma_thresh else None, render_backscatter=args.render_backscatter)
            im_unw.clamp_(0.0, 1.0)
            im_unw = im_unw.cpu().numpy()
            im_unw = np.concatenate([im_gt, im_unw], axis=1)
            im_unw_path = path.join(render_dir, f'{img_id:04d}_unw.png')
            im_unw = (im_unw * 255).astype(np.uint8)
            imageio.imwrite(im_unw_path, im_unw)

        n_images_gen += 1
