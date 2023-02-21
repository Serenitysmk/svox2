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

        # # Render RGB image
        # im = grid.volume_render_image(cam, use_kernel=True)
        # im.clamp_(0.0, 1.0)
        # im = im.cpu().numpy()
        # im = np.concatenate([im_gt, im], axis=1)
        # img_path = path.join(render_dir, f'{img_id:04d}.png')
        # im = (im * 255).astype(np.uint8)
        # imageio.imwrite(img_path, im)

        # # Render depth image
        # im_depth = grid.volume_render_depth_image(
        #     cam, args.sigma_thresh if args.use_sigma_thresh else None)
        # im_depth = viridis_cmap(im_depth.cpu())
        # im_depth = (im_depth * 255).astype(np.uint8)
        # img_depth_path = path.join(render_dir, f'{img_id:04d}_depth.png')
        # imageio.imwrite(img_depth_path, im_depth)

        if args.render_unw:
            # Render unwcolor image (integrated version)
            im_unw = grid.volume_render_unw_image(
                cam,
                use_kernel=True,
                sigma_thresh=args.sigma_thresh
                if args.use_sigma_thresh else None)
            im_unw.clamp_(0.0, 1.0)
            im_unw = im_unw.cpu().numpy()
            im_unw = np.concatenate([im_gt, im_unw], axis=1)
            im_unw_path = path.join(render_dir, f'{img_id:04d}_unw.png')
            im_unw = (im_unw * 255).astype(np.uint8)
            imageio.imwrite(im_unw_path, im_unw)
        n_images_gen += 1
        break