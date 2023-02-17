# This script attempts to copy svox2.py line by line for better understanding the code, not for research.
# Hopefully, this script can reproduce as well the same result as the original one
import torch
from torch import nn, autograd
import torch.nn.functional as F
from typing import Union, List, NamedTuple, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from functools import reduce
from . import utils

import matplotlib.pyplot as plt

BASIS_TYPE_SH = 1
BASIS_TYPE_3D_TEXTURE = 4
BASIS_TYPE_MLP = 255
MAX_SH_BASES = 10


@dataclass
class RenderOptions:
    """
    Rendering options, see comments
    available:
    :param backend: str, renderer backend
    :param background_brightness: float
    :param step_size: float, step size for rendering
    :param sigma_thresh: float
    :param stop_thresh: float
    """

    backend: str = "cuvol"  # One of cuvol, svox1, nvol

    # [0, 1], the background color black-white
    background_brightness: float = 1.0

    # Step size, in normalized voxels (not used for svox1)
    step_size: float = 0.5
    #  (i.e. 1 = 1 voxel width, different from svox where 1 = grid width!)

    # Voxels with sigmas < this are ignored, in [0, 1]
    sigma_thresh: float = 1e-10
    #  make this higher for fast rendering

    stop_thresh: float = (
        # Stops rendering if the remaining light intensity/termination, in [0, 1]
        1e-7
    )
    #  probability is <= this much (forward only)
    #  make this higher for fast rendering

    # Make the last sample opaque (for forward-facing)
    last_sample_opaque: bool = False

    near_clip: float = 0.0
    use_spheric_clip: bool = False

    # Noise to add to sigma (only if randomize=True)
    random_sigma_std: float = 1.0
    random_sigma_std_background: float = 1.0        # Noise to add to sigma
    # (for the BG model; only if randomize=True)


@dataclass
class Rays:
    origins: torch.Tensor
    dirs: torch.Tensor

    def __getitem__(self, key):
        return Rays(self.origins[key], self.dirs[key])

    @property
    def is_cuda(self):
        return self.origins.is_cuda() and self.dirs.is_cuda()


@dataclass
class Camera:
    c2w: torch.Tensor
    fx: float = 1111.11
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    width: int = 800
    height: int = 800

    ndc_coeffs: Union[Tuple[float, float], List[float]] = (-1.0, 1.0)

    @property
    def fx_val(self):
        return self.fx

    @property
    def fy_val(self):
        return self.fx if self.fy is None else self.fy

    @property
    def cx_val(self):
        return self.width * 0.5 if self.cx is None else self.cx

    @property
    def cy_val(self):
        return self.height * 0.5 if self.cy is None else self.cy

    @property
    def using_ndc(self):
        return self.ndc_coeffs[0] > 0.0

    @property
    def is_cuda(self):
        return self.c2w.is_cuda()

    def gen_rays(self) -> Rays:
        """
        Generating rays for this camera
        :return: (origins (H*W, 3), dirs (H*W, 3))
        """
        origins = self.c2w[..., :3, 3].expand(
            self.height * self.width, -1).contiguous()
        yy, xx = torch.meshgrid(torch.arange(self.height, dtype=torch.float64,
                                device=self.c2w.device) + 0.5, torch.arange(self.width, dtype=torch.float64) + 0.5)

        xx = (xx - self.cx_val) / self.fx_val
        yy = (yy - self.cy_val) / self.fy_val
        zz = torch.ones_like(xx)
        dirs = torch.stack([xx, yy, zz], dim=-1)
        del xx, yy, zz
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = (self.c2w[None, :3, :3].double() @
                dirs.reshape(-1, 3, 1)).squeeze().float()
        return Rays(origins, dirs)


class SparseGrid(nn.Module):
    """
    Main sparse grid data structure.
    initially it will be a dense grid of resolution <reso>
    Only float32 is supported.

    :param reso: int or List[int, int, int], resolution for resampled grid, as in constructor
    :param radius: float or List[float, float, float], the 1/2 side length of the grid, optionally in each direction
    :param center: float or List[float ,float, float], the center of the grid
    :param basis_type: int, basis type; may use svox2.BASIS_TYPE_*(1 = SH, 4 = learned 3D texture, 255 = learned MLP)
    :param basis_dim: int, size of basis / number of SH components (must be square number in case of SH)
    :param device: torch.device or str, device to store the grid
    """

    def __init__(self,
                 reso: Union[int, List[int], Tuple[int, int, int]] = 128,
                 radius: Union[float, List[float]] = 1.0,
                 center: Union[float, List[float]] = [
                     0.0, 0.0, 0.0],
                 basis_type: int = BASIS_TYPE_SH,
                 basis_dim: int = 9,
                 device: Union[torch.device, str] = 'cpu'):
        super().__init__()
        self.basis_type = basis_type
        if basis_type == BASIS_TYPE_SH:
            assert basis_dim == 9, "For simlicity, I only do basis when it is 9 such that I can be lazy to skip the squre root check here"
        assert (
            basis_dim >= 1 and basis_dim <= MAX_SH_BASES
        ), f'basis_dim 1-{MAX_SH_BASES} is supported'
        self.basis_dim = basis_dim

        if isinstance(reso, int):
            reso = [reso] * 3
        else:
            assert (len(reso) ==
                    3), "reso must be an integer or indexable object of 3 ints"

        if isinstance(radius, float) or isinstance(radius, int):
            radius = [radius] * 3
        if isinstance(radius, torch.Tensor):
            radius = radius.to(device="cpu", dtype=torch.float32)
        else:
            radius = torch.tensor(radius, dtype=torch.float32, device="cpu")
        if isinstance(center, torch.Tensor):
            center = center.to(device="cpu", dtype=torch.float32)
        else:
            center = torch.tensor(center, dtype=torch.float32, device="cpu")

        self.radius: torch.Tensor = radius
        self.center: torch.Tensor = center
        # The volume cube center will eventually sit at (0.5, 0.5, 0.5) with a total length (2 * radius) of 1.0
        # after applying this offset and scaling
        # so, this is to calculate the offset and scale given the current radius and center
        self._offset = 0.5 * (1.0 - self.center / self.radius)
        self._scaling = 0.5 / self.radius

        n3: int = reduce(lambda x, y: x * y, reso)

        # Skip Z-order and sphere bound stuff
        self.capacity = n3

        init_links = torch.arange(n3, device=device, dtype=torch.int32)

        # define parameters here
        self.density_data = nn.Parameter(
            torch.zeros(self.capacity, 1, dtype=torch.float32, device=device)
        )
        # Called sh for legacy reasons, but it's just the coefficents for whatever
        # spherical basis functions
        self.sh_data = nn.Parameter(
            torch.zeros(self.capacity, self.basis_dim * 3,
                        dtype=torch.float32, device=device)
        )

        if self.basis_type == BASIS_TYPE_SH:
            self.basis_data = nn.Parameter(
                torch.empty(0, 0, 0, 0, dtype=torch.float32, device=device), requires_grad=False
            )

        # after registering buffer, you have the data as well in self.links
        self.register_buffer("links", init_links.view(reso))
        self.links: torch.Tensor
        self.opt = RenderOptions()
        self.sparse_grad_indexer: Optional[torch.Tensor] = None
        self.sparse_sh_grad_indexer: Optional[torch.Tensor] = None
        self.sparse_background_indexer: Optional[torch.Tensor] = None
        self.density_rms: Optional[torch.Tensor] = None
        self.sh_rms: Optional[torch.Tensor] = None
        # END constructor

    @property
    def data_dim(self):
        """
        Get the number of channels in the data, including color + density
        (similar to svox1)
        """
        return self.sh_data.size(1) + 1

    @property
    def shape(self):
        return list(self.links.shape) + [self.data_dim]

    def _grid_size(self):
        return torch.tensor(self.links.shape, device='cpu', dtype=torch.float32)

    def _fetch_links(self, links: torch.Tensor):
        results_sigma = torch.zeros(
            (links.size(0), 1), device=links.device, dtype=torch.float32)
        results_sh = torch.zeros((links.size(0), self.sh_data.size(
            1)), device=links.device, dtype=torch.float32)
        # (later link can be < 0 means that there is no data here)
        mask = links >= 0
        idxs = links[mask].long()
        results_sigma[mask] = self.density_data[idxs]
        results_sh[mask] = self.sh_data[idxs]
        return results_sigma, results_sh

    def world2grid(self, points: torch.Tensor):
        """
        World coordinates to grid coordinates. Grid coordinates are
        normalized to [0, n_voxels] in each side

        :param points: (N, 3)
        :return: (N, 3)
        """
        gsz = self._grid_size()
        offset = self._offset * gsz - 0.5
        scaling = self._scaling * gsz
        return torch.addcmul(offset.to(device=points.device), points, scaling.to(device=points.device))

    # Most important part of the code, how do I extract the sigma and the color given a set of 3D points in the world

    def sample(self, points: torch.Tensor, use_kernel: bool = False, grid_coords: bool = False, want_colors: bool = True):
        """
        Grid sampling with triliner interpolation
        """
        if use_kernel:
            raise NotImplementedError("Do not implement this part")
        else:
            if not grid_coords:
                points = self.world2grid(points)
            points.clamp_min_(0.0)
            for i in range(3):
                points[:, i].clamp_max_(self.links.size(i)-1)
            l = points.to(torch.long)
            for i in range(3):
                l[:, i].clamp_max_(self.links.size(i)-2)

            # This line corresponds to the trilinear interpolation tutorial
            # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/interpolation/trilinear-interpolation.html
            # gx = location.x * nvoxels; gxi = int(gx); tx = gx - gxi;
            # gy = location.y * nvoxels; gyi = int(gy); ty = gy - gyi;
            # gz = location.z * nvoxels; gzi = int(gz); tz = gz - gzi;
            # where tx, ty, tz are essentially wb here
            # using the grid coords (floating number) - grid coords (integer number)
            wb = points - l
            wa = 1.0 - wb

            # X, Y, Z coords of the sample points (grid coords)
            lx, ly, lz = l.unbind(-1)
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]

            # Fetch link data
            sigma000, rgb000 = self._fetch_links(links000)
            sigma001, rgb001 = self._fetch_links(links001)
            sigma010, rgb010 = self._fetch_links(links010)
            sigma011, rgb011 = self._fetch_links(links011)
            sigma100, rgb100 = self._fetch_links(links100)
            sigma101, rgb101 = self._fetch_links(links101)
            sigma110, rgb110 = self._fetch_links(links110)
            sigma111, rgb111 = self._fetch_links(links111)

            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            samples_sigma = c0 * wa[:, :1] + c1 * wb[:, :1]

            if want_colors:
                c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
                c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
                c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
                c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
                c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
                c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
                samples_rgb = c0 * wa[:, :1] + c1 * wb[:, :1]
            else:
                samples_rgb = torch.empty_like(self.sh_data[:0])
            return samples_sigma, samples_rgb

    def forward(self, points: torch.Tensor, use_kernel: bool = False):
        return self.sample(points, use_kernel=use_kernel)

    # The most complicated part of the code, need to understand them every line to
    def volume_render(self, rays: Rays,
                      use_kernel: bool = False,
                      randomize: bool = False,
                      batch_size: int = 5000,
                      return_raylen: bool = False):
        """
        trilerp gradcheck version
        """
        origins = self.world2grid(rays.origins)
        dirs = rays.dirs / torch.norm(rays.dirs, dim=-1, keepdim=True)
        viewdirs = dirs
        B = dirs.size(0)
        assert origins.size(0) == B
        # I don't understand this part
        gsz = self._grid_size()
        dirs = dirs * (self._scaling * gsz).to(device=dirs.device) # Direction in grid coordinate system (imagine a rectanguler is pressed into a cube, then ray direction must be changed)
        delta_scale = 1.0 / dirs.norm(dim=1) # Just normalize into a unit vector
        dirs *= delta_scale.unsqueeze(-1) 

        if self.basis_type != BASIS_TYPE_SH:
            raise NotImplementedError("Guess what, I don't want to implement this")
        else:
            sh_mult = utils.eval_sh_bases(self.basis_dim, viewdirs)
        invdirs = 1.0 / dirs # What do you want to get from a invert direction like this
        gsz_cu = gsz.to(device=dirs.device)
        t1 = (-0.5 - origins) * invdirs
        t2 = (gsz_cu - 0.5 - origins) * invdirs
        
        ### This part compute the starting distance and ending distance for ray marching.
        ### Didn't really understand it, but, just continue pretend that I understood
        t = torch.min(t1, t2)
        t[dirs == 0] = -1e9
        t = torch.max(t, dim=-1).values.clamp_min_(self.opt.near_clip)

        tmax = torch.max(t1, t2)
        tmax[dirs == 0] = 1e9
        tmax = torch.min(tmax, dim=-1).values
        if return_raylen:
            return tmax - t
        
        log_light_intensity = torch.zeros(B, device=origins.device)
        out_rgb = torch.zeros((B,3), device=origins.device)
        good_indices = torch.arange(B, device=origins.device)

        origins_ini = origins
        dirs_ini = dirs

        mask = t <= tmax
        good_indices = good_indices[mask]
        origins = origins[mask]
        dirs = dirs[mask]

        del invdirs
        t = t[mask]
        sh_mult = sh_mult[mask]
        tmax = tmax[mask]

        N_iter = 0
        print('Ray marching forward')
        while good_indices.numel() > 0:
            print(f'Marching iter: {N_iter}')
            pos = origins + t[:, None] * dirs
            pos = pos.clamp_min_(0.0)
            pos[:, 0] = torch.clamp_max(pos[:, 0], gsz_cu[0] - 1)
            pos[:, 1] = torch.clamp_max(pos[:, 1], gsz_cu[1] - 1)
            pos[:, 2] = torch.clamp_max(pos[:, 2], gsz_cu[2] - 1)

            l = pos.to(torch.long)
            l.clamp_min_(0)
            l[:, 0] = torch.clamp_max(l[:, 0], gsz_cu[0] - 2)
            l[:, 1] = torch.clamp_max(l[:, 1], gsz_cu[1] - 2)
            l[:, 2] = torch.clamp_max(l[:, 2], gsz_cu[2] - 2)
            
            pos -= l
            wa, wb = 1.0 - pos, pos

            # BEGIN CRAZY TRILERP
            lx, ly, lz = l.unbind(-1)
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]

            sigma000, rgb000 = self._fetch_links(links000)
            sigma001, rgb001 = self._fetch_links(links001)
            sigma010, rgb010 = self._fetch_links(links010)
            sigma011, rgb011 = self._fetch_links(links011)
            sigma100, rgb100 = self._fetch_links(links100)
            sigma101, rgb101 = self._fetch_links(links101)
            sigma110, rgb110 = self._fetch_links(links110)
            sigma111, rgb111 = self._fetch_links(links111)

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
            print(f'sigma shape: {sigma.shape}, color shape: {rgb.shape}')
            log_att = (
                -self.opt.step_size 
                * torch.relu(sigma[..., 0])
                * delta_scale[good_indices]
            )
            weight = torch.exp(log_light_intensity[good_indices]) * (1.0 - torch.exp(log_att))

            # [B', 3, n_sh_coeffs]
            rgb_sh = rgb.reshape(-1, 3, self.basis_dim) 
            print(f'rgb_sh: {rgb_sh.shape}')
            print(f'sh_mult: {sh_mult.shape}')
            rgb = torch.clamp_min(
                torch.sum(sh_mult.unsqueeze(-2) * rgb_sh, dim=-1) + 0.5, 0.0
            ) # [B', 3]
            rgb = weight[:, None] * rgb[:, :3]
            out_rgb[good_indices] += rgb
            log_light_intensity[good_indices] += log_att
            t += self.opt.step_size

            mask = t <= tmax
            good_indices = good_indices[mask]
            origins = origins[mask]
            dirs = dirs[mask]

            t = t[mask]
            sh_mult = sh_mult[mask]
            tmax = tmax[mask]
            N_iter+=1
        
        # Add background color
        if self.opt.background_brightness:
            out_rgb += (torch.exp(log_light_intensity).unsqueeze(-1) * self.opt.background_brightness )
            
        return out_rgb

    def volume_render_image(self, camera: Camera, use_kernel: bool = False, randomize: bool = False, batch_size: int = 5000, return_raylen: bool = False):
        """
        Standard volume rendering (entire image version).

        :param: camera: Camera
        :param: use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :return: (H, W, 3), predicted RGB image
        """
        if use_kernel:
            raise NotImplementedError("Guess what, this is not implemented")
        else:
            # Manully generate rays for now
            rays = camera.gen_rays()
            all_rgb_out = []
            for batch_start in range(0, camera.height * camera.width, batch_size):
                rgb_out_part = self.volume_render(
                    rays[batch_start:batch_start+batch_size], use_kernel=use_kernel, randomize=randomize, return_raylen=return_raylen)
                all_rgb_out.append(rgb_out_part)
                break
            all_rgb_out = torch.cat(all_rgb_out, dim=0)
        return all_rgb_out.view(camera.height, camera.width, -1)

    def save(self, path: str, compress: bool = False):
        """
        Save to a path
        """
        save_fn = np.savez_compressed if compress else np.savez
        data = {
            "radius": self.radius.numpy(),
            "center": self.center.numpy(),
            "links": self.links.cpu().numpy(),
            "density_data": self.density_data.data.cpu().numpy(),
            "sh_data": self.sh_data.data.cpu().numpy().astype(np.float16)
        }
        data["basis_type"] = self.basis_type
        save_fn(path, **data)

    @classmethod
    def load(cls, path: str, device: Union[torch.device, str] = 'cpu'):
        """
        Load from path
        """
        z = np.load(path)
        if "data" in z.keys():
            all_data = z.f.data
            sh_data = all_data[..., 1:]
            density_data = all_data[..., :1]
        else:
            sh_data = z.f.sh_data
            density_data = z.f.density_data

        links = z.f.links
        basis_dim = sh_data.shape[1] // 3
        radius = z.f.radius.tolist() if "radius" in z.files else [
            1.0, 1.0, 1.0]
        center = z.f.center.tolist() if "center" in z.files else [
            0.0, 0.0, 0.0]
        grid = cls(
            1,
            radius=radius,
            center=center,
            basis_dim=basis_dim,
            device="cpu",
            basis_type=z.f.basis_type.item() if 'basis_type' in z.files else BASIS_TYPE_SH
        )
        if sh_data.dtype != np.float32:
            sh_data = sh_data.astype(np.float32)
        if density_data.dtype != np.float32:
            density_data = density_data.astype(np.float32)
        sh_data = torch.from_numpy(sh_data).to(device=device)
        density_data = torch.from_numpy(density_data).to(device=device)
        grid.sh_data = nn.Parameter(sh_data)
        grid.density_data = nn.Parameter(density_data)
        grid.links = torch.from_numpy(links).to(device=device)
        grid.capacity = grid.sh_data.shape[0]

        # Load basis_data (actually doesn't exist)
        grid.basis_data = nn.Parameter(grid.basis_data.data.to(device=device))
        return grid


if __name__ == "__main__":
    grid = SparseGrid([512, 512 ,512], radius=[0.5, 0.5, 0.5], center=[
                     0.5, 0.5, 0.5], device='cpu')
    # path = '../opt/ckpt/lego-debug/ckpt.npz'
    # grid = SparseGrid.load(path, device='cpu')

    print(f'grid center: {grid.center}')
    print(f'grid radius: {grid.radius}')

    print("Now test volume rendering")
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([0.5, 0.5, 0.5])
    camera = Camera(c2w, fx = 960, width=1000, height=1000)
    img = grid.volume_render_image(camera, batch_size=10)
    print(img)

