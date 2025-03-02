import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import lpips
import mmcv
import trimesh

from copy import deepcopy
from glob import glob
from torch.nn.parallel.distributed import DistributedDataParallel
from mmcv.runner import load_checkpoint
from mmgen.models.builder import MODULES, build_module
from mmgen.models.architectures.common import get_module_device

from ...core import eval_psnr, eval_ssim_skimage, reduce_mean, rgetattr, rsetattr, extract_geometry, \
    module_requires_grad, get_cam_rays, get_rays, get_ray_directions
import pdb
LPIPS_BS = 4


@MODULES.register_module()
class TanhCode(nn.Module):
    def __init__(self, scale=1.0, eps=1e-5):
        super(TanhCode, self).__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, code_, update_stats=False):
        return code_.tanh() if self.scale == 1 else code_.tanh() * self.scale

    def inverse(self, code):
        return code.clamp(min=-1 + self.eps, max=1 - self.eps).atanh() if self.scale == 1 \
            else (code / self.scale).clamp(min=-1 + self.eps, max=1 - self.eps).atanh()


@MODULES.register_module()
class IdentityCode(nn.Module):
    @staticmethod
    def forward(code_, update_stats=False):
        return code_.clone()

    @staticmethod
    def inverse(code):
        return code.clone()


@MODULES.register_module()
class NormalizedTanhCode(nn.Module):
    def __init__(self, mean=0.0, std=1.0, clip_range=1, eps=1e-5, momentum=0.001):
        super(NormalizedTanhCode, self).__init__()
        self.mean = mean
        self.std = std
        self.clip_range = clip_range
        self.register_buffer('running_mean', torch.tensor([0.0]))
        self.register_buffer('running_var', torch.tensor([std ** 2]))
        self.momentum = momentum
        self.eps = eps

    def forward(self, code_, update_stats=False):
        if update_stats and self.training:
            with torch.no_grad():
                var, mean = torch.var_mean(code_)
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * reduce_mean(mean))
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * reduce_mean(var))
        scale = (self.std / (self.running_var.sqrt() + self.eps)).to(code_.device)
        return (code_ * scale + (self.mean - self.running_mean.to(code_.device) * scale)
                ).div(self.clip_range).tanh().mul(self.clip_range)

    def inverse(self, code):
        scale = ((self.running_var.sqrt() + self.eps) / self.std).to(code.device)
        return code.div(self.clip_range).clamp(min=-1 + self.eps, max=1 - self.eps).atanh().mul(
            self.clip_range * scale) + (self.running_mean.to(code.device) - self.mean * scale)


class BaseNeRF(nn.Module):
    def __init__(self,
                 code_size=(3, 8, 64, 64),
                 code_activation=dict(
                     type='TanhCode',
                     scale=1),
                 grid_size=64,
                 encoder=None,
                 decoder=dict(
                     type='TriPlaneDecoder'),
                 decoder_use_ema=False,
                 bg_color=1,
                 pixel_loss=dict(
                     type='MSELoss'),
                 patch_loss=None,
                 patch_reg_loss=None,
                 patch_size=64,
                 reg_loss=None,
                 update_extra_interval=16,
                 update_extra_iters=1,
                 use_lpips_metric=True,
                 init_from_mean=False,
                 init_scale=1e-4,
                 mean_ema_momentum=0.001,
                 mean_scale=1.0,
                 train_cfg=dict(),
                 test_cfg=dict(),
                 pretrained=None):
        super().__init__()
        self.code_size = code_size
        self.code_activation = build_module(code_activation)
        self.grid_size = grid_size
        self.encoder = build_module(encoder) if encoder is not None else None
        self.decoder = build_module(decoder)
        self.decoder_use_ema = decoder_use_ema
        if self.decoder_use_ema:
            self.decoder_ema = deepcopy(self.decoder)
        self.bg_color = bg_color
        self.pixel_loss = build_module(pixel_loss)
        self.reg_loss = build_module(reg_loss) if reg_loss is not None else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.update_extra_interval = update_extra_interval
        self.update_extra_iters = update_extra_iters
        self.lpips = [] if use_lpips_metric else None  # use a list to avoid registering the LPIPS model in state_dict

        if patch_loss is not None:
            patch_loss = deepcopy(patch_loss)
            patch_loss.update(lpips_list=self.lpips)
            self.patch_loss = build_module(patch_loss)
        else:
            self.patch_loss = None
        if patch_reg_loss is not None:
            self.patch_reg_loss = build_module(patch_reg_loss)
        else:
            self.patch_reg_loss = None
        self.patch_size = patch_size

        if init_from_mean:
            self.register_buffer('init_code', torch.zeros(code_size))
        else:
            self.init_code = None
        self.init_scale = init_scale
        self.mean_ema_momentum = mean_ema_momentum
        self.mean_scale = mean_scale
        if pretrained is not None and os.path.isfile(pretrained):
            load_checkpoint(self, pretrained, map_location='cpu')

        self.train_cfg_backup = dict()
        for key, value in self.test_cfg.get('override_cfg', dict()).items():
            self.train_cfg_backup[key] = rgetattr(self, key, None)

    def train(self, mode=True):
        if mode:
            for key, value in self.train_cfg_backup.items():
                rsetattr(self, key, value)
        else:
            for key, value in self.test_cfg.get('override_cfg', dict()).items():
                if self.training:
                    self.train_cfg_backup[key] = rgetattr(self, key)
                rsetattr(self, key, value)
        super().train(mode)
        return self

    def load_scene(self, data, load_density=False):
        device = get_module_device(self)
        code_list = []
        density_grid = []
        density_bitfield = []
        for code_state_single in data['code']:
            code_list.append(
                code_state_single['param']['code'] if 'code' in code_state_single['param']
                else self.code_activation(code_state_single['param']['code_']))
            if load_density:
                density_grid.append(code_state_single['param']['density_grid'])
                density_bitfield.append(code_state_single['param']['density_bitfield'])
        code = torch.stack(code_list, dim=0).to(device)
        density_grid = torch.stack(density_grid, dim=0).to(device) if load_density else None
        density_bitfield = torch.stack(density_bitfield, dim=0).to(device) if load_density else None
        return code, density_grid, density_bitfield

    @staticmethod
    def save_scene(save_dir, code, density_grid, density_bitfield, scene_name):
        os.makedirs(save_dir, exist_ok=True)
        for scene_id, scene_name_single in enumerate(scene_name):
            results = dict(
                scene_name=scene_name_single,
                param=dict(
                    code=code.data[scene_id].cpu(),
                    density_grid=density_grid.data[scene_id].cpu(),
                    density_bitfield=density_bitfield.data[scene_id].cpu()))
            torch.save(results, os.path.join(save_dir, scene_name_single) + '.pth')

    @staticmethod
    def save_mesh(save_dir, decoder, code, scene_name, mesh_resolution, mesh_threshold):
        os.makedirs(save_dir, exist_ok=True)
        for code_single, scene_name_single in zip(code, scene_name):
            vertices, triangles = extract_geometry(
                decoder,
                code_single,
                mesh_resolution,
                mesh_threshold)
            mesh = trimesh.Trimesh(vertices, triangles, process=False)
            mesh.export(os.path.join(save_dir, scene_name_single) + '.stl')

    def get_init_code_(self, num_scenes, device=None):
        code_ = torch.empty(
            self.code_size if num_scenes is None else (num_scenes, *self.code_size),
            device=device, requires_grad=True, dtype=torch.float32)
        if self.init_code is None:
            code_.data.uniform_(-self.init_scale, self.init_scale)
        else:
            code_.data[:] = self.code_activation.inverse(self.init_code * self.mean_scale)
        return code_

    def get_init_density_grid(self, num_scenes, device=None):
        return torch.zeros(
            self.grid_size ** 3 if num_scenes is None else (num_scenes, self.grid_size ** 3),
            device=device, dtype=torch.float16)

    def get_init_density_bitfield(self, num_scenes, device=None):
        return torch.zeros(
            self.grid_size ** 3 // 8 if num_scenes is None else (num_scenes, self.grid_size ** 3 // 8),
            device=device, dtype=torch.uint8)

    @staticmethod
    def build_optimizer(code_, cfg):
        optimizer_cfg = cfg['optimizer'].copy()
        optimizer_class = getattr(torch.optim, optimizer_cfg.pop('type'))
        if isinstance(code_, list):
            code_optimizer = [
                optimizer_class([code_single_], **optimizer_cfg)
                for code_single_ in code_]
        else:
            code_optimizer = optimizer_class([code_], **optimizer_cfg)
        return code_optimizer

    @staticmethod
    def build_scheduler(code_optimizer, cfg):
        if 'lr_scheduler' in cfg:
            scheduler_cfg = cfg['lr_scheduler'].copy()
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_cfg.pop('type'))
            if isinstance(code_optimizer, list):
                code_scheduler = [
                    scheduler_class(code_optimizer_single, **scheduler_cfg)
                    for code_optimizer_single in code_optimizer]
            else:
                code_scheduler = scheduler_class(code_optimizer, **scheduler_cfg)
        else:
            code_scheduler = None
        return code_scheduler

    def ray_sample(self, cond_rays_o, cond_rays_d, cond_imgs, cond_imgs_bg, n_samples, cond_times=None, sample_inds=None):
        """
        Args:
            cond_rays_o (torch.Tensor): (num_scenes, num_imgs, h, w, 3)
            cond_rays_d (torch.Tensor): (num_scenes, num_imgs, h, w, 3)
            cond_imgs (torch.Tensor): (num_scenes, num_imgs, h, w, 3)
            n_samples (int): number of samples
            sample_inds (None | torch.Tensor): (num_scenes, n_samples) or (num_scenes, n_patches)

        Returns:
            rays_o (torch.Tensor): (num_scenes, n_samples, 3)
            rays_d (torch.Tensor): (num_scenes, n_samples, 3)
            target_rgbs (torch.Tensor): (num_scenes, n_samples, 3) or (-1, ps, ps, 3)
        """
        device = cond_rays_o.device
        num_scenes, num_imgs, h, w, _ = cond_rays_o.size()
        num_scene_pixels = num_imgs * h * w
        rays_time = None
        if self.patch_loss is None and self.patch_reg_loss is None:
            rays_o = cond_rays_o.reshape(num_scenes, num_scene_pixels, 3)
            rays_d = cond_rays_d.reshape(num_scenes, num_scene_pixels, 3)
            if cond_times is not None:
                assert list(cond_times.shape) == [num_scenes, num_imgs]
                rays_time = cond_times[..., None, None].repeat(1, 1, h, w).reshape(num_scenes, num_scene_pixels, 1)
            target_rgbs = cond_imgs.reshape(num_scenes, num_scene_pixels, cond_imgs.shape[-1])
            target_rgbs_bg = cond_imgs_bg.reshape(num_scenes, num_scene_pixels, cond_imgs_bg.shape[-1])
        else:
            assert n_samples % (self.patch_size ** 2) == 0
            assert cond_times is None, "dynamic with patch loss is unsupported"
            rays_o = cond_rays_o.reshape(
                num_scenes, -1, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size, 3
            ).permute(0, 1, 2, 4, 3, 5, 6).reshape(num_scenes, -1, self.patch_size, self.patch_size, 3)
            rays_d = cond_rays_d.reshape(
                num_scenes, -1, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size, 3
            ).permute(0, 1, 2, 4, 3, 5, 6).reshape(num_scenes, -1, self.patch_size, self.patch_size, 3)
            target_rgbs = cond_imgs.reshape(
                num_scenes, -1, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size, cond_imgs.shape[-1]
            ).permute(0, 1, 2, 4, 3, 5, 6).reshape(num_scenes, -1, self.patch_size, self.patch_size, cond_imgs.shape[-1])
            target_rgbs_bg = cond_imgs.reshape(
                num_scenes, -1, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size, cond_imgs_bg.shape[-1]
            ).permute(0, 1, 2, 4, 3, 5, 6).reshape(num_scenes, -1, self.patch_size, self.patch_size, cond_imgs_bg.shape[-1])
        if num_scene_pixels > n_samples:
            if sample_inds is None:
                sample_inds = [
                    torch.randperm(
                        target_rgbs.size(1), device=device
                    )[:n_samples if self.patch_loss is None and self.patch_reg_loss is None else n_samples // (self.patch_size ** 2)]
                    for _ in range(num_scenes)]
                sample_inds = torch.stack(sample_inds, dim=0)
            scene_arange = torch.arange(num_scenes, device=device)[:, None]
            rays_o = rays_o[scene_arange, sample_inds]
            rays_d = rays_d[scene_arange, sample_inds]
            target_rgbs = target_rgbs[scene_arange, sample_inds]
            target_rgbs_bg = target_rgbs_bg[scene_arange, sample_inds]
            if rays_time is not None:
                rays_time = rays_time[scene_arange, sample_inds]
        if self.patch_loss is not None or self.patch_reg_loss is not None:
            rays_o = rays_o.reshape(num_scenes, -1, 3)
            rays_d = rays_d.reshape(num_scenes, -1, 3)
            target_rgbs = target_rgbs.reshape(-1, self.patch_size, self.patch_size, cond_imgs.shape[-1])
            target_rgbs_bg = target_rgbs_bg.reshape(-1, self.patch_size, self.patch_size, cond_imgs.shape[-1])
        return rays_o, rays_d, target_rgbs, target_rgbs_bg, sample_inds, rays_time

    def get_raybatch_inds(self, cond_imgs, n_inverse_rays):
        device = cond_imgs.device
        num_scenes, num_imgs, h, w, _ = cond_imgs.size()
        num_scene_pixels = num_imgs * h * w
        if num_scene_pixels > n_inverse_rays:
            if self.patch_loss is None and self.patch_reg_loss is None:
                raybatch_inds = [torch.randperm(num_scene_pixels, device=device) for _ in range(num_scenes)]
                raybatch_inds = torch.stack(raybatch_inds, dim=0).split(n_inverse_rays, dim=1)
                num_raybatch = len(raybatch_inds)
            else:
                raybatch_inds = [torch.randperm(
                    num_scene_pixels // (self.patch_size ** 2), device=device) for _ in range(num_scenes)]
                raybatch_inds = torch.stack(raybatch_inds, dim=0).split(
                    n_inverse_rays // (self.patch_size ** 2), dim=1)
                num_raybatch = len(raybatch_inds)
        else:
            raybatch_inds = num_raybatch = None
        return raybatch_inds, num_raybatch

    def loss(self, decoder, code, density_bitfield, target_rgbs, target_rgbs_bg, sample_inds,
             rays_o, rays_d, dt_gamma=0.0, return_decoder_loss=False, scale_num_ray=1.0,
             rays_time=None, cfg=dict(), **kwargs):
        outputs = decoder(
            rays_o, rays_d, code, density_bitfield, self.grid_size, sample_inds=sample_inds, rays_time=rays_time,
            dt_gamma=dt_gamma, perturb=True, return_loss=return_decoder_loss, **kwargs)
        out_weights = outputs['weights_sum']
        # if self.training:
        #     out_rgbs = outputs['image'] + target_rgbs_bg * (1 - out_weights.unsqueeze(-1))
        # else:
        out_rgbs = outputs['image'] + outputs.get('bg', self.bg_color) * (1 - out_weights.unsqueeze(-1))
        scale = 1 - math.exp(-cfg['loss_coef'] * scale_num_ray) if 'loss_coef' in cfg else 1
        if target_rgbs.shape[-1] == 4:
            # if self.training:
            #     pass
            # else:
            target_rgbs[..., :3] = target_rgbs[..., :3] * target_rgbs[..., 3:] + self.bg_color * (1 - target_rgbs[..., 3:])
            out_rgbs = torch.cat([out_rgbs, out_weights.unsqueeze(-1)], dim=-1)
        pixel_loss = self.pixel_loss(
            out_rgbs, target_rgbs.reshape(out_rgbs.size())) * (scale * 3)
        loss = pixel_loss
        loss_dict = dict(pixel_loss=pixel_loss)
        if self.patch_loss is not None:
            assert target_rgbs.dim() == 4  # (num_patches, h, w, 3)
            patch_loss = self.patch_loss(
                out_rgbs.reshape(target_rgbs.size()).permute(0, 3, 1, 2),  # (num_patches, 3, h, w)
                target_rgbs.permute(0, 3, 1, 2),  # (num_patches, 3, h, w)
            ) * scale
            loss = loss + patch_loss
            loss_dict.update(patch_loss=patch_loss)
        if self.patch_reg_loss is not None:
            assert target_rgbs.dim() == 4  # (num_patches, h, w, 3)
            patch_reg_loss = self.patch_reg_loss(
                outputs['weights_sum'].reshape(*target_rgbs.shape[:-1], 1).permute(0, 3, 1, 2),  # (num_patches, 1, h, w)
                outputs['depth'].reshape(*target_rgbs.shape[:-1], 1).permute(0, 3, 1, 2)
            ) * scale
            loss = loss + patch_reg_loss
            loss_dict.update(patch_reg_loss=patch_reg_loss)
        if self.reg_loss is not None:
            reg_loss = self.reg_loss(code)
            loss = loss + reg_loss
            loss_dict.update(reg_loss=reg_loss)
        if return_decoder_loss and outputs['decoder_reg_loss'] is not None:
            decoder_reg_loss = outputs['decoder_reg_loss']
            loss = loss + decoder_reg_loss
            loss_dict.update(decoder_reg_loss=decoder_reg_loss)
        return (out_rgbs, target_rgbs), loss, loss_dict

    def loss_decoder(self, decoder, code, density_bitfield, cond_rays_o, cond_rays_d,
                     cond_imgs, cond_imgs_bg, cond_times=None, dt_gamma=0.0, cfg=dict(), **kwargs):
        decoder_training_prev = decoder.training
        decoder.train(True)
        n_decoder_rays = cfg.get('n_decoder_rays', 4096)
        rays_o, rays_d, target_rgbs, target_rgbs_bg, sample_inds, rays_time = self.ray_sample(
            cond_rays_o, cond_rays_d, cond_imgs, cond_imgs_bg, cond_times=cond_times, n_samples=n_decoder_rays)
        (out_rgbs, target_rgbs), loss, loss_dict = self.loss(
            decoder, code, density_bitfield, target_rgbs, target_rgbs_bg, sample_inds,
            rays_o, rays_d, dt_gamma, return_decoder_loss=True, scale_num_ray=cond_rays_o.shape[1:4].numel(),
            rays_time=rays_time,
            cfg=cfg, **kwargs)
        log_vars = dict()
        for key, val in loss_dict.items():
            log_vars.update({key: float(val)})

        decoder.train(decoder_training_prev)

        return loss, log_vars, out_rgbs, target_rgbs.reshape(out_rgbs.size())

    def get_density(self, decoder, code, cfg=dict()):
        if isinstance(decoder, DistributedDataParallel):
            decoder = decoder.module
        density_thresh = cfg.get('density_thresh', 0.01)
        density_step = cfg.get('density_step', 8)
        num_scenes = code.size(0)
        device = code.device
        density_grid = self.get_init_density_grid(num_scenes, device)
        density_bitfield = self.get_init_density_bitfield(num_scenes, device)
        with torch.no_grad():
            for i in range(density_step):
                decoder.update_extra_state(
                    code, density_grid, density_bitfield, i, density_thresh=density_thresh, decay=1.0)
        return density_grid, density_bitfield

    def inverse_code(self, decoder, cond_imgs, cond_rays_o, cond_rays_d, dt_gamma=0, cfg=dict(),
                     code_=None, density_grid=None, density_bitfield=None, iter_density=None,
                     code_optimizer=None, code_scheduler=None,
                     prior_grad=None, show_pbar=False):
        """
        Obtain scene codes via optimization-based inverse rendering.
        """
        device = get_module_device(self)
        decoder_training_prev = decoder.training
        decoder.train(True)

        with module_requires_grad(decoder, False):
            n_inverse_steps = cfg.get('n_inverse_steps', 1000)
            n_inverse_rays = cfg.get('n_inverse_rays', 4096)

            num_scenes, num_imgs, h, w, _ = cond_imgs.size()
            num_scene_pixels = num_imgs * h * w
            raybatch_inds, num_raybatch = self.get_raybatch_inds(cond_imgs, n_inverse_rays)

            if code_ is None:
                code_ = self.get_init_code_(num_scenes, device=device)
            if density_grid is None:
                density_grid = self.get_init_density_grid(num_scenes, device)
            if density_bitfield is None:
                density_bitfield = self.get_init_density_bitfield(num_scenes, device)
            if iter_density is None:
                iter_density = 0

            if code_optimizer is None:
                assert code_scheduler is None
                code_optimizer = self.build_optimizer(code_, cfg)
            if code_scheduler is None:
                code_scheduler = self.build_scheduler(code_optimizer, cfg)

            assert n_inverse_steps > 0
            if show_pbar:
                pbar = mmcv.ProgressBar(n_inverse_steps)

            for inverse_step_id in range(n_inverse_steps):
                code = self.code_activation(
                    torch.stack(code_, dim=0) if isinstance(code_, list)
                    else code_)

                if inverse_step_id % self.update_extra_interval == 0:
                    update_extra_state = self.update_extra_iters
                    extra_args = (density_grid, density_bitfield, iter_density)
                    extra_kwargs = dict(
                        density_thresh=self.train_cfg['density_thresh']
                    ) if 'density_thresh' in self.train_cfg else dict()
                else:
                    update_extra_state = 0
                    extra_args = extra_kwargs = None

                inds = raybatch_inds[inverse_step_id % num_raybatch] if raybatch_inds is not None else None
                rays_o, rays_d, target_rgbs, sample_inds, rays_time = self.ray_sample(
                    cond_rays_o, cond_rays_d, cond_imgs, n_inverse_rays, sample_inds=inds)
                (out_rgbs, target_rgbs), loss, loss_dict = self.loss(
                    decoder, code, density_bitfield,
                    target_rgbs, sample_inds, rays_o, rays_d, dt_gamma, scale_num_ray=num_scene_pixels, cfg=cfg,
                    update_extra_state=update_extra_state, extra_args=extra_args, extra_kwargs=extra_kwargs)

                if prior_grad is not None:
                    if isinstance(code_, list):
                        for code_single_, prior_grad_single in zip(code_, prior_grad):
                            code_single_.grad.copy_(prior_grad_single)
                    else:
                        code_.grad.copy_(prior_grad)
                else:
                    if isinstance(code_optimizer, list):
                        for code_optimizer_single in code_optimizer:
                            code_optimizer_single.zero_grad()
                    else:
                        code_optimizer.zero_grad()

                loss.backward()

                if isinstance(code_optimizer, list):
                    for code_optimizer_single in code_optimizer:
                        code_optimizer_single.step()
                else:
                    code_optimizer.step()

                if code_scheduler is not None:
                    if isinstance(code_scheduler, list):
                        for code_scheduler_single in code_scheduler:
                            code_scheduler_single.step()
                    else:
                        code_scheduler.step()

                if show_pbar:
                    pbar.update()

        decoder.train(decoder_training_prev)

        return code.detach(), density_grid, density_bitfield, \
               loss, loss_dict, out_rgbs, target_rgbs.reshape(out_rgbs.size())

    def render(self, decoder, code, density_bitfield, h, w, intrinsics, poses, times=None, cfg=dict()):
        code = code.to(next(decoder.parameters()).dtype)

        decoder_training_prev = decoder.training
        decoder.train(False)

        dt_gamma_scale = cfg.get('dt_gamma_scale', 0.0)
        # (num_scenes,)
        dt_gamma = dt_gamma_scale * 2 / (intrinsics[..., 0] + intrinsics[..., 1]).mean(dim=-1)
        directions = get_ray_directions(
            h, w, intrinsics, norm=False, device=intrinsics.device)  # (num_scenes, num_imgs, h, w, 3)
        rays_o, rays_d = get_rays(directions, poses, norm=True)
        num_scenes, num_imgs, h, w, _ = rays_o.size()
        
        assert times is None or list(times.shape) == [num_scenes, num_imgs]
        if times is not None:
            rays_time = times[..., None, None].repeat(1, 1, h, w).reshape(num_scenes, num_imgs * h * w, 1)
        else:
            rays_time = None

        rays_o = rays_o.reshape(num_scenes, num_imgs * h * w, 3)
        rays_d = rays_d.reshape(num_scenes, num_imgs * h * w, 3)
        max_render_rays = cfg.get('max_render_rays', -1)
        if 0 < max_render_rays < rays_o.size(1):
            rays_o = rays_o.split(max_render_rays, dim=1)
            rays_d = rays_d.split(max_render_rays, dim=1)
            if rays_time is not None:
                rays_time = rays_time.split(max_render_rays, dim=1)
        else:
            rays_o = [rays_o]
            rays_d = [rays_d]
            if rays_time is not None:
                rays_time = [rays_time]

        out_image = []
        out_normal = []
        out_depth = []
        if rays_time is None:
            rays_time = len(rays_o) * [None]
        for rays_o_single, rays_d_single, rays_time_single in zip(rays_o, rays_d, rays_time):
            outputs = decoder(
                rays_o_single, rays_d_single,
                code, density_bitfield, self.grid_size,
                rays_time=rays_time_single,
                dt_gamma=dt_gamma, perturb=False)
            weights_sum = torch.stack(outputs['weights_sum'], dim=0) if num_scenes > 1 else outputs['weights_sum'][0]
            return_rgba = cfg.get('return_rgba', False)
            if not return_rgba:
                rgbs = (torch.stack(outputs['image'], dim=0) if num_scenes > 1 else outputs['image'][0]) \
                    + outputs.get('bg', self.bg_color) * (1 - weights_sum.unsqueeze(-1))
                out_image.append(rgbs)
                # normals = ((torch.stack(outputs['normal'], dim=0) if num_scenes > 1 else outputs['normal'][0]) + 1) / 2 \
                #     + outputs.get('bg', self.bg_color) * (1 - weights_sum.unsqueeze(-1))
                # out_normal.append(normals)
            else:
                rgbs = (torch.stack(outputs['image'], dim=0) if num_scenes > 1 else outputs['image'][0])
                rgbas = torch.cat([rgbs, weights_sum.unsqueeze(-1)], dim=-1)
                out_image.append(rgbas)
            depth = torch.stack(outputs['depth'], dim=0) if num_scenes > 1 else outputs['depth'][0]
            out_depth.append(depth)
        out_image = torch.cat(out_image, dim=-2) if len(out_image) > 1 else out_image[0]
        out_image = out_image.reshape(num_scenes, num_imgs, h, w, 4 if return_rgba else 3)
        # out_normal = torch.cat(out_normal, dim=-2) if len(out_normal) > 1 else out_normal[0]
        # out_normal = out_normal.reshape(num_scenes, num_imgs, h, w, 3)
        out_depth = torch.cat(out_depth, dim=-1) if len(out_depth) > 1 else out_depth[0]
        out_depth = out_depth.reshape(num_scenes, num_imgs, h, w)
        
        if cfg.get('inverse_z_depth', True):
            out_depth = out_depth * torch.linalg.norm(directions, dim=-1)

        decoder.train(decoder_training_prev)
        return out_image, out_normal, out_depth

    def eval_and_viz(self, data, decoder, code, density_bitfield, viz_dir=None, cfg=dict(), save=True):
        scene_name = data['scene_name']  # (num_scenes,)
        test_intrinsics = data['test_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
        test_poses = data['test_poses']
        test_times = data.get('test_times', None)
        num_scenes, num_imgs, _, _ = test_poses.size()
        
        if 'test_imgs' in data and not cfg.get('skip_eval', False):
            test_imgs = data['test_imgs']  # (num_scenes, num_imgs, h, w, 3)
            _, _, h, w, _ = test_imgs.size()
            test_img_paths = data.get('test_img_paths', None)  # (num_scenes, (num_imgs,))
            target_imgs = test_imgs.permute(0, 1, 4, 2, 3).reshape(num_scenes * num_imgs, 3, h, w)
        else:
            test_imgs = test_img_paths = target_imgs = None
            h, w = cfg['img_size']
        image, normal, depth = self.render(decoder, code, density_bitfield, h, w, test_intrinsics, test_poses, times=test_times, cfg=cfg)
        
        code_range = cfg.get('clip_range', [-1, 1])
        pred_imgs = image.permute(0, 1, 4, 2, 3).reshape(
            num_scenes * num_imgs, 3, h, w).clamp(min=0, max=1)
        pred_imgs = torch.round(pred_imgs * 255) / 255
        pred_depths = torch.round(depth * 255) / 255
        
        # pred_normals = normal.permute(0, 1, 4, 2, 3).reshape(
        #     num_scenes * num_imgs, 3, h, w).clamp(min=0, max=1)
        # pred_normals = torch.round(pred_normals * 255) / 255
        # pdb.set_trace()
        log_vars = None
        if save and not os.path.isfile(os.path.join(viz_dir, scene_name[0] + '.glb')):
            if test_imgs is not None:
                test_psnr = eval_psnr(pred_imgs, target_imgs)
                test_ssim = eval_ssim_skimage(pred_imgs, target_imgs, data_range=1)
                log_vars = dict(test_psnr=float(test_psnr.mean()),
                                test_ssim=float(test_ssim.mean()))
                if self.lpips is not None:
                    if len(self.lpips) == 0:
                        lpips_eval = lpips.LPIPS(
                            net='vgg', eval_mode=True, pnet_tune=False).to(
                            device=pred_imgs.device, dtype=torch.bfloat16)
                        self.lpips.append(lpips_eval)
                    test_lpips = []
                    for pred_imgs_batch, target_imgs_batch in zip(
                            pred_imgs.split(LPIPS_BS, dim=0), target_imgs.split(LPIPS_BS, dim=0)):
                        test_lpips.append(self.lpips[0](
                            (pred_imgs_batch * 2 - 1).to(torch.bfloat16),
                            (target_imgs_batch * 2 - 1).to(torch.bfloat16)).flatten())
                    test_lpips = torch.cat(test_lpips, dim=0)
                    log_vars.update(test_lpips=float(test_lpips.mean()))
                else:
                    test_lpips = [math.nan for _ in range(num_scenes * num_imgs)]
            else:
                log_vars = dict()

            if viz_dir is None:
                viz_dir = cfg.get('viz_dir', None)
            if viz_dir is not None:
                os.makedirs(viz_dir, exist_ok=True)
                output_viz = torch.round(pred_imgs.permute(0, 2, 3, 1) * 255).to(
                    torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 3)
                if test_imgs is not None:
                    real_imgs_viz = (target_imgs.permute(0, 2, 3, 1) * 255).to(
                        torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 3)
                    output_viz = np.concatenate([real_imgs_viz, output_viz], axis=-2)
                for scene_id, scene_name_single in enumerate(scene_name):
                    for img_id in range(num_imgs):
                        if test_img_paths is not None:
                            base_name = 'scene_' + scene_name_single + '_' + os.path.splitext(
                                os.path.basename(test_img_paths[scene_id][img_id]))[0]
                            name = base_name + '_psnr{:02.1f}_ssim{:.2f}_lpips{:.3f}.png'.format(
                                test_psnr[scene_id * num_imgs + img_id],
                                test_ssim[scene_id * num_imgs + img_id],
                                test_lpips[scene_id * num_imgs + img_id])
                            existing_files = glob(os.path.join(viz_dir, base_name + '*.png'))
                            for file in existing_files:
                                os.remove(file)
                        else:
                            name = 'scene_' + scene_name_single + '_{:03d}.png'.format(img_id)
                            if test_imgs is not None:
                                name = name + '_psnr{:02.1f}_ssim{:.2f}_lpips{:.3f}.png'.format(
                                    test_psnr[scene_id * num_imgs + img_id],
                                    test_ssim[scene_id * num_imgs + img_id],
                                    test_lpips[scene_id * num_imgs + img_id])
                        plt.imsave(
                            os.path.join(viz_dir, name),
                            output_viz[scene_id][img_id])
                        plt.imsave(
                            os.path.join(viz_dir, "depth_" + name),
                            depth[scene_id][img_id].cpu().numpy())
                if isinstance(decoder, DistributedDataParallel):
                    decoder = decoder.module
                decoder.visualize(code, scene_name, viz_dir, code_range=code_range)
                if self.init_code is not None:
                    decoder.visualize(self.init_code[None], ['000_mean'], viz_dir, code_range=code_range)

        return log_vars, pred_imgs.reshape(num_scenes, num_imgs, 3, h, w), pred_depths.reshape(num_scenes, num_imgs, h, w)

    def mean_ema_update(self, code):
        if self.init_code is None:
            return
        mean_code = reduce_mean(code.detach().mean(dim=0))
        self.init_code.mul_(1 - self.mean_ema_momentum).add_(
            mean_code.data, alpha=self.mean_ema_momentum)

    def train_step(self, data, optimizer, running_status=None):
        raise NotImplementedError

    def val_step(self, data, viz_dir=None, show_pbar=False, **kwargs):
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder

        if 'code' in data:
            code, density_grid, density_bitfield = self.load_scene(
                data, load_density=True)
            out_rgbs = target_rgbs = None
        else:
            cond_imgs = data['cond_imgs']  # (num_scenes, num_imgs, h, w, 3)
            cond_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            cond_poses = data['cond_poses']

            num_scenes, num_imgs, h, w, _ = cond_imgs.size()
            # (num_scenes, num_imgs, h, w, 3)
            cond_rays_o, cond_rays_d = get_cam_rays(cond_poses, cond_intrinsics, h, w)
            dt_gamma_scale = self.test_cfg.get('dt_gamma_scale', 0.0)
            # (num_scenes,)
            dt_gamma = dt_gamma_scale / cond_intrinsics[..., :2].mean(dim=(-2, -1))

            with torch.enable_grad():
                (code, density_grid, density_bitfield,
                 loss, loss_dict, out_rgbs, target_rgbs) = self.inverse_code(
                    decoder, cond_imgs, cond_rays_o, cond_rays_d,
                    dt_gamma=dt_gamma, cfg=self.test_cfg, show_pbar=show_pbar)

        # ==== evaluate reconstruction ====
        with torch.no_grad():
            if 'test_poses' in data:
                log_vars, pred_imgs = self.eval_and_viz(
                    data, decoder, code, density_bitfield,
                    viz_dir=viz_dir, cfg=self.test_cfg)
            else:
                log_vars = dict()
                pred_imgs = None
            if out_rgbs is not None and target_rgbs is not None:
                train_psnr = eval_psnr(out_rgbs, target_rgbs)
                log_vars.update(train_psnr=float(train_psnr.mean()))
            code_rms = code.square().flatten(1).mean().sqrt()
            log_vars.update(code_rms=float(code_rms.mean()))

        # ==== save 3D code ====
        save_dir = self.test_cfg.get('save_dir', None)
        if save_dir is not None:
            self.save_scene(save_dir, code, density_grid, density_bitfield, data['scene_name'])

        # ==== outputs ====
        outputs_dict = dict(
            log_vars=log_vars,
            num_samples=len(data['scene_name']),
            pred_imgs=pred_imgs)

        return outputs_dict
