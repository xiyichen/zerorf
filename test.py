import sys

sys.path.append('.')

import os
import cv2
import torch
from sklearn.cluster import KMeans
from lib.models.autoencoders import MultiSceneNeRF
from mmgen.models import build_model
from lib.datasets.nerf_synthetic import NerfSynthetic
from opt import config_parser
from pprint import pprint
import pdb
import numpy as np

def kmeans_downsample(points, n_points_to_sample):
    kmeans = KMeans(n_points_to_sample).fit(points)
    return ((points - kmeans.cluster_centers_[..., None, :]) ** 2).sum(-1).argmin(-1).tolist()

torch.backends.cuda.matmul.allow_tf32 = True

args = config_parser()
pprint(args)

device = args.device

code_size = (3, args.model_ch, args.model_res, args.model_res)

dataset = NerfSynthetic([f"{args.data_dir}/{args.obj}/transforms_train.json"], rgba=True, file_id=args.proj_name, split='train')
test = NerfSynthetic([f"{args.data_dir}/{args.obj}/transforms_test.json"], file_id=args.proj_name, split='test')

entry = dataset[0]
selected_idxs = kmeans_downsample(entry['cond_poses'][..., :3, 3], 4)
    
data_entry = dict(
    bg_color=torch.tensor(entry['bg_color']).float().to(device),
    cond_imgs=torch.tensor(entry['cond_imgs'][selected_idxs][None]).float().to(device),
    cond_imgs_bg=torch.tensor(entry['cond_imgs_bg'][selected_idxs][None]).float().to(device),
    cond_poses=torch.tensor(entry['cond_poses'])[selected_idxs][None].float().to(device),
    cond_intrinsics=torch.tensor(entry['cond_intrinsics'])[selected_idxs][None].float().to(device),
    scene_id=[0],
    scene_name=[args.proj_name]
)

entry = test[0]
test_entry = dict(
    test_imgs=torch.tensor(entry['cond_imgs'][:][None]).float().to(device),
    test_poses=torch.tensor(entry['cond_poses'][:])[None].float().to(device),
    test_intrinsics=torch.tensor(entry['cond_intrinsics'][:])[None].float().to(device),
    scene_id=[0],
    scene_name=[args.proj_name]
)

pic_h = data_entry['cond_imgs'].shape[-3]
pic_w = data_entry['cond_imgs'].shape[-2]

decoder_1 = dict(
    type='TensorialDecoder',
    preprocessor=dict(
        type='TensorialGenerator',
        in_ch=args.model_ch, out_ch=16, noise_res=args.model_res,
        tensor_config=(
            ['xy', 'z', 'yz', 'x', 'zx', 'y']
        )
    ),
    subreduce=1 if args.load_image else 2,
    reduce='cat',
    separate_density_and_color=False,
    sh_coef_only=False,
    sdf_mode=False,
    max_steps=1024 if not args.load_image else 320,
    n_images=args.n_views,
    image_h=pic_h,
    image_w=pic_w,
    has_time_dynamics=False,
    visualize_mesh=True
)

nerf: MultiSceneNeRF = build_model(dict(
    type='MultiSceneNeRF',
    code_size=code_size,
    code_activation=dict(type='IdentityCode'),
    grid_size=64,
    patch_size=32,
    decoder=decoder_1,
    decoder_use_ema=False,
    bg_color=1.0,
    pixel_loss=dict(
        type='MSELoss',
        loss_weight=3.2
    ),
    use_lpips_metric=torch.cuda.mem_get_info()[1] // 1000 ** 3 >= 32,
    cache_size=1,
    cache_16bit=False,
    init_from_mean=True
), 
train_cfg = dict(
    dt_gamma_scale=0.5,
    density_thresh=0.05,
    extra_scene_step=0,
    n_inverse_rays=args.n_rays_init,
    n_decoder_rays=args.n_rays_init,
    loss_coef=0.1 / (pic_h * pic_w),
    optimizer=dict(type='Adam', lr=0, weight_decay=0.),
    lr_scheduler=dict(type='ExponentialLR', gamma=0.99),
    cache_load_from=None,
    viz_dir=None,
    loss_denom=1.0,
    decoder_grad_clip=1.0
),
test_cfg = dict(
    img_size=(pic_h, pic_w),
    density_thresh=0.01,
    max_render_rays=pic_h * pic_w,
    dt_gamma_scale=0.5,
    n_inverse_rays=args.n_rays_init,
    loss_coef=0.1 / (pic_h * pic_w),
    n_inverse_steps=400,
    optimizer=dict(type='Adam', lr=0.0, weight_decay=0.),
    lr_scheduler=dict(type='ExponentialLR', gamma=0.998),
    return_depth=False
))

nerf.bg_color = nerf.decoder.bg_color = torch.nn.Parameter(torch.ones(3) * args.bg_color, requires_grad=args.learn_bg)
nerf.to(device)
code_list_, code_optimizers, density_grid, density_bitfield = nerf.load_cache(data_entry, True)
# code_list_[0] = latents
nerf.save_cache(code_list_, code_optimizers, density_grid, density_bitfield, [0], [0])

torch.set_grad_enabled(False)
nerf.load_state_dict(torch.load(f"/fs/gamma-projects/3dnvs_gamma/zerorf/results_p2/{args.proj_name}/nerf-zerorf.pt"))
for i in range(100):
    nerf.decoder.update_extra_state(code_list_[0][None], density_grid, density_bitfield, 1, density_thresh=0.01, decay=0.9, S=64)
cache = nerf.cache[0]
_ = nerf.eval_and_viz(
    test_entry, nerf.decoder,
    cache['param']['code_'][None].to(device),
    density_bitfield,
    "viz_p2/test_viz",
    cfg=nerf.test_cfg,
    save=True
)
# pdb.set_trace()