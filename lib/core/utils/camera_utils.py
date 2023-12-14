import numpy as np
import torch
import torch.nn.functional as F


def look_at(center, target, up):
    f = F.normalize(target - center, dim=-1)
    s = F.normalize(torch.cross(f, up, dim=-1), dim=-1)
    u = F.normalize(torch.cross(s, f, dim=-1), dim=-1)
    m = torch.stack([s, -u, f], dim=-1)
    return m


def surround_views(initial_pose, angle_amp=1.0, num_frames=60):
    rad = torch.from_numpy(
        np.linspace(0, 2 * np.pi, num=num_frames, endpoint=False)).to(initial_pose)

    initial_pos = initial_pose[:3, -1]
    initial_pos_dist = torch.linalg.norm(initial_pos)
    initial_pos_norm = initial_pos / initial_pos_dist
    initial_angle = torch.asin(initial_pos_norm[-1])

    angles = initial_angle * (rad.sin() * angle_amp + 1)
    pos_xy = F.normalize(initial_pos_norm[:2], dim=0) @ torch.stack(
        [rad.cos(), -rad.sin(),
         rad.sin(), rad.cos()], dim=-1).reshape(-1, 2, 2)
    pos = torch.cat(
        [pos_xy * angles.cos().unsqueeze(-1), angles.sin().unsqueeze(-1)],
        dim=-1) * initial_pos_dist
    rot = look_at(pos, torch.zeros_like(pos), pos.new_tensor([0, 0, 1]).expand(pos.size()))
    poses = torch.cat(
        [torch.cat([rot, pos.unsqueeze(-1)], dim=-1),
         rot.new_tensor([0, 0, 0, 1]).expand(num_frames, 1, -1)], dim=-2)

    return poses


def random_surround_views(camera_distance, num_cameras, min_angle=0.1, max_angle=0.4, use_linspace=False, begin_rad=0):
    if use_linspace:
        rad = torch.from_numpy(
            np.linspace(0 + np.pi / num_cameras, 2 * np.pi - np.pi / num_cameras, num=num_cameras, dtype=np.float32))
    else:
        rad = torch.rand(num_cameras) * (2 * np.pi)
    rad += begin_rad - rad[0]
    angles = torch.rand(num_cameras) * (max_angle - min_angle) + min_angle
    pos_xy = torch.stack([rad.cos(), rad.sin()], dim=-1)
    pos = torch.cat([pos_xy * angles.cos().unsqueeze(-1), angles.sin().unsqueeze(-1)], dim=-1) * camera_distance
    rot = look_at(pos, torch.zeros_like(pos), pos.new_tensor([0, 0, 1]).expand(pos.size()))
    poses = torch.cat(
        [torch.cat([rot, pos.unsqueeze(-1)], dim=-1),
         rot.new_tensor([0, 0, 0, 1]).expand(num_cameras, 1, -1)], dim=-2)
    return poses
