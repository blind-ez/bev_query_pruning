import numpy as np
import torch


def normalize_bbox(bboxes):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]

    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    rot_sine = rot.sin()
    rot_cosine = rot.cos()

    if bboxes.size(-1) == 9:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        return torch.cat((cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy), dim=-1)
    else:
        return torch.cat((cx, cy, w, l, cz, h, rot_sine, rot_cosine), dim=-1)


def denormalize_bbox(bboxes):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 4:5]

    w = bboxes[..., 2:3].exp()
    l = bboxes[..., 3:4].exp()
    h = bboxes[..., 5:6].exp()

    rot_sine = bboxes[..., 6:7]
    rot_cosine = bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    if bboxes.size(-1) == 10:
        vx = bboxes[..., 8:9]
        vy = bboxes[..., 9:10]
        return torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        return torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)


def normalize_coords(coords, pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
    out_coords = torch.zeros_like(coords)

    out_coords[..., 0:1] = (coords[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
    out_coords[..., 1:2] = (coords[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
    if coords.shape[-1] == 3:
        out_coords[..., 2:3] = (coords[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

    return out_coords


def denormalize_coords(coords, pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
    out_coords = torch.zeros_like(coords)

    out_coords[..., 0:1] = coords[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    out_coords[..., 1:2] = coords[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    if coords.shape[-1] == 3:
        out_coords[..., 2:3] = coords[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

    return out_coords


def generate_2d_grid(H, W):
    xs = torch.linspace(0.5, W-0.5, W).reshape(1, W).expand(H, W) / W
    ys = torch.linspace(0.5, H-0.5, H).reshape(H, 1).expand(H, W) / H

    coords_2d = torch.stack([xs, ys], dim=-1)  # (H, W, 2)
    coords_2d = coords_2d.flatten(0, 1)        # (H*W, 2)

    return coords_2d


def generate_3d_grid(D, H, W):
    xs = torch.linspace(0.5, W-0.5, W).reshape(1, 1, W).expand(D, H, W) / W
    ys = torch.linspace(0.5, H-0.5, H).reshape(1, H, 1).expand(D, H, W) / H
    zs = torch.linspace(0.5, D-0.5, D).reshape(D, 1, 1).expand(D, H, W) / D

    coords_3d = torch.stack([xs, ys, zs], dim=-1)  # (D, H, W, 3)
    coords_3d = coords_3d.flatten(0, 2)            # (D*H*W, 3)

    return coords_3d


# def transform_coords_between_frames(coords_3d, source_img_metas, target_img_metas):
#     source_canbus = coords_3d.new_tensor([each['can_bus'] for each in source_img_metas])
#     target_canbus = coords_3d.new_tensor([each['can_bus'] for each in target_img_metas])

#     source_canbus = source_canbus.reshape(*coords_3d.shape[:-2], -1)
#     target_canbus = target_canbus.reshape(*coords_3d.shape[:-2], -1)

#     ego_delta_yaw = target_canbus[..., -2] - source_canbus[..., -2]
#     cos_yaw = torch.cos(ego_delta_yaw)
#     sin_yaw = torch.sin(ego_delta_yaw)
#     rotation_matrix = torch.stack([
#         torch.stack([cos_yaw, -sin_yaw], dim=-1),
#         torch.stack([sin_yaw,  cos_yaw], dim=-1)
#     ], dim=-2)
#     rotation_matrix = torch.stack([torch.stack([cos_yaw, -sin_yaw], dim=-1), torch.stack([sin_yaw,  cos_yaw], dim=-1)], dim=-2)

#     ego_delta_x = target_canbus[..., 0] - source_canbus[..., 0]
#     ego_delta_y = target_canbus[..., 1] - source_canbus[..., 1]
#     ego_delta_z = target_canbus[..., 2] - source_canbus[..., 2]

#     ego_translation_distance = torch.sqrt(ego_delta_x ** 2 + ego_delta_y ** 2) + 1e-6
#     ego_translation_angle = torch.atan2(ego_delta_y, ego_delta_x)
#     ego_heading = target_canbus[..., -2]
#     bev_heading = ego_heading - ego_translation_angle
#     shift_x = ego_translation_distance * torch.sin(bev_heading)
#     shift_y = ego_translation_distance * torch.cos(bev_heading)
#     shift = torch.stack([shift_x, shift_y, ego_delta_z], dim=-1)

#     coords_3d[..., :2] = coords_3d[..., :2] @ rotation_matrix
#     coords_3d = coords_3d - shift[..., None, :]

#     return coords_3d


def point_sampling(coords_3d, img_metas, eps=1e-6):
    lidar2img = torch.stack([coords_3d.new_tensor(each['lidar2img']) for each in img_metas], dim=0)  # (B, N, 4, 4)
    homo_coords_3d = torch.cat([coords_3d, torch.ones_like(coords_3d[..., :1])], dim=-1)             # (B, P, 4)

    lidar2img = lidar2img[:, :, None, :, :]               # (B, N, 1, 4, 4)
    homo_coords_3d = homo_coords_3d[:, None, :, :, None]  # (B, 1, P, 4, 1)

    assert lidar2img.dtype == torch.float32 and homo_coords_3d.dtype == torch.float32

    homo_coords_pixel = (lidar2img @ homo_coords_3d).squeeze(-1)  # (B, N, P, 4)
    # (cx*zc + f*xc, cy*zc + f*yc, zc, 1)

    coords_z = homo_coords_pixel[..., 2:3]                                       # (B, N, P, 1)
    coords_pixel  = homo_coords_pixel[..., :2] / torch.clamp(coords_z, min=eps)  # (B, N, P, 2)
    # (cx + f*(xc/zc), cy + f*(yc/zc))

    coords_pixel_norm = coords_pixel.clone()
    coords_pixel_norm[..., 0] /= img_metas[0]['img_shape'][0][1]  # width
    coords_pixel_norm[..., 1] /= img_metas[0]['img_shape'][0][0]  # height

    cam_mask = ((coords_z[..., 0] > eps)
                & (coords_pixel_norm[..., 1] > 0)
                & (coords_pixel_norm[..., 1] < 1)
                & (coords_pixel_norm[..., 0] > 0)
                & (coords_pixel_norm[..., 0] < 1))  # (B, N, P)
    cam_mask = torch.nan_to_num(cam_mask)           # (B, N, P)

    return coords_pixel_norm, cam_mask


def compose_rt_matrix(r, t):
    assert type(r) in (torch.Tensor, np.ndarray)

    if type(r) == torch.Tensor:
        t = r.new_tensor(t)

        assert (r.shape[-2:] == (3, 3)) and (t.shape[-1] in (1, 3))

        r_h = torch.cat([r, r.new_zeros(*r.shape[:-2], 1, 3)], dim=-2)

        if t.shape[-1] == 3:
            t_h = torch.cat([t, t.new_ones(*t.shape[:-1], 1)], dim=-1)[..., None]
        elif t.shape[-1] == 1:
            t_h = torch.cat([t, t.new_ones(*t.shape[:-2], 1, 1)], dim=-2)

        return torch.cat([r_h, t_h], dim=-1)

    else:
        t = np.array(t, dtype=r.dtype)

        assert (r.shape[-2:] == (3, 3)) and (t.shape[-1] in (1, 3))

        r_h = np.concatenate([r, np.zeros((*r.shape[:-2], 1, 3), dtype=r.dtype)], axis=-2)

        if t.shape[-1] == 3:
            t_h = np.concatenate([t, np.ones((*t.shape[:-1], 1), dtype=t.dtype)], axis=-1)[..., None]
        elif t.shape[-1] == 1:
            t_h = np.concatenate([t, np.ones((*t.shape[:-2], 1, 1), dtype=t.dtype)], axis=-2)

        return np.concatenate([r_h, t_h], axis=-1)
