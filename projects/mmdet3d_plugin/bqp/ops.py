import numpy as np
import torch


def generate_densification_offsets(densification_radius, voxel_size):
    voxel_radius = int(densification_radius/voxel_size) + 1

    grid_1d = torch.arange(-voxel_radius, voxel_radius+1)
    grid_x, grid_y = torch.meshgrid(grid_1d, grid_1d)

    grid_mask = (grid_x**2 + grid_y**2) <= voxel_radius**2

    densification_offsets = torch.stack((grid_x[grid_mask], grid_y[grid_mask]), dim=1)

    return densification_offsets


def densify_anchor_coords(anchor_coords, densification_offsets, bev_h, bev_w):
    densified_bev_coords = (anchor_coords[:, None, :] + densification_offsets[None, :, :]).reshape(-1, 2)

    valid_mask = ((densified_bev_coords[:, 0] >= 0) & (densified_bev_coords[:, 0] < bev_w) & (densified_bev_coords[:, 1] >= 0) & (densified_bev_coords[:, 1] < bev_h))
    densified_bev_coords = densified_bev_coords[valid_mask]

    bev_mask = torch.zeros((bev_w, bev_h), dtype=torch.bool, device=densified_bev_coords.device)
    bev_mask[densified_bev_coords[:, 0], densified_bev_coords[:, 1]] = True
    densified_bev_coords = bev_mask.nonzero()

    return densified_bev_coords


def lidar_coords_to_bev_coords(lidar_coords, bev_h, bev_w, pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
    bev_coords = lidar_coords.new_zeros(*lidar_coords.shape[:-1], 2)
    bev_coords[..., 0] = ((lidar_coords[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])) * bev_w
    bev_coords[..., 1] = ((lidar_coords[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])) * bev_h
    bev_coords = bev_coords.long()

    return bev_coords


def gt_bbox_centers_to_bev_coords(gt_bboxes, gt_labels, bev_h, bev_w, pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
    gt_bboxes = gt_bboxes.tensor
    gt_bboxes = gt_bboxes[gt_labels != -1]

    lidar_coords = gt_bboxes[:, :2]
    bev_coords = lidar_coords_to_bev_coords(lidar_coords, bev_h, bev_w, pc_range)

    valid_mask = ((bev_coords[:, 0] >= 0) & (bev_coords[:, 0] < bev_w) & (bev_coords[:, 1] >= 0) & (bev_coords[:, 1] < bev_h))
    bev_coords = bev_coords[valid_mask]

    return bev_coords


def get_ego_motion(img_meta):
    ego_delta_yaw = img_meta['can_bus'][-1] * (np.pi / 180)

    ego_delta_x = img_meta['can_bus'][0]
    ego_delta_y = img_meta['can_bus'][1]

    ego_global_yaw = img_meta['can_bus'][-2]

    sin_yaw = np.sin(ego_global_yaw)
    cos_yaw = np.cos(ego_global_yaw)

    ego_shift_x = ego_delta_x * sin_yaw - ego_delta_y * cos_yaw
    ego_shift_y = ego_delta_x * cos_yaw + ego_delta_y * sin_yaw

    ego_shift = [ego_shift_x, ego_shift_y]

    return ego_delta_yaw, ego_shift


def propagate_previous_detections(prev_preds, ego_delta_yaw, ego_shift, delta_t=0.5):
    prev_centers = prev_preds[:, :2]
    prev_vels = prev_preds[:, 7:9]

    projected_centers = prev_centers + (prev_vels * delta_t)

    ego_rot_mat = projected_centers.new_tensor([
        [np.cos(-ego_delta_yaw), -np.sin(-ego_delta_yaw)],
        [np.sin(-ego_delta_yaw),  np.cos(-ego_delta_yaw)]
    ])

    ego_trans = projected_centers.new_tensor(ego_shift)

    ego_compensated_centers = projected_centers @ ego_rot_mat.T - ego_trans

    return ego_compensated_centers


def precompute_column_offsets(spatial_shapes):
    base_offset = spatial_shapes.new_tensor([-1, 1])
    widths = spatial_shapes[:, 1]

    offsets = base_offset[None, :] / widths[:, None]

    return offsets


def build_column_value_mask(ref_pixel, bev_mask, spatial_shapes, offsets=None, eps=1e-6):
    N, _, L, Z, _ = ref_pixel.shape
    S = spatial_shapes.size(0)

    widths = spatial_shapes[:, 1]
    heights = spatial_shapes[:, 0]

    widths_list = widths.tolist()
    heights_list = heights.tolist()

    offset_per_scale = torch.cat([widths.new_zeros([1]), widths.cumsum(0)[:-1]])
    total_width = sum(widths_list)

    ref_pixel = ref_pixel.squeeze(1)
    bev_mask = bev_mask.squeeze(1)

    cam_idxs, query_idxs = bev_mask.any(-1).nonzero(as_tuple=True)
    ref_x = ref_pixel[cam_idxs, query_idxs, :, 0].mean(-1)

    value_mask = bev_mask.new_zeros([N, total_width])

    use_single_column = offsets is None

    if use_single_column:
        ref_x = ref_x.unsqueeze(1).clamp_(min=0, max=(1 - eps))
        ref_x = (ref_x * widths).long()

        valid_cam_idxs = cam_idxs.unsqueeze(1).expand(-1, S).flatten()
        valid_idxs = (ref_x + offset_per_scale).flatten()

        value_mask[valid_cam_idxs, valid_idxs] = True

    else:
        ref_x = ref_x[:, None, None].expand(-1, *offsets.shape) + offsets
        ref_x = ref_x.clamp(min=0, max=(1 - eps))
        ref_x = (ref_x * widths[None, :, None]).long()

        x_min, x_max = ref_x.unbind(-1)
        widths_range = (x_max - x_min + 1).clamp(min=0)

        x_range = torch.arange(widths_range.max(), device=widths_range.device)
        x_start = x_min

        valid_mask = x_range[None, None, :] < widths_range[:, :, None]
        idx_col, idx_scale, idx_x = torch.where(valid_mask)

        valid_cam_idxs = cam_idxs[:, None, None].expand(*valid_mask.shape)[idx_col, idx_scale, idx_x]

        flat_indices = x_start[:, :, None] + x_range[None, None, :] + offset_per_scale[None, :, None]
        valid_idxs = flat_indices[idx_col, idx_scale, idx_x]

        value_mask[valid_cam_idxs, valid_idxs] = True

    chunks = value_mask.split(widths_list, dim=1)
    expanded_chunks = [chunk.repeat(1, h) for chunk, h in zip(chunks, heights_list)]
    value_mask = torch.cat(expanded_chunks, dim=1)

    return value_mask.nonzero(as_tuple=True)
