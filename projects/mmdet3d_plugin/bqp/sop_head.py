import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from mmcv.runner.base_module import BaseModule
from mmdet.models import DETECTORS
from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS, build_head, build_loss
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.models.utils import clip_sigmoid

from projects.mmdet3d_plugin.bqp.utils import denormalize_coords, generate_3d_grid, point_sampling
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask


@HEADS.register_module()
class SOPHead(BaseModule):
    def __init__(self,
                 bev_size=None,
                 num_scales=None,
                 fast_ray_transform=None,
                 train_cfg=None):
        super().__init__()

        self.Hb, self.Wb = bev_size

        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(256*num_scales, 256, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(256),
            nn.ReLU(),
        )

        self.height_collapse = nn.Sequential(
            nn.Conv2d(256*6, 256, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(256),
            nn.ReLU()
        )

        self.fast_ray_transform = build_head(fast_ray_transform)

        self.objectness_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        )

        self.train_cfg = train_cfg
        self.loss_cls = build_loss(train_cfg['loss_cls'])

    def align_multi_scale_features(self, multi_scale_feats, target_scale):
        B, N, C, _, _ = multi_scale_feats[0].shape

        aligned_feats = list()
        for scale, feat in enumerate(multi_scale_feats):
            feat = feat.flatten(0, 1)
            aligned_feat = F.interpolate(feat, size=target_scale, mode='bilinear')
            aligned_feat = aligned_feat.reshape(B, N, *aligned_feat.shape[1:])
            aligned_feats.append(aligned_feat)

        return aligned_feats

    def forward_test(self, img_feats, img_metas, objectness_threshold):
        B, N, C, H, W = img_feats[0].shape

        assert B == 1

        aligned_feats = self.align_multi_scale_features(img_feats, (H, W))
        aligned_feats = torch.cat(aligned_feats, dim=2)
        fused_feats = self.multi_scale_fusion(aligned_feats.flatten(0, 1)).reshape(B, N, -1, H, W)

        voxel_features = self.fast_ray_transform(fused_feats, img_metas)
        bev_features = self.height_collapse(voxel_features.flatten(1, 2))

        objectness_logits = self.objectness_head(bev_features)

        objectness_probs = objectness_logits.sigmoid()
        objectness_probs = F.interpolate(objectness_probs, size=[self.Hb, self.Wb], mode='bilinear')
        objectness_probs = objectness_probs[0, 0].transpose(0, 1)

        spatial_anchors = (objectness_probs > objectness_threshold).nonzero()

        if len(spatial_anchors) == 0:
            return None

        return spatial_anchors

    def forward_train(self, img_feats, img_metas, gt_labels_3d, gt_bboxes_3d):
        B, N, C, H, W = img_feats[0].shape

        aligned_feats = self.align_multi_scale_features(img_feats, (H, W))
        aligned_feats = torch.cat(aligned_feats, dim=2)
        fused_feats = self.multi_scale_fusion(aligned_feats.flatten(0, 1)).reshape(B, N, -1, H, W)

        voxel_features = self.fast_ray_transform(fused_feats, img_metas)
        bev_features = self.height_collapse(voxel_features.flatten(1, 2))

        objectness_logits = self.objectness_head(bev_features)

        loss = self.loss_objectness(objectness_logits, gt_bboxes_3d, gt_labels_3d)

        return loss

    def loss_objectness(self, pred_objectness_logits, gt_bboxes_3d, gt_labels_3d):
        pred_objectness = clip_sigmoid(pred_objectness_logits)
        objectness_targets = self.get_objectness_targets(gt_bboxes_3d, gt_labels_3d)

        num_pos = (objectness_targets==1).sum().item()

        loss_objectness = self.loss_cls(pred_objectness.flatten(0, 1), objectness_targets.flatten(0, 1), avg_factor=max(num_pos, 1))

        loss_dict = dict()        
        loss_dict['loss_objectness'] = loss_objectness

        return loss_dict

    def get_objectness_targets(self, gt_bboxes_3d, gt_labels_3d):
        B = len(gt_bboxes_3d)

        objectness_targets = list()
        for batch in range(B):
            objectness_target = self.get_objectness_targets_single(gt_bboxes_3d[batch], gt_labels_3d[batch])
            objectness_targets.append(objectness_target.unsqueeze(0))

        objectness_targets = torch.stack(objectness_targets, dim=0)

        return objectness_targets

    def get_objectness_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        gt_bboxes_3d = torch.cat([gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1).to(gt_labels_3d.device)

        grid_size = self.train_cfg['sampling_grid_size']
        grid_range = self.train_cfg['sampling_grid_range']
        voxel_size = [(grid_range[3] - grid_range[0]) / grid_size[2], (grid_range[4] - grid_range[1]) / grid_size[1]]

        objectness_target = gt_bboxes_3d.new_zeros([grid_size[1], grid_size[2]])

        num_objs = min(len(gt_bboxes_3d), self.train_cfg['max_objs'])

        for k in range(num_objs):
            width = gt_bboxes_3d[k][3] / voxel_size[0]
            length = gt_bboxes_3d[k][4] / voxel_size[1]

            if not ((width > 0) and (length > 0)):
                continue

            radius = gaussian_radius([length, width], min_overlap=self.train_cfg['gaussian_overlap'])
            radius = max(self.train_cfg['min_radius'], int(radius))

            x = gt_bboxes_3d[k][0]
            y = gt_bboxes_3d[k][1]

            coord_x = (x - grid_range[0]) / voxel_size[0]
            coord_y = (y - grid_range[1]) / voxel_size[1]

            center = gt_bboxes_3d.new_tensor([coord_x, coord_y]).long()

            if not ((0 <= center[0] < grid_size[2]) and (0 <= center[1] < grid_size[1])):
                continue

            draw_heatmap_gaussian(objectness_target, center, radius)

        return objectness_target


@DETECTORS.register_module()
class TrainSOPHead(MVXTwoStageDetector):
    def __init__(self,
                 img_backbone=None,
                 img_neck=None,
                 sop_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)

        self.sop_head = builder.build_head(sop_head)

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            raise AssertionError("TrainSOPHead is for training only. forward_test is not supported.")

    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      **kwargs):
        len_queue = img.shape[1]
        img = img[:, -1, :, :, :, :]
        img_metas = [each[len_queue-1] for each in img_metas]

        self.eval()
        with torch.no_grad():
            img_feats = self.extract_feat(img=img)
        self.train()

        for i, feat in enumerate(img_feats):
            img_feats[i] = feat.float()

        losses = dict()

        loss_sop = self.sop_head.forward_train(img_feats, img_metas, gt_labels_3d, gt_bboxes_3d)
        losses.update(loss_sop)

        return losses

    @autocast()
    def extract_feat(self, img, len_queue=None):
        B = img.size(0)

        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)

        img = self.grid_mask(img)

        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped


@HEADS.register_module()
class FastRayTransform(BaseModule):
    def __init__(self, sampling_grid_size, sampling_grid_range):
        super().__init__()

        self.sampling_grid_size = sampling_grid_size
        self.sampling_grid_range = sampling_grid_range

        self.register_buffer('sampling_points_3d', self.generate_sampling_points_3d(sampling_grid_size, sampling_grid_range))

    def generate_sampling_points_3d(self, grid_size, grid_range):
        xyz_norm = generate_3d_grid(*grid_size)
        xyz_real = denormalize_coords(xyz_norm, pc_range=grid_range)

        return xyz_real

    def forward(self, img_feats, img_metas):
        B, N, C, H, W = img_feats.shape

        projected_pixels, valid_cam_mask = point_sampling(
            coords_3d=self.sampling_points_3d.unsqueeze(0).repeat(B, 1, 1),
            img_metas=img_metas
        )

        projected_pixels_int = projected_pixels * projected_pixels.new_tensor([W, H])
        projected_pixels_int = projected_pixels_int.long()

        voxel_features = img_feats.new_zeros([B, C, len(self.sampling_points_3d)])
        for batch in range(B):
            for cam in range(N):
                valid = valid_cam_mask[batch, cam, :]

                x = projected_pixels_int[..., 0][batch, cam, valid]
                y = projected_pixels_int[..., 1][batch, cam, valid]

                assert ((x >= 0) & (x < W) & (y >= 0) & (y < H)).all()

                voxel_features[batch, :, valid] = img_feats[batch, cam, :, y, x]

        voxel_features = voxel_features.reshape(B, C, *self.sampling_grid_size)

        return voxel_features
