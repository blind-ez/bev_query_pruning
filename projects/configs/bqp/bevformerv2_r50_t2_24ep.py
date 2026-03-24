_base_ = [
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(mean=[103.53, 116.28, 123.675], std=[1, 1, 1], to_rgb=False)

class_names = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

bev_h_ = 200
bev_w_ = 200
frames = (-1, 0,)
group_detr = 11

ida_aug_conf = {
    "reisze": [512, 544, 576, 608, 640, 672, 704, 736, 768],
    "crop": (0, 260, 1600, 900),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}
ida_aug_conf_eval = {
    "reisze": [640, ],
    "crop": (0, 260, 1600, 900),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}

runtime_options = dict(
    run_backbone_fp16=True,
    bqp=True,
    oracle=False,
    sop=True,
    tap=True,
    sop_threshold=0.20,
    tap_threshold=0.10,
    densification_radius=6.0,
    qpa=True,
    prune_values_in_tsa=False,
    prune_values_in_sca=False,
    count_num_qkv=False,
    measure_latency=False,
    num_qkv_log_path='log/num_qkv/num_qkv.txt',
    latency_log_path='log/latency/latency.txt',
)

_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
_num_levels_ = 4
_num_mono_levels_ = 5

model = dict(
    type='BEVFormerV2',
    use_grid_mask=True,
    video_test_mode=False,
    num_levels=_num_levels_,
    num_mono_levels=_num_mono_levels_,
    mono_loss_weight=1.0,
    frames=frames,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN'),
        norm_eval=False,
        style='caffe'),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_mono_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerHead_GroupDETR',
        group_detr=group_detr,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformerV2',
            embed_dims=_dim_,
            frames=frames,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=4),
                            embed_dims=_dim_)
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='GroupMultiheadAttention',
                            group=group_detr,
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1)
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=0.75, beta=1.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    fcos3d_bbox_head=dict(
        type='NuscenesDD3D',
        num_classes=10,
        in_channels=_dim_,
        strides=[8, 16, 32, 64, 128],
        box3d_on=True,
        feature_locations_offset='none',
        fcos2d_cfg=dict(
            num_cls_convs=4,
            num_box_convs=4,
            norm='SyncBN',
            use_deformable=False,
            use_scale=True,
            box2d_scale_init_factor=1.0),
        fcos2d_loss_cfg=dict(
            focal_loss_alpha=0.25, focal_loss_gamma=2.0, loc_loss_type='giou'),
        fcos3d_cfg=dict(
            num_convs=4,
            norm='SyncBN',
            use_scale=True,
            depth_scale_init_factor=0.3,
            proj_ctr_scale_init_factor=1.0,
            use_per_level_predictors=False,
            class_agnostic=False,
            use_deformable=False,
            mean_depth_per_level=[44.921, 20.252, 11.712, 7.166, 8.548],
            std_depth_per_level=[24.331, 9.833, 6.223, 4.611, 8.275]),
        fcos3d_loss_cfg=dict(
            min_depth=0.1,
            max_depth=80.0,
            box3d_loss_weight=2.0,
            conf3d_loss_weight=1.0,
            conf_3d_temperature=1.0,
            smooth_l1_loss_beta=0.05,
            max_loss_per_group=20,
            predict_allocentric_rot=True,
            scale_depth_by_focal_lengths=True,
            scale_depth_by_focal_lengths_factor=500.0,
            class_agnostic=False,
            predict_distance=False,
            canon_box_sizes=[[2.3524184, 0.5062202, 1.0413622],
                             [0.61416006, 1.7016163, 1.3054738],
                             [2.9139307, 10.725025, 3.2832346],
                             [1.9751819, 4.641267, 1.74352],
                             [2.772134, 6.565072, 3.2474296],
                             [0.7800532, 2.138673, 1.4437162],
                             [0.6667362, 0.7181772, 1.7616143],
                             [0.40246472, 0.4027083, 1.0084083],
                             [3.0059454, 12.8197, 4.1213827],
                             [2.4986045, 6.9310856, 2.8382742]]),
        target_assign_cfg=dict(
            center_sample=True,
            pos_radius=1.5,
            sizes_of_interest=((-1, 64), (64, 128), (128, 256), (256, 512),
                               (512, 100000000.0))),
        nusc_loss_weight=dict(attr_loss_weight=0.2, speed_loss_weight=0.2)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='SmoothL1Cost', weight=0.75),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range))))

model.update(bev_size=[bev_h_, bev_w_])
model.update(runtime_options=runtime_options)

if runtime_options['sop']:
    from projects.configs.bqp.train_sop_head_bevformerv2_r50_t2_24ep import model as m
    model.update(sop_head=m['sop_head'])

eval_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, ),
    dict(type='CropResizeFlipImage', data_aug_conf=ida_aug_conf_eval, training=False, debug=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 640),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='CustomCollect3D', keys=['img', 'ego2global_translation', 'ego2global_rotation', 'lidar2ego_translation', 'lidar2ego_rotation', 'timestamp'])
        ]
    )
]

if runtime_options['oracle']:
    eval_pipeline.insert(0, dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False))
    eval_pipeline[-1]['transforms'][-1]['keys'].extend(['gt_bboxes_3d', 'gt_labels_3d'])

dataset_type = 'CustomNuScenesDatasetV2'
data_root = 'data/nuscenes/'

data = dict(
    workers_per_gpu=4,
    nonshuffler_sampler=dict(type='DistributedSampler'),
    test=dict(
        type=dataset_type,
        frames=frames,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        pipeline=eval_pipeline,
        classes=class_names,
        modality=input_modality
    )
)
