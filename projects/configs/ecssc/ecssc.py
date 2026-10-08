"""
ECSSC (full model): VoxFormer-T stage 2 + event encoders (E_ego / E_obj) with separate 3D
cross-attention and 3-way gated fusion (ECSSCHead), plus auxiliary depth / moving-object supervision.
"""
_base_ = ['../voxformer/voxformer-T_deform3D.py']

work_dir = 'work_dirs/ecssc'

model = dict(
    type='ECSSC',
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),

    # event branch
    event_channels=128,    # same as the FPN output channels
    num_event_bins=5,
    lambda_ego=0.1,        # weight of the auxiliary depth loss (E_ego)
    lambda_obj=0.1,        # weight of the auxiliary moving-object loss (E_obj)
    img_H=370,
    img_W=1220,
    ev_H=352,
    ev_W=1216,
    ablation=None,         # None | 'ego_only' | 'obj_only'

    # test-time options (override from tools/test.py: --blur-severity / --blur-angle / --event-zero)
    blur_severity=0,
    blur_angle=0.0,
    event_zero=False,

    pts_bbox_head=dict(
        _delete_=True,
        type='ECSSCHead',
        use_event_3d=True,

        # same transformer settings as the VoxFormer-T stage-2 head
        bev_h=128, bev_w=128, bev_z=16,
        embed_dims=128,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        cross_transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            embed_dims=128,
            num_cams=5,
            encoder=dict(
                type='VoxFormerEncoder',
                num_layers=3,
                pc_range=[0, -25.6, -2.0, 51.2, 25.6, 4.4],
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='VoxFormerLayer',
                    attn_cfgs=[dict(
                        type='DeformCrossAttention',
                        pc_range=[0, -25.6, -2.0, 51.2, 25.6, 4.4],
                        num_cams=5,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=128,
                            num_points=8,
                            num_levels=1),
                        embed_dims=128)],
                    ffn_cfgs=dict(
                        type='FFN', embed_dims=128,
                        feedforward_channels=1024,
                        num_fcs=2, ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=('cross_attn','norm','ffn','norm')))),
        self_transformer=dict(
            type='PerceptionTransformer3D',
            rotate_prev_bev=True,
            use_shift=True,
            embed_dims=128,
            num_cams=5,
            encoder=dict(
                type='VoxFormerEncoder3D',
                num_layers=2,
                pc_range=[0, -25.6, -2.0, 51.2, 25.6, 4.4],
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='VoxFormerLayer3D',
                    attn_cfgs=[dict(
                        type='DeformSelfAttention3DCustom',
                        embed_dims=128,
                        num_levels=1,
                        num_points=8)],
                    ffn_cfgs=dict(
                        type='FFN', embed_dims=128,
                        feedforward_channels=1024,
                        num_fcs=2, ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=('self_attn','norm','ffn','norm')))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=64,
            row_num_embed=512,
            col_num_embed=512),
    ),
)

dataset_type = 'SemanticKittiDatasetECSSC'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(type=dataset_type),
    val=dict(type=dataset_type),
    test=dict(type=dataset_type),
)

checkpoint_config = dict(_delete_=True, interval=1, max_keep_ckpts=3)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            # event encoders are trained from scratch: higher lr
            'ego_encoder': dict(lr_mult=2.0),
            'obj_encoder': dict(lr_mult=2.0),
            'event_fusion': dict(lr_mult=2.0),
            # pretrained RGB backbone: lower lr
            'img_backbone': dict(lr_mult=0.1),
        }
    )
)

log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook')]
)

evaluation = dict(interval=1, save_best='ssc_SemanticKITTI/mIoU')
