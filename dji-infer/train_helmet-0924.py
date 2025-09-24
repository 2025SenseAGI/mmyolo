# ==============================================================
# Small-object focused config for drone helmet detection (MMYOLO)
# - 统一输入分辨率 1280（train/val/test）
# - 解除 backbone 冻结；小目标友好的 assigner/NMS/损失权重
# - AMP 半精度 + 降 Mosaic 画布，缓解 OOM
# - 不改网络深/宽/步长/检测层/BN 等结构性参数
# 建议环境变量（可选，缓解碎片化）：
#   export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"
# ==============================================================

_base_ = '../configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'

import matplotlib.pyplot as plt

# -------------------- 数据集与类别 --------------------
data_root = '../dataset/'  # 按需修改

def get_palette(num_classes):
    cmap = plt.get_cmap('tab20' if num_classes <= 20 else 'hsv')
    return [tuple(int(255 * c) for c in cmap(i / num_classes)[:3]) for i in range(num_classes)]

class_names = ('helmet', 'vest', 'head', 'person')
num_classes = len(class_names)
metainfo = dict(classes=class_names, palette=get_palette(num_classes))

# -------------------- 训练时长/批大小/线程数 --------------------
max_epochs = 200
close_mosaic_epochs = 30  # 提前关 Mosaic，减轻显存尖峰
train_batch_size_per_gpu = 6    # 如仍 OOM 可降到 4
train_num_workers = 4

# -------------------- 模型覆写（只动可改项） --------------------
model = dict(
    backbone=dict(frozen_stages=-1),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        loss_bbox=dict(loss_weight=10.0)
    ),
    train_cfg=dict(
        assigner=dict(
            num_classes=num_classes,
            topk=8,
            beta=7.0
        )
    ),
    test_cfg=dict(
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=400
    )
)

# -------------------- 优化器（AMP 半精度以省显存） --------------------
# 注意：不要在 optimizer 里放 batch_size_per_gpu
# 学习率建议结合实际总 batch 调整；也可启用 auto_scale_lr（如下）
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.937, weight_decay=5e-4
    ),
    loss_scale='dynamic'
)

# 可选：让框架按总 batch 自动线性缩放学习率（需你的版本支持）
auto_scale_lr = dict(enable=True, base_batch_size=128)  # 基线 8x16 = 128

# -------------------- dataloader：小框不过滤，稳定显存 --------------------
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=1)
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root, metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/')
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = val_evaluator

# -------------------- 训练钩子/时长 --------------------
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=3, save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5),
)

# 关闭 Mosaic 的切换点（保持与基线相同的 custom_hooks 结构）
_base_.custom_hooks[1].switch_epoch = max_epochs - close_mosaic_epochs

# -------------------- 统一分辨率 = 1280，且“偏放大”的仿射 --------------------
img_scale = (1280, 1280)

# 覆盖 test/val 的 Resize（基线一般是 KeepRatioResize -> LetterResize）
_base_.test_pipeline[1]['scale'] = img_scale  # YOLOv5KeepRatioResize
_base_.test_pipeline[2]['scale'] = img_scale  # LetterResize

# 训练主阶段：Mosaic + RandomAffine 偏向放大，但避免超大画布
for p in _base_.train_pipeline:
    if p.get('type') == 'Mosaic':
        p['img_scale'] = img_scale
    if p.get('type') == 'YOLOv5RandomAffine':
        p['scaling_ratio_range'] = (0.9, 1.5)
        p['border'] = (-img_scale[0] // 4, -img_scale[1] // 4)

# 收尾阶段（关闭 Mosaic 后）也统一 1280，并允许少量放大
for p in _base_.train_pipeline_stage2:
    if p.get('type') in ('YOLOv5KeepRatioResize', 'LetterResize'):
        p['scale'] = img_scale
    if p.get('type') == 'YOLOv5RandomAffine':
        p['scaling_ratio_range'] = (1.0, 1.5)
        p['border'] = (-img_scale[0] // 4, -img_scale[1] // 4)

# -------------------- 预训练（可选） --------------------
# load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'
