_base_ = '../configs/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco.py'

data_root = '../dataset/'

import matplotlib.pyplot as plt

# class_names = ['class0', 'class1', 'class2']  # Replace with your actual class names

def get_palette(num_classes):
    # Use a colormap to get visually distinct colors
    cmap = plt.get_cmap('tab20' if num_classes <= 20 else 'hsv')
    palette = []
    for i in range(num_classes):
        color = cmap(i / num_classes)[:3]  # Get RGB, ignore alpha
        rgb = tuple(int(255 * c) for c in color)
        palette.append(rgb)
    return palette

# palette = get_palette(len(class_names))
# print(palette)

class_names = ('helmet', 'vest', 'head', 'person')
num_classes = len(class_names)

metainfo = dict(classes=class_names,
    
    # 每个类别的可视化颜色（RGB格式）
    palette=get_palette(len(class_names))
    )

close_mosaic_epochs = 5

base_lr = 0.1
max_epochs = 50
train_batch_size_per_gpu = 12

train_num_workers = 4

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'  # noqa

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)),
    data_preprocessor=dict(
    type='YOLOv5DetDataPreprocessor',
    pad_size_divisor=32,
    batch_augments=[
        dict(
            type='YOLOXBatchSyncRandomResize',
            random_size_range=(480, 800),
            size_divisor=32,
            interval=1)
    ]))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/')))

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
# _base_.optim_wrapper.optimizer.lr = 0.1

_base_.custom_hooks[1].switch_epoch = max_epochs - close_mosaic_epochs

val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=1000),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa