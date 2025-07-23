_base_ = '../configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'

data_root = 'dji-data/'

class_names = ('ConcretePumpTruck', 'Concretetanktruck', 'DrillingMachine', 'RefuseCar', 'bulldozer', 'car', 'excavator')
num_classes = len(class_names)

metainfo = dict(classes=class_names,
    
    # 每个类别的可视化颜色（RGB格式）
    palette=[
        (255, 0, 0),     # 红色 - ConcretePumpTruck
        (0, 255, 0),     # 绿色 - Concretetanktruck
        (0, 0, 255),     # 蓝色 - DrillingMachine
        (255, 255, 0),   # 黄色 - RefuseCar
        (255, 0, 255),   # 粉色 - bulldozer
        (0, 255, 255),   # 青色 - car
        (128, 0, 128)    # 紫色 - excavator
    ])
close_mosaic_epochs = 5

max_epochs = 40
train_batch_size_per_gpu = 12
train_num_workers = 4

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'  # noqa

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
_base_.custom_hooks[1].switch_epoch = max_epochs - close_mosaic_epochs

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json')
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa