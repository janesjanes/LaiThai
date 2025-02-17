import torchvision
from mmengine.dataset import DefaultSampler

from diffengine.datasets import HFControlNetDataset
from diffengine.datasets.transforms import (
    DumpImage,
    PackInputs,
    RandomCrop,
    RandomHorizontalFlip,
    TorchVisonTransformWrapper,
)
from diffengine.engine.hooks import ControlNetSaveHook, VisualizationHook

train_pipeline = [
    dict(
        type=TorchVisonTransformWrapper,
        transform=torchvision.transforms.Resize,
        size=550,
        interpolation="nearest",
        keys=["img", "condition_img"]),
    dict(type=RandomCrop, size=512, keys=["img", "condition_img"]),
    dict(type=RandomHorizontalFlip, p=0.5, keys=["img", "condition_img"]),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.ToTensor,
         keys=["img", "condition_img"]),
    dict(type=DumpImage, max_imgs=10, dump_dir="work_dirs/dump"),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Normalize, mean=[0.5], std=[0.5]),
    dict(type=PackInputs, input_keys=["img", "condition_img", "text"]),
]
train_dataloader = dict(
    batch_size=10,
    num_workers=1,
    dataset=dict(
        type=HFControlNetDataset,
        dataset="datasets/",
        condition_column="conditioning_image",
        caption_column="text",
        image_column="file_name",
        csv="metadata_train.csv",
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type=VisualizationHook,
        prompt=["Traditional Thai Line Art"],
        condition_image=[
            'datasets/LaiThai_dataset_sketch/test/00034_05.jpg'  # noqa
        ]),
    dict(type=ControlNetSaveHook),
]
