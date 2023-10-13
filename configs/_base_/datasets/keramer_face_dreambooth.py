train_pipeline = [
    {
        "type": "torchvision/Resize",
        "size": 512,
        "interpolation": "bilinear",
    },
    {
        "type": "RandomCrop",
        "size": 512,
    },
    {
        "type": "RandomHorizontalFlip",
        "p": 0.5,
    },
    {
        "type": "torchvision/ToTensor",
    },
    {
        "type": "torchvision/Normalize",
        "mean": [0.5],
        "std": [0.5],
    },
    {
        "type": "PackInputs",
    },
]
train_dataloader = {
    "batch_size": 4,
    "num_workers": 4,
    "dataset": {
        "type": "HFDreamBoothDataset",
        "dataset": "diffusers/keramer-face-example",
        "instance_prompt": "a photo of sks person",
        "pipeline": train_pipeline,
        "class_prompt": "a photo of person",
    },
    "sampler": {
        "type": "InfiniteSampler",
        "shuffle": True,
    },
}

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    {
        "type": "VisualizationHook",
        "prompt": ["a photo of sks person in suits"] * 4,
        "by_epoch": False,
        "interval": 100,
    },
    {
        "type": "LoRASaveHook",
    },
]
