_base_ = [
    "../_base_/models/tiny_sd_lora.py",
    "../_base_/datasets/keramer_face_dreambooth.py",
    "../_base_/schedules/stable_diffusion_1k.py",
    "../_base_/default_runtime.py",
]

train_dataloader = {
    "dataset": {
        "class_image_config": {
            "model": {{_base_.model.model}},
        },
        "instance_prompt": "Portrait photo of a sks person",
        "class_prompt": "Portrait photo of a person",
    },
}

custom_hooks = [
    {
        "type": "VisualizationHook",
        "prompt": ["Portrait photo of a sks person in suits"] * 4,
        "by_epoch": False,
        "interval": 100,
    },
    {
        "type": "LoRASaveHook",
    },
]
