from mmengine.config import read_base
from transformers import AutoImageProcessor, SiglipVisionModel

with read_base():
    from .._base_.datasets.pokemon_blip_xl_ip_adapter_siglip_384 import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl_ip_adapter_plus import *
    from .._base_.schedules.stable_diffusion_xl_50e import *


model.image_encoder = dict(
    type=SiglipVisionModel.from_pretrained,
    pretrained_model_name_or_path="google/siglip-so400m-patch14-384")
model.feature_extractor = dict(
    type=AutoImageProcessor.from_pretrained,
    pretrained_model_name_or_path="google/siglip-so400m-patch14-384")

train_dataloader.update(batch_size=1)

optim_wrapper.update(accumulative_counts=4)  # update every four times

train_cfg.update(by_epoch=True, max_epochs=100)
