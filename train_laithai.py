from mmengine.config import read_base

with read_base():
  from diffengine.configs._base_.datasets.fill50k_controlnet import *
  from diffengine.configs._base_.default_runtime import *
  from diffengine.configs._base_.models.stable_diffusion_v15_controlnet import *
  from diffengine.configs._base_.schedules.stable_diffusion_50e import *

model.update(loss=dict(type='SNRL2Loss', snr_gamma=5.0, loss_weight=1.0))
