import copy
import os.path as osp
import shutil
from pathlib import Path

from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase
from torch import nn

from diffengine.engine.hooks import IPAdapterSaveHook
from diffengine.models.editors import IPAdapterXL, IPAdapterXLDataPreprocessor
from diffengine.models.losses import L2Loss
from diffengine.models.utils import TimeSteps, WhiteNoise


class DummyWrapper(BaseModel):

    def __init__(self, model) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            model = MODELS.build(model)
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class TestIPAdapterSaveHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name="DummyWrapper", module=DummyWrapper)
        MODELS.register_module(name="IPAdapterXL", module=IPAdapterXL)
        MODELS.register_module(
            name="IPAdapterXLDataPreprocessor",
            module=IPAdapterXLDataPreprocessor)
        MODELS.register_module(name="L2Loss", module=L2Loss)
        MODELS.register_module(name="WhiteNoise", module=WhiteNoise)
        MODELS.register_module(name="TimeSteps", module=TimeSteps)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop("DummyWrapper")
        MODELS.module_dict.pop("IPAdapterXL")
        MODELS.module_dict.pop("IPAdapterXLDataPreprocessor")
        MODELS.module_dict.pop("L2Loss")
        MODELS.module_dict.pop("WhiteNoise")
        MODELS.module_dict.pop("TimeSteps")
        return super().tearDown()

    def test_init(self):
        IPAdapterSaveHook()

    def test_before_save_checkpoint(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "IPAdapterXL"
        cfg.model.model = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        cfg.model.image_encoder = 'hf-internal-testing/unidiffuser-diffusers-test'  # noqa
        cfg.model.image_encoder_sub_folder = "image_encoder"
        runner = self.build_runner(cfg)
        checkpoint = dict(
            state_dict=IPAdapterXL(
                model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
                image_encoder="hf-internal-testing/unidiffuser-diffusers-test",
                image_encoder_sub_folder="image_encoder",
            ).state_dict())
        hook = IPAdapterSaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}",
                     "ip_adapter.bin")).exists()
        shutil.rmtree(
            osp.join(runner.work_dir, f"step{runner.iter}"))

        for key in checkpoint["state_dict"]:
            assert key.startswith(("unet", "image_projection"))
            assert ".processor." in key or key.startswith("image_projection")
