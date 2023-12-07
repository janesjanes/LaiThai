import copy
import os.path as osp
import shutil
from copy import deepcopy
from pathlib import Path

from diffusers import DDPMWuerstchenScheduler
from diffusers.pipelines.wuerstchen import WuerstchenPrior
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase
from torch import nn
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffengine.engine.hooks import PeftSaveHook
from diffengine.models.editors import (
    SDDataPreprocessor,
    SDXLDataPreprocessor,
    StableDiffusion,
    StableDiffusionXL,
    WuerstchenPriorModel,
)
from diffengine.models.editors.wuerstchen.efficient_net_encoder import (
    EfficientNetEncoder,
)
from diffengine.models.losses import L2Loss
from diffengine.models.utils import TimeSteps, WhiteNoise, WuerstchenRandomTimeSteps


class DummyWuerstchenPriorModel(WuerstchenPriorModel):
    def __init__(
        self,
        decoder_model: str = "warp-ai/wuerstchen",
        prior_model: str = "warp-ai/wuerstchen-prior",
        loss: dict | None = None,
        prior_lora_config: dict | None = None,
        text_encoder_lora_config: dict | None = None,
        prior_loss_weight: float = 1.,
        data_preprocessor: dict | nn.Module | None = None,
        noise_generator: dict | None = None,
        timesteps_generator: dict | None = None,
        input_perturbation_gamma: float = 0.0,
        *,
        finetune_text_encoder: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": "SDDataPreprocessor"}
        if noise_generator is None:
            noise_generator = {"type": "WhiteNoise"}
        if timesteps_generator is None:
            timesteps_generator = {"type": "WuerstchenRandomTimeSteps"}
        if loss is None:
            loss = {"type": "L2Loss", "loss_weight": 1.0}
        super(WuerstchenPriorModel, self).__init__(data_preprocessor=data_preprocessor)
        if text_encoder_lora_config is not None:
            assert finetune_text_encoder, (
                "If you want to use LoRA for text encoder, "
                "you should set finetune_text_encoder=True."
            )
        if finetune_text_encoder and prior_lora_config is not None:
            assert text_encoder_lora_config is not None, (
                "If you want to finetune text encoder with LoRA Unet, "
                "you should set text_encoder_lora_config."
            )

        self.decoder_model = decoder_model
        self.prior_model = prior_model
        self.prior_lora_config = deepcopy(prior_lora_config)
        self.text_encoder_lora_config = deepcopy(text_encoder_lora_config)
        self.finetune_text_encoder = finetune_text_encoder
        self.prior_loss_weight = prior_loss_weight
        self.gradient_checkpointing = gradient_checkpointing
        self.input_perturbation_gamma = input_perturbation_gamma

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module: nn.Module = loss
        assert not self.loss_module.use_snr, \
            "WuerstchenPriorModel does not support SNR loss."

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip")
        config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        self.text_encoder = CLIPTextModel(config)

        self.image_encoder = EfficientNetEncoder(c_latent=2)

        self.scheduler = DDPMWuerstchenScheduler()

        model_kwargs = {
            "c_in": 2,
            "c": 8,
            "depth": 2,
            "c_cond": 32,
            "c_r": 8,
            "nhead": 2,
        }
        self.prior = WuerstchenPrior(**model_kwargs)
        self.noise_generator = MODELS.build(noise_generator)
        self.timesteps_generator = MODELS.build(timesteps_generator)
        self.prepare_model()
        self.set_lora()


class DummyWrapper(BaseModel):

    def __init__(self, model) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            model = MODELS.build(model)
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class TestPeftSaveHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name="DummyWrapper", module=DummyWrapper)
        MODELS.register_module(name="StableDiffusion", module=StableDiffusion)
        MODELS.register_module(
            name="StableDiffusionXL", module=StableDiffusionXL)
        MODELS.register_module(
            name="SDDataPreprocessor", module=SDDataPreprocessor)
        MODELS.register_module(
            name="SDXLDataPreprocessor", module=SDXLDataPreprocessor)
        MODELS.register_module(name="L2Loss", module=L2Loss)
        MODELS.register_module(name="WhiteNoise", module=WhiteNoise)
        MODELS.register_module(name="TimeSteps", module=TimeSteps)
        MODELS.register_module(name="DummyWuerstchenPriorModel",
                               module=DummyWuerstchenPriorModel)
        MODELS.register_module(name="WuerstchenRandomTimeSteps",
                               module=WuerstchenRandomTimeSteps)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop("DummyWrapper")
        MODELS.module_dict.pop("StableDiffusion")
        MODELS.module_dict.pop("StableDiffusionXL")
        MODELS.module_dict.pop("SDDataPreprocessor")
        MODELS.module_dict.pop("SDXLDataPreprocessor")
        MODELS.module_dict.pop("L2Loss")
        MODELS.module_dict.pop("WhiteNoise")
        MODELS.module_dict.pop("TimeSteps")
        MODELS.module_dict.pop("DummyWuerstchenPriorModel")
        MODELS.module_dict.pop("WuerstchenRandomTimeSteps")
        return super().tearDown()

    def test_init(self):
        PeftSaveHook()

    def test_before_save_checkpoint(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "StableDiffusion"
        cfg.model.unet_lora_config = dict(
                    type="LoRA", r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"])
        cfg.model.model = "diffusers/tiny-stable-diffusion-torch"
        runner = self.build_runner(cfg)
        checkpoint = dict(
            state_dict=StableDiffusion(
                model="diffusers/tiny-stable-diffusion-torch",
                unet_lora_config=dict(
                    type="LoRA", r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
                ).state_dict())
        hook = PeftSaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/unet",
                     "adapter_model.safetensors")).exists()
        shutil.rmtree(
            osp.join(runner.work_dir, f"step{runner.iter}"))

        for key in checkpoint["state_dict"]:
            assert key.startswith(("unet", "text_encoder"))
            assert "default" in key

    def test_before_save_checkpoint_text_encoder(self):
        # with text encoder
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "StableDiffusion"
        cfg.model.unet_lora_config = dict(
                    type="LoRA", r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"])
        cfg.model.text_encoder_lora_config = dict(
                    type="LoRA", r=4,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
        cfg.model.finetune_text_encoder = True
        cfg.model.model = "diffusers/tiny-stable-diffusion-torch"
        runner = self.build_runner(cfg)
        checkpoint = dict(
            state_dict=StableDiffusion(
                model="diffusers/tiny-stable-diffusion-torch",
                unet_lora_config=dict(
                    type="LoRA", r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
                text_encoder_lora_config=dict(
                    type="LoRA", r=4,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]),
                finetune_text_encoder=True).state_dict())
        hook = PeftSaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/unet",
                     "adapter_model.safetensors")).exists()
        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/text_encoder",
                     "adapter_model.safetensors")).exists()
        shutil.rmtree(
            osp.join(runner.work_dir, f"step{runner.iter}"))

        for key in checkpoint["state_dict"]:
            assert key.startswith(("unet", "text_encoder"))
            assert "default" in key

        # sdxl
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "StableDiffusionXL"
        cfg.model.unet_lora_config = dict(
                    type="LoRA", r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"])
        cfg.model.text_encoder_lora_config = dict(
                    type="LoRA", r=4,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
        cfg.model.finetune_text_encoder = True
        cfg.model.model = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        runner = self.build_runner(cfg)
        checkpoint = dict(
            state_dict=StableDiffusionXL(
                model="hf-internal-testing/tiny-stable-diffusion-xl-pipe",
                unet_lora_config=dict(
                    type="LoRA", r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
                text_encoder_lora_config=dict(
                    type="LoRA", r=4,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]),
                finetune_text_encoder=True).state_dict())
        hook = PeftSaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/unet",
                     "adapter_model.safetensors")).exists()
        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/text_encoder_one",
                     "adapter_model.safetensors")).exists()
        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/text_encoder_two",
                     "adapter_model.safetensors")).exists()
        shutil.rmtree(
            osp.join(runner.work_dir, f"step{runner.iter}"))

        for key in checkpoint["state_dict"]:
            assert key.startswith(("unet", "text_encoder"))
            assert "default" in key

    def test_before_save_checkpoint_wuerstchen(self):
        # with text encoder
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "DummyWuerstchenPriorModel"
        cfg.model.prior_lora_config = dict(
                    type="LoRA", r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"])
        cfg.model.text_encoder_lora_config = dict(
                    type="LoRA", r=4,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
        cfg.model.finetune_text_encoder = True
        runner = self.build_runner(cfg)
        checkpoint = dict(
            state_dict=DummyWuerstchenPriorModel(
                prior_lora_config=dict(
                    type="LoRA", r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
                text_encoder_lora_config=dict(
                    type="LoRA", r=4,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]),
                finetune_text_encoder=True).state_dict())
        hook = PeftSaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/prior",
                     "adapter_model.safetensors")).exists()
        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/text_encoder",
                     "adapter_model.safetensors")).exists()
        shutil.rmtree(
            osp.join(runner.work_dir, f"step{runner.iter}"))

        for key in checkpoint["state_dict"]:
            assert key.startswith(("prior", "text_encoder"))
            assert "default" in key
