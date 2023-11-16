from unittest import TestCase

import pytest
import torch
from mmengine.optim import OptimWrapper
from torch.optim import SGD

from diffengine.models.editors import IPAdapterXL, IPAdapterXLDataPreprocessor
from diffengine.models.losses import L2Loss


class TestIPAdapterXL(TestCase):

    def test_init(self):
        with pytest.raises(
                AssertionError, match="`unet_lora_config` should be None"):
            _ = IPAdapterXL(
                "hf-internal-testing/tiny-stable-diffusion-pipe",
                image_encoder="hf-internal-testing/unidiffuser-diffusers-test",
                data_preprocessor=IPAdapterXLDataPreprocessor(),
                unet_lora_config=dict(type="dummy"))

        with pytest.raises(
                AssertionError, match="`text_encoder_lora_config` should be None"):
            _ = IPAdapterXL(
                "hf-internal-testing/tiny-stable-diffusion-pipe",
                image_encoder="hf-internal-testing/unidiffuser-diffusers-test",
                data_preprocessor=IPAdapterXLDataPreprocessor(),
                text_encoder_lora_config=dict(type="dummy"))

        with pytest.raises(
                AssertionError,
                match="`finetune_text_encoder` should be False"):
            _ = IPAdapterXL(
                "hf-internal-testing/tiny-stable-diffusion-pipe",
                image_encoder="hf-internal-testing/unidiffuser-diffusers-test",
                data_preprocessor=IPAdapterXLDataPreprocessor(),
                finetune_text_encoder=True)

    def test_infer(self):
        StableDiffuser = IPAdapterXL(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            image_encoder="hf-internal-testing/unidiffuser-diffusers-test",
            data_preprocessor=IPAdapterXLDataPreprocessor())

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        # test device
        assert StableDiffuser.device.type == "cpu"

        # test infer with negative_prompt
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            negative_prompt="noise",
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        # output_type = 'latent'
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            output_type="latent",
            height=64,
            width=64)
        assert len(result) == 1
        assert type(result[0]) == torch.Tensor
        assert result[0].shape == (4, 32, 32)

    def test_train_step(self):
        # test load with loss module
        StableDiffuser = IPAdapterXL(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            image_encoder="hf-internal-testing/unidiffuser-diffusers-test",
            loss=L2Loss(),
            data_preprocessor=IPAdapterXLDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                text=["a dog"],
                clip_img=[torch.zeros((3, 32, 32))],
                time_ids=[torch.zeros((1, 6))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_gradient_checkpointing(self):
        # test load with loss module
        StableDiffuser = IPAdapterXL(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            image_encoder="hf-internal-testing/unidiffuser-diffusers-test",
            loss=L2Loss(),
            data_preprocessor=IPAdapterXLDataPreprocessor(),
            gradient_checkpointing=True)

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                text=["a dog"],
                clip_img=[torch.zeros((3, 32, 32))],
                time_ids=[torch.zeros((1, 6))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_val_and_test_step(self):
        StableDiffuser = IPAdapterXL(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            image_encoder="hf-internal-testing/unidiffuser-diffusers-test",
            loss=L2Loss(),
            data_preprocessor=IPAdapterXLDataPreprocessor())

        # test val_step
        with pytest.raises(NotImplementedError, match="val_step is not"):
            StableDiffuser.val_step(torch.zeros((1, )))

        # test test_step
        with pytest.raises(NotImplementedError, match="test_step is not"):
            StableDiffuser.test_step(torch.zeros((1, )))
