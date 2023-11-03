from unittest import TestCase

import pytest
import torch
from mmengine.optim import OptimWrapper
from torch.optim import SGD

from diffengine.models.editors import SDDataPreprocessor, StableDiffusion
from diffengine.models.losses import DeBiasEstimationLoss, L2Loss, SNRL2Loss


class TestStableDiffusion(TestCase):

    def test_infer(self):
        StableDiffuser = StableDiffusion(
            "diffusers/tiny-stable-diffusion-torch",
            data_preprocessor=SDDataPreprocessor())

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        # test device
        assert StableDiffuser.device.type == "cpu"

        # test infer with negative_prompt
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            negative_prompt="noise",
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        # output_type = 'latent'
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            output_type="latent",
            height=64,
            width=64)
        assert len(result) == 1
        assert type(result[0]) == torch.Tensor
        assert result[0].shape == (4, 32, 32)

    def test_train_step(self):
        # test load with loss module
        StableDiffuser = StableDiffusion(
            "diffusers/tiny-stable-diffusion-torch",
            loss=L2Loss(),
            data_preprocessor=SDDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_input_perturbation(self):
        # test load with loss module
        StableDiffuser = StableDiffusion(
            "diffusers/tiny-stable-diffusion-torch",
            input_perturbation_gamma=0.1,
            loss=L2Loss(),
            data_preprocessor=SDDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_gradient_checkpointing(self):
        # test load with loss module
        StableDiffuser = StableDiffusion(
            "diffusers/tiny-stable-diffusion-torch",
            loss=L2Loss(),
            data_preprocessor=SDDataPreprocessor(),
            gradient_checkpointing=True)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_dreambooth(self):
        # test load with loss module
        StableDiffuser = StableDiffusion(
            "diffusers/tiny-stable-diffusion-torch",
            loss=L2Loss(),
            data_preprocessor=SDDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a sks dog"]))
        data["inputs"]["result_class_image"] = dict(
            img=[torch.zeros((3, 64, 64))],
            text=["a dog"])  # type: ignore[assignment]
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_snr_loss(self):
        # test load with loss module
        StableDiffuser = StableDiffusion(
            "diffusers/tiny-stable-diffusion-torch",
            loss=SNRL2Loss(),
            data_preprocessor=SDDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_debias_estimation_loss(self):
        # test load with loss module
        StableDiffuser = StableDiffusion(
            "diffusers/tiny-stable-diffusion-torch",
            loss=DeBiasEstimationLoss(),
            data_preprocessor=SDDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_val_and_test_step(self):
        StableDiffuser = StableDiffusion(
            "diffusers/tiny-stable-diffusion-torch",
            loss=L2Loss(),
            data_preprocessor=SDDataPreprocessor())

        # test val_step
        with pytest.raises(NotImplementedError, match="val_step is not"):
            StableDiffuser.val_step(torch.zeros((1, )))

        # test test_step
        with pytest.raises(NotImplementedError, match="test_step is not"):
            StableDiffuser.test_step(torch.zeros((1, )))
