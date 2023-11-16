from unittest import TestCase

import pytest
import torch
from mmengine.optim import OptimWrapper
from peft import PeftModel
from torch.optim import SGD

from diffengine.models.editors import DeepFloydIF, SDDataPreprocessor
from diffengine.models.losses import L2Loss


class TestDeepFloydIF(TestCase):

    def test_init(self):
        with pytest.raises(
                AssertionError, match="If you want to use LoRA for text"):
            _ = DeepFloydIF(
                "hf-internal-testing/tiny-if-pipe",
                finetune_text_encoder=False,
                text_encoder_lora_config = dict(
                    type="LoRA", r=4,
                    target_modules=["q", "k", "v", "o"]),
                data_preprocessor=SDDataPreprocessor())
        with pytest.raises(
                AssertionError, match="If you want to finetune text encoder"):
            _ = DeepFloydIF(
                "hf-internal-testing/tiny-if-pipe",
                finetune_text_encoder=True,
                unet_lora_config=dict(
                    type="LoRA", r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
                data_preprocessor=SDDataPreprocessor())

    def test_infer(self):
        StableDiffuser = DeepFloydIF(
            "hf-internal-testing/tiny-if-pipe",
            data_preprocessor=SDDataPreprocessor())

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            height=16,
            width=16)
        assert len(result) == 1
        assert result[0].shape == (16, 16, 3)

        # test device
        assert StableDiffuser.device.type == "cpu"

        # test infer with negative_prompt
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            negative_prompt="noise",
            height=16,
            width=16)
        assert len(result) == 1
        assert result[0].shape == (16, 16, 3)

        # output_type = 'pt'
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            output_type="pt",
            height=16,
            width=16)
        assert len(result) == 1
        assert type(result[0]) == torch.Tensor
        assert result[0].shape == (3, 16, 16)

    def test_infer_lora(self):
        StableDiffuser = DeepFloydIF(
            "hf-internal-testing/tiny-if-pipe",
            finetune_text_encoder=True,
            unet_lora_config=dict(
                    type="LoRA", r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
            text_encoder_lora_config = dict(
                    type="LoRA", r=4,
                    target_modules=["q", "k", "v", "o"]),
            data_preprocessor=SDDataPreprocessor())
        assert isinstance(StableDiffuser.unet, PeftModel)
        assert isinstance(StableDiffuser.text_encoder, PeftModel)

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            height=16,
            width=16)
        assert len(result) == 1
        assert result[0].shape == (16, 16, 3)

    def test_train_step(self):
        # test load with loss module
        StableDiffuser = DeepFloydIF(
            "hf-internal-testing/tiny-if-pipe",
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
        StableDiffuser = DeepFloydIF(
            "hf-internal-testing/tiny-if-pipe",
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
        StableDiffuser = DeepFloydIF(
            "hf-internal-testing/tiny-if-pipe",
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
        StableDiffuser = DeepFloydIF(
            "hf-internal-testing/tiny-if-pipe",
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

    def test_train_step_lora(self):
        # test load with loss module
        StableDiffuser = DeepFloydIF(
            "hf-internal-testing/tiny-if-pipe",
            loss=L2Loss(),
            unet_lora_config=dict(
                    type="LoRA", r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
            text_encoder_lora_config = dict(
                    type="LoRA", r=4,
                    target_modules=["q", "k", "v", "o"]),
            finetune_text_encoder=True,
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
        StableDiffuser = DeepFloydIF(
            "hf-internal-testing/tiny-if-pipe",
            loss=L2Loss(),
            data_preprocessor=SDDataPreprocessor())

        # test val_step
        with pytest.raises(NotImplementedError, match="val_step is not"):
            StableDiffuser.val_step(torch.zeros((1, )))

        # test test_step
        with pytest.raises(NotImplementedError, match="test_step is not"):
            StableDiffuser.test_step(torch.zeros((1, )))
