model = dict(
    type="DeepFloydIF",
    model="DeepFloyd/IF-I-XL-v1.0",
    unet_lora_config=dict(
        type="LoRA",
        r=8,
        lora_alpha=1,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
    gradient_checkpointing=True)
