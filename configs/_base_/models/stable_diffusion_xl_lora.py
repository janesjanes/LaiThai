model = dict(
    type="StableDiffusionXL",
    model="stabilityai/stable-diffusion-xl-base-1.0",
    vae_model="madebyollin/sdxl-vae-fp16-fix",
    unet_lora_config=dict(
        type="LoRA",
        r=8,
        lora_alpha=1,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"]))
