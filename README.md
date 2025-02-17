# LaiThaiGen: Generating High-Quality Lai Thai Art from Simple Line Drawings

This repository contains the implementation of **LaiThaiGen**, as introduced in our paper:

**"LaiThaiGen: Generating High-Quality Lai Thai Art from Simple Line Drawings"**  
by *Ronnagorn Rattanatamma, Sittiphong Pornudomthap, and Patsorn Sangkloy*  

[Paper Link (coming soon)]  

LaiThaiGen is a fine tuned ControlNet model that transforms rough line sketches into high-fidelity **Lai Thai** ornamental designs while preserving artistic structure and composition. Our goal is to make **traditional Thai pattern design more accessible**, even for those without formal artistic training.

---

# Training
To train LaiThaiGen, use the following command:

```mim train diffengine train_laithai.py --cfg-options work_dir="work_dirs/laithai"```

# Inference
Example inference code is provided in ```inference_example.py```. Adjust the paths to your sketch file and model checkpoint accordingly.

# Model checkpoint
The model checkpoint can be downloaded from [here](https://drive.google.com/file/d/1fjMbI6VKom4SVjcvvG7AuB0ws9CDJTrw/view?usp=sharing). Extract it to a folder, and point to that folder when loading the checkpoint.

# Dataset
Our Lai Thai dataset is available upon request. For access or further information, please contact patsorn.sangkloy@gmail.com

# Citation
If you find this work useful, please consider citing:
```
@article{LaiThaiGen2024,
  author    = {Ronnagorn Rattanatamma and Sittiphong Pornudomthap and Patsorn Sangkloy},
  title     = {LaiThaiGen: Generating High-Quality Lai Thai Art from Simple Line Drawings},
  journal   = {TBD},
  year      = {2024}
}
```

# Acknowledgement
This repository is a fork of [okotaku/diffengine](https://github.com/okotaku/diffengine).
We thank the authors of previous works for their contributions to diffusion-based generative modeling.
