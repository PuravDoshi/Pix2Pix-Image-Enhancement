# Pix2Pix Image Enhancement

A PyTorch implementation of a Pix2Pix-style image enhancement GAN tuned for **low-light -> daylight** translation.
- Generator: U-Net style encoder–decoder with skip connections.
- Discriminator: PatchGAN.
- Losses: adversarial (BCE), L1 (pixel), and LPIPS perceptual loss.
- Evaluation: LPIPS used as the primary perceptual metric (we achieved ~0.27 LPIPS with careful tuning).

Files:
- `dataset.py` — dataset loader for paired images.
- `model.py` — generator & discriminator implementations.
- `train.py` — training loop with LPIPS evaluation and checkpoint saving.
- `test.py` — load a checkpoint and generate enhanced images.

Install:
```bash
pip install torch torchvision lpips pillow tqdm
```
