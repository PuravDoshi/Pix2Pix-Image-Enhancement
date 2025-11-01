import os
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
import lpips

from dataset import PairedImageDataset
from model import Generator, Discriminator

def denorm(tensor):
    return (tensor + 1) / 2.0

def train(
    input_dir,
    target_dir,
    out_dir="runs",
    image_size=256,
    epochs=60,
    batch_size=8,
    lr=2e-4,
    device=None,
    debug=False,
    debug_subset=500,
    eval_every=5,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(out_dir, exist_ok=True)
    dataset = PairedImageDataset(input_dir, target_dir, image_size=image_size, augment=True)

    if debug:
        dataset = Subset(dataset, range(min(debug_subset, len(dataset))))
        print(f"[DEBUG MODE] Using {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True if device=='cuda' else False)
    val_loader = DataLoader(PairedImageDataset(input_dir, target_dir, image_size=image_size, augment=False), batch_size=batch_size, shuffle=False)

    G = Generator().to(device)
    D = Discriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    perceptual = lpips.LPIPS(net='alex').to(device)

    best_lpips = float('inf')

    total_steps = 0
    for epoch in range(1, epochs+1):
        G.train(); D.train()
        epoch_g_loss = 0.0; epoch_d_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for in_img, target_img, _ in pbar:
            in_img = in_img.to(device); target_img = target_img.to(device)

            # Train Discriminator
            fake = G(in_img).detach()
            real_pair = torch.cat([in_img, target_img], dim=1)
            fake_pair = torch.cat([in_img, fake], dim=1)

            opt_D.zero_grad()
            d_real = D(real_pair)
            d_fake = D(fake_pair)
            loss_D = 0.5 * (bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake)))
            loss_D.backward()
            opt_D.step()

            # Train Generator
            opt_G.zero_grad()
            fake_for_g = G(in_img)
            fake_pair_for_g = torch.cat([in_img, fake_for_g], dim=1)
            d_fake_for_g = D(fake_pair_for_g)

            adv_loss = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))
            l1_loss = l1(fake_for_g, target_img) * 100.0  # typical lambda
            lpips_loss = perceptual(fake_for_g, target_img).mean() * 10.0  # weight perceptual

            # Combined generator loss (adversarial + L1 + LPIPS)
            g_loss = adv_loss + l1_loss + lpips_loss
            g_loss.backward()
            opt_G.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += loss_D.item()
            total_steps += 1

            pbar.set_postfix({"G_loss": f"{g_loss.item():.4f}", "D_loss": f"{loss_D.item():.4f}"})

        avg_g = epoch_g_loss / len(loader)
        avg_d = epoch_d_loss / len(loader)
        print(f"[Epoch {epoch}] Avg G: {avg_g:.4f} | Avg D: {avg_d:.4f}")

        # Save sample visuals
        G.eval()
        with torch.no_grad():
            sample_in, sample_target, names = next(iter(val_loader))
            sample_in = sample_in.to(device)
            fake_sample = G(sample_in)
            img_grid = torch.cat([denorm(sample_in[:4]), denorm(fake_sample[:4]), denorm(sample_target[:4])], dim=0)
            save_image(img_grid, os.path.join(out_dir, f"epoch_{epoch:03d}_sample.png"), nrow=4)

        # Evaluate LPIPS every eval_every epochs
        if epoch % eval_every == 0 or epoch == epochs:
            G.eval()
            perceptual_scores = []
            with torch.no_grad():
                for in_img_v, target_img_v, _ in val_loader:
                    in_img_v = in_img_v.to(device); target_img_v = target_img_v.to(device)
                    fake_v = G(in_img_v)
                    score = perceptual(fake_v, target_img_v).view(-1)
                    perceptual_scores.extend(score.cpu().tolist())
            mean_lpips = sum(perceptual_scores) / len(perceptual_scores)
            print(f"===> Eval LPIPS: {mean_lpips:.4f}")
            # Save best model (by LPIPS)
            if mean_lpips < best_lpips:
                best_lpips = mean_lpips
                torch.save(G.state_dict(), os.path.join(out_dir, "generator_best_lpips.pth"))
                torch.save(D.state_dict(), os.path.join(out_dir, "discriminator_best_lpips.pth"))
                print(f"✨ New best LPIPS: {best_lpips:.4f} (models saved)")

    print("Training finished.")
    return G, D

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="night2day/night")
    parser.add_argument("--target_dir", default="night2day/day")
    parser.add_argument("--out_dir", default="runs")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    train(args.input_dir, args.target_dir, out_dir=args.out_dir,
        epochs=args.epochs, batch_size=args.batch_size, debug=args.debug)
