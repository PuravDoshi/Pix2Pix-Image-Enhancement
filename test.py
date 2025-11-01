import os
import argparse
import torch
from torchvision.utils import save_image
import torchvision.transforms as T
from PIL import Image
from model import Generator

def denorm(x):
    return (x + 1) / 2.0

def load_image(path, image_size=256, device="cpu"):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def translate(model_path, input_path, out_path, device=None, image_size=256):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator().to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()

    inp = load_image(input_path, image_size=image_size, device=device)
    with torch.no_grad():
        out = G(inp)
    save_image(denorm(out), out_path)
    print(f"Saved enhanced image to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to generator .pth")
    parser.add_argument("--input", required=True, help="Path to low-light image")
    parser.add_argument("--out", default="enhanced.png")
    args = parser.parse_args()
    translate(args.model, args.input, args.out)