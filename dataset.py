import os
import glob
from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms as T

class PairedImageDataset(Dataset):
    """
    Dataset of paired low-light (input) and enhanced/daylight (target) images.
    Expects matching filenames in `input_dir` and `target_dir`.
    Example structure:
        input_dir/0001.png
        target_dir/0001.png
    """
    def __init__(self, input_dir, target_dir, image_size=256, augment=False):
        self.input_paths = sorted(glob.glob(os.path.join(input_dir, "*")))
        self.target_dir = target_dir
        self.image_size = image_size
        self.augment = augment

        self.base_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)  
        ])

        self.augment_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(5),
        ])

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        in_path = self.input_paths[idx]
        fname = os.path.basename(in_path)
        target_path = os.path.join(self.target_dir, fname)

        # load
        in_img = Image.open(in_path).convert("RGB")
        try:
            target_img = Image.open(target_path).convert("RGB")
        except FileNotFoundError:
            target_img = in_img.copy()

        if self.augment and random.random() < 0.5:
            in_img = T.functional.hflip(in_img)
            target_img = T.functional.hflip(target_img)
        in_t = self.base_transform(in_img)
        target_t = self.base_transform(target_img)
        return in_t, target_t, fname