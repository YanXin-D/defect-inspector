import csv
from pathlib import Path

import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from core import TRANSFORM, build_extractor, flatten_features



ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "splits" / "bottle.csv"
OUTPUT = ROOT / "outputs" / "baseline_v1" / "memory_bank.pt"



class NormalDataset(Dataset):
    def __init__(self):
        with MANIFEST.open(encoding="utf-8") as file:
            self.rows = [
                row for row in csv.DictReader(file)
                if row["split"] == "train"
            ]

        if not self.rows:
            raise ValueError("训练清单为空")

        if any(int(row["label"]) != 0 for row in self.rows):
            raise ValueError("正常特征库不能包含缺陷图片")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path = ROOT / self.rows[index]["path"]
        with Image.open(path) as image:
            tensor = TRANSFORM(image.convert("RGB"))
        return tensor


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请先检查项目环境")

    device = torch.device("cuda")
    torch.hub.set_dir(str(ROOT / ".cache" / "torch"))

    dataset = NormalDataset()
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,  # Windows 首个基线先用单进程读取
        pin_memory=True,
    )

    extractor = build_extractor(device)

    batches = []
    with torch.inference_mode():
        for step, images in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            features = extractor(images)

            if step == 1:
                print("输入形状：", tuple(images.shape))
                print("特征图形状：", tuple(features.shape))

            # [B, C, H, W] -> [B*H*W, C]
            patches = flatten_features(features)

            if not torch.isfinite(patches).all():
                raise ValueError("特征中出现 NaN 或 Inf")

            batches.append(patches.cpu())
            print(f"提取进度：{step}/{len(loader)}")

    memory_bank = torch.cat(batches, dim=0)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "memory_bank": memory_bank,
        "train_paths": [row["path"] for row in dataset.rows],
        "backbone": "resnet18",
        "weights": "IMAGENET1K_V1",
        "feature_layer": "layer3",
        "image_size": 224,
        "normalization": "imagenet",
        "l2_normalized": True,
        "torch_version": str(torch.__version__),
    }, OUTPUT)

    print("训练图片数量：", len(dataset))
    print("特征库形状：", tuple(memory_bank.shape))
    print("保存位置：", OUTPUT)


if __name__ == "__main__":
    main()
