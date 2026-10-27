import csv
import json
import math
from pathlib import Path

import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader


from core import TRANSFORM, build_extractor, flatten_features, nearest_distances


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "splits" / "bottle.csv"
OUTPUT = ROOT / "outputs" / "baseline_v1"


class ValidationDataset(Dataset):
    def __init__(self):
        with MANIFEST.open(encoding="utf-8") as file:
            self.rows = [
                row for row in csv.DictReader(file)
                if row["split"] == "val"
            ]

        if not self.rows:
            raise ValueError("验证清单为空")

        if any(int(row["label"]) != 0 for row in self.rows):
            raise ValueError("当前阈值校准只使用正常验证图")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        relative_path = self.rows[index]["path"]
        with Image.open(ROOT / relative_path) as image:
            tensor = TRANSFORM(image.convert("RGB"))
        return tensor, relative_path



def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    device = torch.device("cuda")
    torch.hub.set_dir(str(ROOT / ".cache" / "torch"))

    checkpoint = torch.load(
        OUTPUT / "memory_bank.pt",
        map_location="cpu",
        weights_only=True,
    )
    bank = checkpoint["memory_bank"].to(device)

    dataset = ValidationDataset()
    overlap = set(checkpoint["train_paths"]) & {
        row["path"] for row in dataset.rows
    }
    if overlap:
        raise ValueError("验证集与特征库训练图片存在重叠")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    extractor = build_extractor(device)

    records = []
    with torch.inference_mode():
        for index, (images, paths) in enumerate(loader, start=1):
            features = extractor(
                images.to(device, non_blocking=True)
            )
            patches = flatten_features(features)

            distances = nearest_distances(patches, bank)
            score = distances.max().item()

            if not math.isfinite(score):
                raise ValueError(f"异常分数不是有限值：{paths[0]}")

            records.append({
                "path": paths[0],
                "anomaly_score": score,
            })
            print(f"[{index}/{len(dataset)}] score={score:.6f}")

    # 用正常验证分数的 95% 分位数设定初始阈值。
    scores = torch.tensor([
        row["anomaly_score"] for row in records
    ], dtype=torch.float64)
    threshold = torch.quantile(scores, 0.95).item()

    # 统一约定：只有严格大于阈值才判为异常。
    false_positives = sum(
        row["anomaly_score"] > threshold for row in records
    )
    report = {
        "score_method": "max_patch_nearest_l2",
        "threshold_method": "normal_val_quantile_0.95_linear",
        "threshold": threshold,
        "decision_rule": "anomaly_score > threshold",
        "validation_count": len(records),
        "validation_false_positives": false_positives,
        "validation_false_positive_rate": (
            false_positives / len(records)
        ),
        "score_min": scores.min().item(),
        "score_median": scores.median().item(),
        "score_max": scores.max().item(),
    }

    with (OUTPUT / "val_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=["path", "anomaly_score"]
        )
        writer.writeheader()
        writer.writerows(records)

    report_path = OUTPUT / "threshold.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n校准结果：")
    print(json.dumps(report, indent=2))
    print(f"保存位置：{report_path}")


if __name__ == "__main__":
    main()

