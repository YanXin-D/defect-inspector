import csv
import json
from pathlib import Path

import torch

from PIL import Image


from core import TRANSFORM, build_extractor, flatten_features, nearest_distances



ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "baseline_v1"
MANIFEST = ROOT / "data" / "splits" / "bottle.csv"


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


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

    calibration = json.loads(
        (OUTPUT / "threshold.json").read_text(encoding="utf-8")
    )
    threshold = calibration["threshold"]

    with MANIFEST.open(encoding="utf-8") as file:
        rows = [
            row for row in csv.DictReader(file)
            if row["split"] == "test"
        ]

    if not rows:
        raise ValueError("测试清单为空")

    if set(checkpoint["train_paths"]) & {row["path"] for row in rows}:
        raise ValueError("测试集与特征库训练图片存在重叠")

    extractor = build_extractor(device)

    predictions = []
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            with Image.open(ROOT / row["path"]) as image:
                tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0)

            features = extractor(tensor.to(device))
            patches = flatten_features(features)
            score = nearest_distances(patches, bank).max().item()

            if not torch.isfinite(torch.tensor(score)):
                raise ValueError(f"分数无效：{row['path']}")

            label = int(row["label"])
            prediction = int(score > threshold)
            predictions.append({
                "path": row["path"],
                "defect_type": row["defect_type"],
                "label": label,
                "prediction": prediction,
                "anomaly_score": score,
                "correct": int(label == prediction),
            })

            print(
                f"[{index}/{len(rows)}] "
                f"true={label} pred={prediction} score={score:.6f}"
            )

    tp = sum(p["label"] == 1 and p["prediction"] == 1 for p in predictions)
    fn = sum(p["label"] == 1 and p["prediction"] == 0 for p in predictions)
    fp = sum(p["label"] == 0 and p["prediction"] == 1 for p in predictions)
    tn = sum(p["label"] == 0 and p["prediction"] == 0 for p in predictions)

    by_defect = {}
    for category in sorted({p["defect_type"] for p in predictions}):
        if category == "good":
            continue
        subset = [
            p for p in predictions if p["defect_type"] == category
        ]
        detected = sum(p["prediction"] == 1 for p in subset)
        by_defect[category] = {
            "total": len(subset),
            "detected": detected,
            "recall": detected / len(subset),
        }

    report = {
        "threshold": threshold,
        "test_count": len(predictions),
        "confusion_matrix": {
            "TN_normal_correct": tn,
            "FP_normal_flagged": fp,
            "FN_defect_missed": fn,
            "TP_defect_detected": tp,
        },
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "false_positive_rate": divide(fp, fp + tn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "accuracy": divide(tp + tn, len(predictions)),
        "by_defect": by_defect,
    }

    with (OUTPUT / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(predictions[0].keys())
        )
        writer.writeheader()
        writer.writerows(predictions)

    (OUTPUT / "test_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("\n测试结果：")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
