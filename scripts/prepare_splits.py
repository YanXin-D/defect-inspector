import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mvtec_ad" / "bottle"
OUTPUT = ROOT / "data" / "splits" / "bottle.csv"
SEED = 42


def relative(path):
    return path.relative_to(ROOT).as_posix()


def main():
    normal_images = sorted((DATA / "train" / "good").glob("*.png"))
    if len(normal_images) != 209:
        raise ValueError(f"预期 209 张正常训练图，实际 {len(normal_images)}")

    # 先排序再随机打乱，保证相同数据和种子得到相同划分。
    random.Random(SEED).shuffle(normal_images)
    n_val = round(len(normal_images) * 0.2)
    val_images = normal_images[:n_val]
    train_images = normal_images[n_val:]

    rows = []
    for split, images in [
        ("train", train_images),
        ("val", val_images),
    ]:
        for path in images:
            rows.append({
                "path": relative(path),
                "split": split,
                "label": 0,
                "defect_type": "good",
                "mask_path": "",
            })

    # 测试集保留官方划分，只登记路径。
    for path in sorted((DATA / "test").glob("*/*.png")):
        defect_type = path.parent.name
        is_defect = defect_type != "good"
        mask_path = ""

        if is_defect:
            mask = (
                DATA / "ground_truth" / defect_type
                / f"{path.stem}_mask.png"
            )
            if not mask.is_file():
                raise FileNotFoundError(mask)
            mask_path = relative(mask)

        rows.append({
            "path": relative(path),
            "split": "test",
            "label": int(is_defect),
            "defect_type": defect_type,
            "mask_path": mask_path,
        })

    # 每张图片只能出现在一个划分中。
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("存在跨划分的重复路径")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "path", "split", "label", "defect_type", "mask_path"
        ])
        writer.writeheader()
        writer.writerows(rows)

    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split]
        defects = sum(row["label"] for row in subset)
        print(
            f"{split}: total={len(subset)}, "
            f"normal={len(subset) - defects}, defect={defects}"
        )

    print(f"划分清单已保存：{OUTPUT}")


if __name__ == "__main__":
    main()