from pathlib import Path
from collections import Counter
import random

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mvtec_ad" / "bottle"
OUT = ROOT / "outputs"
DEFECTS = ("broken_large", "broken_small", "contamination")
TILE = 240


def check_images():
    """检查图片完整性，统计各目录数量和图像尺寸。"""
    sizes = Counter()
    folders = [DATA / "train" / "good", DATA / "test" / "good"]

    for defect in DEFECTS:
        folders.extend([
            DATA / "test" / defect,
            DATA / "ground_truth" / defect,
        ])

    for folder in folders:
        files = sorted(folder.glob("*.png"))
        if not files:
            raise ValueError(f"目录为空或不存在：{folder}")

        print(f"{folder.relative_to(DATA)}: {len(files)}")
        for path in files:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                sizes[image.size] += 1

    print("图像尺寸统计（包含掩码）：", dict(sizes))


def check_masks():
    """检查缺陷图与掩码的对应关系、尺寸和像素值。"""
    for defect in DEFECTS:
        images = sorted((DATA / "test" / defect).glob("*.png"))
        masks = sorted((DATA / "ground_truth" / defect).glob("*.png"))
        expected = {f"{path.stem}_mask.png" for path in images}

        if expected != {path.name for path in masks}:
            raise ValueError(f"{defect} 的图片与掩码没有一一对应")

        for path in images:
            mask_path = DATA / "ground_truth" / defect / f"{path.stem}_mask.png"
            with Image.open(path) as image, Image.open(mask_path) as raw_mask:
                mask = raw_mask.convert("L")
                if image.size != mask.size:
                    raise ValueError(f"图片与掩码尺寸不一致：{path.name}")

                values = set(mask.getdata())
                if not values.issubset({0, 255}) or 255 not in values:
                    raise ValueError(f"掩码不是预期的非空二值图：{mask_path}")

    print("图片与掩码对应检查：PASS")


def make_grid():
    """每类抽一张测试图，展示原图、掩码和红色叠加图。"""
    rng = random.Random(42)
    categories = ("good",) + DEFECTS
    grid = Image.new("RGB", (TILE * 3, (TILE + 32) * 4), "white")
    draw = ImageDraw.Draw(grid)

    for row, category in enumerate(categories):
        path = rng.choice(sorted((DATA / "test" / category).glob("*.png")))

        with Image.open(path) as source:
            image = source.convert("RGB")

        if category == "good":
            mask = Image.new("L", image.size, 0)
        else:
            mask_path = (
                DATA / "ground_truth" / category / f"{path.stem}_mask.png"
            )
            with Image.open(mask_path) as source:
                mask = source.convert("L")

        # 仅在缺陷区域覆盖半透明红色。
        red = Image.new("RGB", image.size, (255, 0, 0))
        tinted = Image.blend(image, red, 0.45)
        overlay = Image.composite(tinted, image, mask)

        panels = [
            ("image", image),
            ("mask", mask.convert("RGB")),
            ("overlay", overlay),
        ]

        y = row * (TILE + 32)
        for col, (title, panel) in enumerate(panels):
            resample = (
                Image.Resampling.NEAREST
                if title == "mask"
                else Image.Resampling.BILINEAR
            )
            preview = ImageOps.contain(panel, (TILE, TILE), method=resample)
            x = col * TILE
            grid.paste(preview, (x, y))
            draw.text(
                (x + 5, y + TILE + 5),
                f"{category} / {title}",
                fill="black",
            )

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "data_preview.png"
    grid.save(output)
    print(f"预览图已保存：{output}")


if __name__ == "__main__":
    check_images()
    check_masks()
    make_grid()