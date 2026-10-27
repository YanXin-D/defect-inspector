"""Backward-compatible visualization command; shared inference lives in core.py."""
import argparse
from pathlib import Path
import torch
from PIL import Image
from core import ROOT, AnomalyDetector, render_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='data/mvtec_ad/bottle/test/good/014.png')
    args = parser.parse_args()
    path = Path(args.image)
    path = path if path.is_absolute() else ROOT / path
    with Image.open(path) as source:
        image = source.convert('RGB')
    prediction = AnomalyDetector().predict(image)
    destination = ROOT / 'outputs' / 'baseline_v1' / 'visualizations'
    destination.mkdir(parents=True, exist_ok=True)
    name = f'{path.parent.name}_{path.stem}'
    torch.save({'distance_map': prediction.distance_map, 'image_path': str(path)},
               destination / f'{name}_map.pt')
    render_comparison(image, prediction).save(destination / f'{name}.png')
    print(prediction.report())
    print(f'Saved: {destination / (name + ".png")}')


if __name__ == '__main__':
    main()
