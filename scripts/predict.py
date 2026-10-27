"""Single-image CLI, using the same inference core as the Streamlit app."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from core import ROOT, AnomalyDetector, render_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'outputs' / 'baseline_v1')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'predictions')
    args = parser.parse_args()
    image_path = args.image if args.image.is_absolute() else ROOT / args.image
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    detector = AnomalyDetector(args.artifacts, args.device)
    prediction = detector.predict(image)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = f'{image_path.parent.name}_{image_path.stem}'
    report = prediction.report()
    report['image'] = str(args.image)
    (args.output_dir / f'{name}.json').write_text(json.dumps(report, indent=2), 'utf-8')
    torch.save(prediction.distance_map, args.output_dir / f'{name}_map.pt')
    render_comparison(image, prediction).save(args.output_dir / f'{name}.png')
    print(json.dumps(report, indent=2))
    print(f'Saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
