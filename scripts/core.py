"""Shared baseline: preprocessing, frozen features, nearest neighbours and rendering."""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parents[1]
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_extractor(device):
    torch.hub.set_dir(str(ROOT / '.cache' / 'torch'))
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(backbone.children())[:7]).to(device).eval()
    model.requires_grad_(False)
    return model


def flatten_features(features):
    patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    patches = F.normalize(patches, p=2, dim=1)
    if not torch.isfinite(patches).all():
        raise ValueError('Features contain NaN or Inf')
    return patches


def nearest_distances(queries, bank, chunk_size=4096):
    if bank.ndim != 2 or len(bank) == 0 or queries.shape[1] != bank.shape[1]:
        raise ValueError('Invalid memory bank shape')
    minimum = torch.full((queries.shape[0],), float('inf'), device=queries.device)
    for chunk in bank.split(chunk_size):
        distances = torch.cdist(queries, chunk, p=2)
        minimum = torch.minimum(minimum, distances.min(dim=1).values)
    return minimum


@dataclass
class Prediction:
    score: float
    threshold: float
    distance_map: torch.Tensor

    def report(self):
        width = self.distance_map.shape[1]
        row, col = divmod(int(self.distance_map.argmax()), width)
        return {
            'anomaly_score': self.score,
            'threshold': self.threshold,
            'prediction': 'anomaly' if self.score > self.threshold else 'normal',
            'decision_rule': 'anomaly_score > threshold',
            'max_grid_row': row,
            'max_grid_col': col,
            'score_method': 'max_patch_nearest_l2',
        }


class AnomalyDetector:
    def __init__(self, artifact_dir=None, device='auto'):
        self.artifact_dir = Path(artifact_dir or ROOT / 'outputs' / 'baseline_v1')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        checkpoint = torch.load(self.artifact_dir / 'memory_bank.pt',
                                map_location='cpu', weights_only=True)
        expected = {'backbone': 'resnet18', 'weights': 'IMAGENET1K_V1',
                    'feature_layer': 'layer3', 'image_size': 224,
                    'normalization': 'imagenet', 'l2_normalized': True}
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f'Unsupported memory bank metadata: {key}')
        self.bank = checkpoint['memory_bank'].to(self.device)
        if self.bank.ndim != 2 or self.bank.shape[1] != 256 or len(self.bank) == 0:
            raise ValueError('Expected a nonempty [N, 256] memory bank')
        if not torch.isfinite(self.bank).all():
            raise ValueError('Memory bank contains NaN or Inf')
        calibration = json.loads((self.artifact_dir / 'threshold.json').read_text('utf-8'))
        if calibration['score_method'] != 'max_patch_nearest_l2':
            raise ValueError('Unsupported scoring method')
        self.threshold = float(calibration['threshold'])
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 2:
            raise ValueError('Invalid threshold')
        self.extractor = build_extractor(self.device)

    @torch.inference_mode()
    def predict(self, image):
        features = self.extractor(TRANSFORM(image.convert('RGB')).unsqueeze(0).to(self.device))
        distances = nearest_distances(flatten_features(features), self.bank)
        if not torch.isfinite(distances).all():
            raise ValueError('Distances contain NaN or Inf')
        distance_map = distances.reshape(features.shape[-2:]).cpu()
        return Prediction(float(distances.max()), self.threshold, distance_map)


def render_panels(image, prediction):
    # Cap display size; original dimensions are not needed for a coarse 14x14 map.
    image = image.convert('RGB').copy()
    image.thumbnail((1024, 1024))
    values = F.interpolate(prediction.distance_map[None, None],
                           size=(image.height, image.width), mode='bilinear',
                           align_corners=False)[0, 0]
    values = (values / 2).clamp(0, 1)
    rgb = torch.stack([(255 * values).round(), torch.zeros_like(values),
                       (255 * (1 - values)).round()], dim=-1).to(torch.uint8)
    heatmap = Image.fromarray(rgb.numpy())
    overlay = Image.blend(image, heatmap, alpha=0.45)
    report = prediction.report()
    h, w = prediction.distance_map.shape
    x = (report['max_grid_col'] + 0.5) * image.width / w
    y = (report['max_grid_row'] + 0.5) * image.height / h
    r = max(4, round(image.width * 0.015))
    ImageDraw.Draw(overlay).ellipse((x-r, y-r, x+r, y+r), outline='yellow', width=3)
    return image, heatmap, overlay


def render_comparison(image, prediction):
    panels = render_panels(image, prediction)
    canvas = Image.new('RGB', (960, 385), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(zip(
            ['Original', 'Distance: blue=0, red=2', 'Overlay / max grid'], panels)):
        panel = panel.copy()
        panel.thumbnail((320, 320))
        canvas.paste(panel, (index * 320 + (320-panel.width)//2, (320-panel.height)//2))
        draw.text((index * 320 + 6, 327), title, fill='black')
    draw.text((6, 352), f'score={prediction.score:.6f}  threshold={prediction.threshold:.6f}'
              f"  prediction={prediction.report()['prediction']}", fill='black')
    return canvas
