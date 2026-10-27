"""Compare user-rerun evaluation with archived baseline; does not run inference."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    def load(path):
        with path.open(encoding='utf-8-sig') as file:
            rows = list(csv.DictReader(file))
        if len({r['path'] for r in rows}) != len(rows):
            raise ValueError('Duplicate prediction paths')
        return {r['path']: r for r in rows}
    old = load(ROOT / 'docs/results/test_predictions.csv')
    new = load(ROOT / 'outputs/baseline_v1/test_predictions.csv')
    if old.keys() != new.keys() or len(new) != 83:
        raise ValueError('Test image list changed')
    largest = 0.0
    for path in old:
        for field in ('label', 'prediction', 'defect_type'):
            if old[path][field] != new[path][field]:
                raise ValueError(f'{field} changed: {path}')
        a, b = float(old[path]['anomaly_score']), float(new[path]['anomaly_score'])
        if not math.isfinite(b) or abs(a-b) > 1e-5:
            raise ValueError(f'Score mismatch: {path}: {a} -> {b}')
        largest = max(largest, abs(a-b))
    old_metrics = json.loads((ROOT / 'docs/results/test_metrics.json').read_text('utf-8-sig'))
    new_metrics = json.loads((ROOT / 'outputs/baseline_v1/test_metrics.json').read_text('utf-8-sig'))
    if old_metrics != new_metrics:
        raise ValueError('Evaluation metrics changed')
    print(f'PASS: 83 images, unchanged decisions/metrics, max score difference={largest:.8f}')
    print('Also verify CLI and Streamlit show 0.474999 / anomaly for good/006.png.')


if __name__ == '__main__':
    main()
