import json
import sys
from pathlib import Path
import torch
import torchvision

def main():
    assert torch.cuda.is_available(), 'CUDA is unavailable'
    torch.manual_seed(42)
    device = torch.device('cuda')
    x = torch.randn(4, 3, 64, 64, device=device)
    model = torchvision.models.resnet18(weights=None, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    targets = torch.tensor([0, 1, 0, 1], device=device)
    optimizer.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.isfinite(loss), 'Loss is not finite'
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    a = torch.randn(128, 128)
    cpu_result = a @ a.T
    gpu_result = (a.to(device) @ a.to(device).T).cpu()
    torch.testing.assert_close(gpu_result, cpu_result, rtol=1e-3, atol=1e-3)
    torch.cuda.synchronize()
    report = dict(python=sys.version, executable=sys.executable, torch=torch.__version__, torchvision=torchvision.__version__, cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0), vram_gib=round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), loss=loss.item(), status='PASS')
    output = Path(__file__).resolve().parents[1] / 'outputs' / 'environment_check.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
