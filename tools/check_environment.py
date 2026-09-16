import platform
import sys

import torch
import torchvision
import transformers


def main():
    print("=" * 70)
    print("SegFormer Potsdam Environment Check")
    print("=" * 70)

    print(f"Python             : {sys.version.split()[0]}")
    print(f"Platform           : {platform.platform()}")
    print(f"PyTorch            : {torch.__version__}")
    print(f"TorchVision        : {torchvision.__version__}")
    print(f"Transformers       : {transformers.__version__}")
    print(f"PyTorch CUDA       : {torch.version.cuda}")
    print(f"CUDA available     : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Stop here before training."
        )

    device = torch.device("cuda:0")

    print(f"GPU                : {torch.cuda.get_device_name(0)}")
    print(f"GPU count          : {torch.cuda.device_count()}")
    print(f"CUDA capability    : {torch.cuda.get_device_capability(0)}")

    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU memory         : {total_memory / 1024**3:.2f} GB")

    print()
    print("Running CUDA matrix multiplication test...")

    x = torch.randn(2048, 2048, device=device)
    y = torch.randn(2048, 2048, device=device)
    z = x @ y

    assert z.is_cuda
    assert torch.isfinite(z).all()

    print(f"Result shape       : {tuple(z.shape)}")
    print(f"Result device      : {z.device}")
    print("CUDA computation   : OK")

    del x, y, z
    torch.cuda.empty_cache()

    print("=" * 70)
    print("Environment check PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
