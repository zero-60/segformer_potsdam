#!/usr/bin/env python3
"""
Stage 10 Model A RGB evaluation entry.
"""

from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    print("Evaluation entry:")
    print("Use frozen evaluation protocol.")
    print("Checkpoint:",
          PROJECT_ROOT / "outputs/training/model_a_rgb/checkpoints/latest.pt")


if __name__ == "__main__":
    main()
