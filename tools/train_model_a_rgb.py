"""
Stage 10 - Model A RGB Formal Trainer

Purpose:
Train the frozen RGB baseline:
RGB -> SegFormer-B0 -> 6 classes

This entrypoint assumes all Stage 8/9 protocols are frozen.
Do not modify dataset, dataloader, evaluation, optimizer,
scheduler, AMP, or accumulation rules.
"""

from pathlib import Path
import json
import time


PROJECT_ROOT = Path("/root/autodl-tmp/projects/segformer_potsdam")
OUTPUT_DIR = PROJECT_ROOT / "outputs/training/model_a_rgb"


def load_manifest():
    manifest = {
        "stage": "Stage 10",
        "model": "SegFormer-B0",
        "input": "RGB",
        "epochs": 100,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 16,
        "optimizer": "AdamW",
        "lr": 6e-5,
        "weight_decay": 0.01,
        "amp": "fp16",
        "validation_frequency": 5,
    }
    return manifest


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()

    with open(OUTPUT_DIR / "training_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Stage 10 Model A RGB Formal Training")
    print("=" * 60)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print()
    print("Training entry initialized.")
    print("Use frozen project-native dataset/model/trainer components.")
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
