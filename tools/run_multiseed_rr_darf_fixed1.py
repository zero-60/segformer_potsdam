"""
3-seed experiment launcher.

Frozen requirements:
- only change random seed
- same split/epoch/LR/augmentation/loss/validation
- seeds:
  20260917, 20260918, 20260919

Outputs:
b2_rgbnir_fixed1_joint_robust4_seedXXXX
b2_darf_rr_joint_robust4_seedXXXX

Connect commands to existing training scripts:
tools/train_model_b2_rgbnir_fixed1_joint_robust4.py
tools/train_model_b2_darf_rr_joint_robust4.py
"""

import argparse
import json
import subprocess
from pathlib import Path

SEEDS = [20260917, 20260918, 20260919]

def run(cmd):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="python")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiments = [
        (
            "fixed1",
            "tools/train_model_b2_rgbnir_fixed1_joint_robust4.py",
            "b2_rgbnir_fixed1_joint_robust4",
        ),
        (
            "rr_darf",
            "tools/train_model_b2_darf_rr_joint_robust4.py",
            "b2_darf_rr_joint_robust4",
        ),
    ]

    manifest = []

    for seed in SEEDS:
        for name, script, prefix in experiments:
            out = f"{prefix}_seed{seed}"
            cmd = [
                args.python,
                script,
                "--seed",
                str(seed),
                "--output-dir",
                out,
            ]
            manifest.append(
                {
                    "model": name,
                    "seed": seed,
                    "output_dir": out,
                    "command": cmd,
                }
            )
            if not args.dry_run:
                run(cmd)

    Path("multiseed_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
