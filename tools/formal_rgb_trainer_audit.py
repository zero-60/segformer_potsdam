"""
Stage 9 Formal RGB Trainer Audit

Purpose:
- Verify trainer artifact exists
- Verify frozen protocol manifest references
- Perform lightweight import check
- No training is executed
"""

from pathlib import Path
import importlib.util
import json
import sys


PROJECT_ROOT = Path("/root/autodl-tmp/projects/segformer_potsdam")
TRAINER = PROJECT_ROOT / "tools" / "formal_rgb_trainer.py"
MANIFEST = PROJECT_ROOT / "tools" / "trainer_manifest.json"


def check():
    result = {}

    result["trainer_exists"] = TRAINER.exists()
    result["manifest_exists"] = MANIFEST.exists()

    if TRAINER.exists():
        spec = importlib.util.spec_from_file_location(
            "formal_rgb_trainer",
            TRAINER
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result["import_success"] = hasattr(module, "FormalRGBTrainer")

    if MANIFEST.exists():
        with open(MANIFEST, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        result["stage"] = manifest.get("stage")
        result["version"] = manifest.get("version")

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not all([
        result.get("trainer_exists"),
        result.get("manifest_exists"),
        result.get("import_success"),
    ]):
        raise SystemExit("AUDIT FAILED")

    print("STAGE9_FORMAL_RGB_TRAINER_AUDIT_PASS")


if __name__ == "__main__":
    check()
