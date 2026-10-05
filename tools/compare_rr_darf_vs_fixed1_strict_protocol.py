"""
RR-DARF vs Fixed g=1.0 strict same-protocol comparison.

Implements the frozen protocol from the experiment instruction:
- Same Potsdam validation tiles
- Raw 6000x6000 tile degradation before crop
- 512 sliding window
- overlap mean-logit fusion
- global confusion matrix
- Clean + Noise/Blur/Underexposure/Fog L1-L3
- condition-wise RR-DARF - Fixed1
- per-class IoU and stress reporting

Adapt import paths to existing repository code:
segformer_potsdam/
"""

import argparse
import json
from pathlib import Path

CONDITIONS = [
    "clean",
    "noise_l1", "noise_l2", "noise_l3",
    "blur_l1", "blur_l2", "blur_l3",
    "underexposure_l1", "underexposure_l2", "underexposure_l3",
    "fog_l1", "fog_l2", "fog_l3",
]

def load_checkpoint(model, checkpoint):
    # TODO: connect repository checkpoint loader
    return model

def build_model(model_type):
    # TODO:
    # fixed1 -> models with Fi=FRGB+1.0*Adapter(FNIR)
    # rr_darf -> models/segformer_b2_darf_rr.py
    raise NotImplementedError

def validate_condition(model, condition, args):
    """
    Must call the existing frozen validation pipeline:
    validate_model_b2_darf_rr.py / validate_model_b2_rgbnir_fixed1_joint_robust4.py
    """
    raise NotImplementedError

def compare(results_rr, results_fixed):
    out = {}
    for k in results_rr:
        if isinstance(results_rr[k], (int, float)):
            out[k] = results_rr[k] - results_fixed[k]
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rr-checkpoint", required=True)
    parser.add_argument("--fixed-checkpoint", required=True)
    parser.add_argument("--output", default="rr_vs_fixed1_results.json")
    args = parser.parse_args()

    all_results = {}
    for model_name, ckpt in [
        ("RR-DARF", args.rr_checkpoint),
        ("Fixed-g1.0", args.fixed_checkpoint),
    ]:
        model = build_model(model_name)
        model = load_checkpoint(model, ckpt)
        all_results[model_name] = {}
        for c in CONDITIONS:
            all_results[model_name][c] = validate_condition(model, c, args)

    comparison = {}
    for c in CONDITIONS:
        comparison[c] = compare(
            all_results["RR-DARF"][c],
            all_results["Fixed-g1.0"][c],
        )

    Path(args.output).write_text(
        json.dumps(
            {
                "protocol": "Joint Robust4 strict validation",
                "conditions": CONDITIONS,
                "results": all_results,
                "RR_minus_Fixed1": comparison,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
