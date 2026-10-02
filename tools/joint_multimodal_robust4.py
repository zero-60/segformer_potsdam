#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint RGB+NIR Robust-4 degradation for Potsdam.

Scientific intent
-----------------
Old protocol:
    RGB degraded, NIR always clean.

New joint protocol:
    If a sample is selected for degradation, RGB and NIR are BOTH degraded
    by the same degradation family and the same base severity.

Families:
    1. Gaussian noise
       - Same sigma severity for RGB and NIR.
       - Independent noise realization per modality/channel.

    2. Gaussian blur
       - Same blur sigma for RGB and NIR.

    3. Underexposure
       - Same multiplicative attenuation for RGB and NIR.

    4. Atmospheric fog
       - Shared low-frequency transmission field.
       - NIR is degraded too, but less strongly than visible RGB by default:
             t_nir = 1 - nir_fog_scatter_ratio * (1 - t_rgb)
         Default nir_fog_scatter_ratio = 0.65.
       - This is a controlled spectral approximation, NOT a calibrated
         atmospheric radiative-transfer model.

Inputs are normalized tensors:
    RGB: ImageNet mean/std.
    NIR: dataset-specific frozen train-only mean/std.

The function first denormalizes both modalities to [0,1], applies degradation,
then re-normalizes each modality with its own statistics.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


ROBUST4_FAMILIES = (
    "gaussian_noise",
    "gaussian_blur",
    "underexposure",
    "fog",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_NIR_FOG_SCATTER_RATIO = 0.65


def _rgb_denormalize(rgb: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(
        IMAGENET_MEAN,
        device=rgb.device,
        dtype=rgb.dtype,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        device=rgb.device,
        dtype=rgb.dtype,
    ).view(1, 3, 1, 1)

    return torch.clamp(
        rgb * std + mean,
        0.0,
        1.0,
    )


def _rgb_normalize(rgb01: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(
        IMAGENET_MEAN,
        device=rgb01.device,
        dtype=rgb01.dtype,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        device=rgb01.device,
        dtype=rgb01.dtype,
    ).view(1, 3, 1, 1)

    return (
        rgb01 - mean
    ) / std


def _nir_denormalize(
    nir: torch.Tensor,
    *,
    nir_mean: float,
    nir_std: float,
) -> torch.Tensor:
    return torch.clamp(
        nir * float(nir_std) + float(nir_mean),
        0.0,
        1.0,
    )


def _nir_normalize(
    nir01: torch.Tensor,
    *,
    nir_mean: float,
    nir_std: float,
) -> torch.Tensor:
    if not math.isfinite(float(nir_std)) or float(nir_std) <= 0:
        raise ValueError(
            f"nir_std must be finite and > 0, got {nir_std}"
        )

    return (
        nir01 - float(nir_mean)
    ) / float(nir_std)


def _gaussian_blur_single(
    image: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    image: [C,H,W], any C >= 1
    """
    sigma = float(sigma)

    if sigma <= 0:
        return image

    radius = max(
        1,
        int(
            math.ceil(
                3.0 * sigma
            )
        ),
    )

    kernel_size = min(
        2 * radius + 1,
        31,
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    radius = kernel_size // 2

    coords = torch.arange(
        -radius,
        radius + 1,
        device=image.device,
        dtype=image.dtype,
    )

    kernel_1d = torch.exp(
        -(
            coords
            * coords
        )
        / (
            2.0
            * sigma
            * sigma
        )
    )

    kernel_1d = (
        kernel_1d
        / kernel_1d.sum()
    )

    kernel_2d = (
        kernel_1d[:, None]
        * kernel_1d[None, :]
    )

    channels = int(
        image.shape[0]
    )

    weight = (
        kernel_2d
        .view(
            1,
            1,
            kernel_size,
            kernel_size,
        )
        .expand(
            channels,
            1,
            kernel_size,
            kernel_size,
        )
        .contiguous()
    )

    x = image.unsqueeze(
        0
    )

    x = F.pad(
        x,
        (
            radius,
            radius,
            radius,
            radius,
        ),
        mode="reflect",
    )

    y = F.conv2d(
        x,
        weight,
        groups=channels,
    )

    return y.squeeze(
        0
    )


def _low_frequency_field(
    *,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    yy = torch.linspace(
        0.0,
        1.0,
        h,
        device=device,
        dtype=dtype,
    ).view(
        h,
        1,
    )

    xx = torch.linspace(
        0.0,
        1.0,
        w,
        device=device,
        dtype=dtype,
    ).view(
        1,
        w,
    )

    field = torch.zeros(
        (
            h,
            w,
        ),
        device=device,
        dtype=dtype,
    )

    weights = (
        0.34,
        0.27,
        0.22,
        0.17,
    )

    for weight in weights:
        fx = (
            0.35
            + 1.25
            * torch.rand(
                (),
                device=device,
                dtype=dtype,
            )
        )

        fy = (
            0.35
            + 1.25
            * torch.rand(
                (),
                device=device,
                dtype=dtype,
            )
        )

        phase = (
            2.0
            * math.pi
            * torch.rand(
                (),
                device=device,
                dtype=dtype,
            )
        )

        sign = torch.where(
            torch.rand(
                (),
                device=device,
            )
            < 0.5,
            torch.tensor(
                -1.0,
                device=device,
                dtype=dtype,
            ),
            torch.tensor(
                1.0,
                device=device,
                dtype=dtype,
            ),
        )

        field = (
            field
            + float(
                weight
            )
            * sign
            * torch.sin(
                2.0
                * math.pi
                * (
                    fx
                    * xx
                    + fy
                    * yy
                )
                + phase
            )
        )

    return torch.clamp(
        field,
        -1.0,
        1.0,
    )


def _joint_fog_single(
    rgb01: torch.Tensor,
    nir01: torch.Tensor,
    *,
    severity: float,
    nir_fog_scatter_ratio: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[str, float],
]:
    """
    Shared fog event for aligned RGB/NIR crop.

    RGB uses:
        t_rgb = mean_t_rgb + variation_rgb * field

    NIR uses the SAME field but reduced attenuation:
        t_nir = 1 - ratio * (1 - t_rgb)

    ratio=1.0 -> same attenuation as RGB.
    ratio<1.0 -> NIR is less affected, but never left clean.
    """
    severity = float(
        min(
            max(
                severity,
                0.0,
            ),
            1.0,
        )
    )

    ratio = float(
        nir_fog_scatter_ratio
    )

    if not (
        0.0
        < ratio
        <= 1.0
    ):
        raise ValueError(
            "nir_fog_scatter_ratio must be in (0,1]."
        )

    _, h, w = rgb01.shape

    field = _low_frequency_field(
        h=h,
        w=w,
        device=rgb01.device,
        dtype=rgb01.dtype,
    )

    mean_t_rgb = (
        0.90
        - 0.50
        * severity
    )

    variation_rgb = (
        0.04
        + 0.11
        * severity
    )

    t_rgb = (
        mean_t_rgb
        + variation_rgb
        * field
    )

    t_rgb = torch.clamp(
        t_rgb,
        0.20,
        0.98,
    )

    # NIR is still degraded, but less strongly by default.
    t_nir = (
        1.0
        - ratio
        * (
            1.0
            - t_rgb
        )
    )

    t_nir = torch.clamp(
        t_nir,
        0.20,
        0.995,
    )

    atmosphere = (
        0.94
        + 0.05
        * torch.rand(
            (),
            device=rgb01.device,
            dtype=rgb01.dtype,
        )
    )

    rgb_out = (
        rgb01
        * t_rgb.unsqueeze(
            0
        )
        + atmosphere
        * (
            1.0
            - t_rgb.unsqueeze(
                0
            )
        )
    )

    nir_out = (
        nir01
        * t_nir.unsqueeze(
            0
        )
        + atmosphere
        * (
            1.0
            - t_nir.unsqueeze(
                0
            )
        )
    )

    return (
        torch.clamp(
            rgb_out,
            0.0,
            1.0,
        ),
        torch.clamp(
            nir_out,
            0.0,
            1.0,
        ),
        {
            "mean_t_rgb": float(
                t_rgb.mean().item()
            ),
            "mean_t_nir": float(
                t_nir.mean().item()
            ),
        },
    )


def degrade_rgb_nir_batch_joint_robust4(
    rgb_normalized: torch.Tensor,
    nir_normalized: torch.Tensor,
    *,
    probability: float,
    nir_mean: float,
    nir_std: float,
    nir_fog_scatter_ratio: float = DEFAULT_NIR_FOG_SCATTER_RATIO,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[str, int],
    torch.Tensor,
    Dict[str, float],
]:
    """
    Apply a paired multimodal Robust-4 event.

    Returns
    -------
    rgb_aug_normalized
    nir_aug_normalized
    family_counts
    base_severity [B], 0 for clean
    diagnostics
    """
    if (
        rgb_normalized.ndim
        != 4
        or rgb_normalized.shape[
            1
        ]
        != 3
    ):
        raise RuntimeError(
            f"RGB must be [B,3,H,W], got {tuple(rgb_normalized.shape)}"
        )

    if (
        nir_normalized.ndim
        != 4
        or nir_normalized.shape[
            1
        ]
        != 1
    ):
        raise RuntimeError(
            f"NIR must be [B,1,H,W], got {tuple(nir_normalized.shape)}"
        )

    if (
        rgb_normalized.shape[
            0
        ]
        != nir_normalized.shape[
            0
        ]
        or rgb_normalized.shape[
            -2:
        ]
        != nir_normalized.shape[
            -2:
        ]
    ):
        raise RuntimeError(
            "RGB/NIR batch is not aligned."
        )

    probability = float(
        probability
    )

    if not (
        0.0
        <= probability
        <= 1.0
    ):
        raise ValueError(
            "probability must be in [0,1]."
        )

    with torch.autocast(
        device_type=rgb_normalized.device.type,
        enabled=False,
    ):
        rgb01 = _rgb_denormalize(
            rgb_normalized.float()
        )

        nir01 = _nir_denormalize(
            nir_normalized.float(),
            nir_mean=nir_mean,
            nir_std=nir_std,
        )

        rgb_out = rgb01.clone()
        nir_out = nir01.clone()

        batch = int(
            rgb01.shape[
                0
            ]
        )

        severities = torch.zeros(
            (
                batch,
            ),
            device=rgb01.device,
            dtype=torch.float32,
        )

        counts = {
            "clean": 0,
            "gaussian_noise": 0,
            "gaussian_blur": 0,
            "underexposure": 0,
            "fog": 0,
        }

        fog_t_rgb_sum = 0.0
        fog_t_nir_sum = 0.0
        fog_count = 0

        for index in range(
            batch
        ):
            do_corrupt = bool(
                (
                    torch.rand(
                        (),
                        device=rgb01.device,
                    )
                    < probability
                )
                .item()
            )

            if not do_corrupt:
                counts[
                    "clean"
                ] += 1
                continue

            severity = float(
                (
                    0.10
                    + 0.90
                    * torch.rand(
                        (),
                        device=rgb01.device,
                    )
                )
                .item()
            )

            family = int(
                torch.randint(
                    low=0,
                    high=4,
                    size=(),
                    device=rgb01.device,
                )
                .item()
            )

            if family == 0:
                sigma_255 = (
                    5.0
                    + 45.0
                    * severity
                )

                sigma = (
                    sigma_255
                    / 255.0
                )

                # Independent sensor noise realization in RGB and NIR.
                rgb_out[
                    index
                ] = torch.clamp(
                    rgb_out[
                        index
                    ]
                    + torch.randn_like(
                        rgb_out[
                            index
                        ]
                    )
                    * sigma,
                    0.0,
                    1.0,
                )

                nir_out[
                    index
                ] = torch.clamp(
                    nir_out[
                        index
                    ]
                    + torch.randn_like(
                        nir_out[
                            index
                        ]
                    )
                    * sigma,
                    0.0,
                    1.0,
                )

                counts[
                    "gaussian_noise"
                ] += 1

            elif family == 1:
                sigma = (
                    0.5
                    + 3.5
                    * severity
                )

                rgb_out[
                    index
                ] = _gaussian_blur_single(
                    rgb_out[
                        index
                    ],
                    sigma,
                )

                nir_out[
                    index
                ] = _gaussian_blur_single(
                    nir_out[
                        index
                    ],
                    sigma,
                )

                counts[
                    "gaussian_blur"
                ] += 1

            elif family == 2:
                alpha = (
                    1.0
                    - 0.60
                    * severity
                )

                rgb_out[
                    index
                ] = torch.clamp(
                    rgb_out[
                        index
                    ]
                    * alpha,
                    0.0,
                    1.0,
                )

                nir_out[
                    index
                ] = torch.clamp(
                    nir_out[
                        index
                    ]
                    * alpha,
                    0.0,
                    1.0,
                )

                counts[
                    "underexposure"
                ] += 1

            else:
                (
                    rgb_fog,
                    nir_fog,
                    fog_diag,
                ) = _joint_fog_single(
                    rgb_out[
                        index
                    ],
                    nir_out[
                        index
                    ],
                    severity=severity,
                    nir_fog_scatter_ratio=(
                        nir_fog_scatter_ratio
                    ),
                )

                rgb_out[
                    index
                ] = rgb_fog

                nir_out[
                    index
                ] = nir_fog

                fog_t_rgb_sum += float(
                    fog_diag[
                        "mean_t_rgb"
                    ]
                )

                fog_t_nir_sum += float(
                    fog_diag[
                        "mean_t_nir"
                    ]
                )

                fog_count += 1

                counts[
                    "fog"
                ] += 1

            severities[
                index
            ] = severity

        rgb_model = _rgb_normalize(
            rgb_out
        )

        nir_model = _nir_normalize(
            nir_out,
            nir_mean=nir_mean,
            nir_std=nir_std,
        )

    diagnostics = {
        "fog_samples": int(
            fog_count
        ),
        "fog_mean_t_rgb": (
            fog_t_rgb_sum
            / fog_count
            if fog_count
            > 0
            else float(
                "nan"
            )
        ),
        "fog_mean_t_nir": (
            fog_t_nir_sum
            / fog_count
            if fog_count
            > 0
            else float(
                "nan"
            )
        ),
        "nir_fog_scatter_ratio": float(
            nir_fog_scatter_ratio
        ),
    }

    return (
        rgb_model,
        nir_model,
        counts,
        severities,
        diagnostics,
    )
