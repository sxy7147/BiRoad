# Third-party notices

This repository builds on 3D FlowMatch Actor and contains code adapted from the
projects listed below. The existing [root MIT license](LICENSE), including the
3D FlowMatch Actor copyright notice, is retained. The third-party portions
identified here retain their respective licenses and copyright notices.

The license texts in `licenses/` are unmodified copies of the upstream files.
The reference commits below identify the snapshots used to verify the source
links and retrieve license texts on 2026-10-29. They do not establish the exact
commits from which the code was originally adapted; that historical provenance
has not been established for this source snapshot.

| Project | Local files or portions | License text |
| --- | --- | --- |
| 3D FlowMatch Actor | Base implementation inherited by this repository | [MIT](LICENSE) |
| PyTorch3D | `utils/pytorch3d_transforms.py`; quaternion conversion helpers in `modeling/utils/utils.py` | [BSD-3-Clause](licenses/PyTorch3D-BSD-3-Clause.txt) |
| OpenAI CLIP | CLIP visual-backbone adaptation in `modeling/encoder/vision/clip.py` | [MIT](licenses/CLIP-MIT.txt) |
| minRF | Rectified-flow scheduler adaptation in `modeling/noise_scheduler/rectified_flow.py` | [Apache-2.0](licenses/minRF-Apache-2.0.txt) |
| RotationContinuity | Continuous 6D rotation helpers in `modeling/utils/utils.py` | [MIT](licenses/RotationContinuity-MIT.txt) |

## 3D FlowMatch Actor

- Project: <https://github.com/nickgkan/3d_flowmatch_actor>
- Reference commit: `ab7093270371964c58fd6cd9169070920c82992c`.
- Copyright (c) 2025 Nikolaos Gkanatsios, Jiahe Xu, Matthew Bronars,
  Arsalan Mousavian, Tsung-Wei Ke and Katerina Fragkiadaki.
- License and disclaimer: [LICENSE](LICENSE).

## PyTorch3D

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Applies to the adapted rotation-conversion code in
`utils/pytorch3d_transforms.py` and the `quaternion_to_matrix`,
`_sqrt_positive_part`, and `matrix_to_quaternion` helpers in
`modeling/utils/utils.py`. The BSD text linked here supplies the terms
referenced by the existing PyTorch3D source header. Local adaptations include
tensor-operation changes in the quaternion helpers.

- Project: <https://github.com/facebookresearch/pytorch3d>
- Source reference: [pytorch3d/transforms/rotation_conversions.py](https://github.com/facebookresearch/pytorch3d/blob/204fe5831b3a5bbe079f09d72afb4515178f8892/pytorch3d/transforms/rotation_conversions.py).
- Reference commit: `204fe5831b3a5bbe079f09d72afb4515178f8892`.
- License and disclaimer: [PyTorch3D-BSD-3-Clause.txt](licenses/PyTorch3D-BSD-3-Clause.txt).
- Upstream license: [original text](https://raw.githubusercontent.com/facebookresearch/pytorch3d/204fe5831b3a5bbe079f09d72afb4515178f8892/LICENSE).
- License-file SHA-256: `6da05663049417a91409fa349bfad5778ed56c9ef68226dde28dffd541481579`.

## OpenAI CLIP

Copyright (c) 2021 OpenAI.

Applies to the CLIP visual-backbone adaptation in
`modeling/encoder/vision/clip.py`, including the modified ResNet forward path
that exposes intermediate feature maps.

- Project: <https://github.com/openai/CLIP>
- Source reference: [clip/model.py](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/model.py).
- Reference commit: `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`.
- License and disclaimer: [CLIP-MIT.txt](licenses/CLIP-MIT.txt).
- Upstream license: [original text](https://raw.githubusercontent.com/openai/CLIP/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/LICENSE).
- License-file SHA-256: `987e63b32f6c89ff5160e429458a872ff048e6860b590a3912e938f9da8f14db`.

## minRF

Project attribution: Simo Ryu and minRF contributors.

**Adaptation notice:** `modeling/noise_scheduler/rectified_flow.py` is a
modified adaptation of the referenced minRF rectified-flow implementation.
It exposes a standalone `RFScheduler`, separates noise-step sampling, noise
interpolation, target preparation and Euler integration, and supports
configurable noise samplers and inference timesteps. These descriptions
document the existing adaptation; the license addition does not change it.

The referenced minRF repository snapshot contains no separate upstream
`NOTICE` file. Its complete Apache License 2.0 text, including the standard
appendix, is reproduced unchanged in the linked file.

- Project: <https://github.com/cloneofsimo/minRF>
- Source reference: [advanced/main_t2i.py](https://github.com/cloneofsimo/minRF/blob/4fc10e0cc8ba976152c7936a1af9717209f03e18/advanced/main_t2i.py).
- Reference commit: `4fc10e0cc8ba976152c7936a1af9717209f03e18`.
- License and disclaimer: [minRF-Apache-2.0.txt](licenses/minRF-Apache-2.0.txt).
- Upstream license: [original text](https://raw.githubusercontent.com/cloneofsimo/minRF/4fc10e0cc8ba976152c7936a1af9717209f03e18/LICENSE).
- License-file SHA-256: `43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1`.

## RotationContinuity

Copyright (c) 2019 Yi_Zhou.

Applies to the continuous 6D rotation representation helpers in
`modeling/utils/utils.py`, including `normalize_vector`, `cross_product`,
and `compute_rotation_matrix_from_ortho6d`.

Associated paper: *On the Continuity of Rotation Representations in Neural
Networks*, <https://arxiv.org/abs/1812.07035>.

- Project: <https://github.com/papagina/RotationContinuity>
- Source reference: [sanity_test/code/tools.py](https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/sanity_test/code/tools.py).
- Reference commit: `758b0ce551c06372cab7022d4c0bdf331c89c696`.
- License and disclaimer: [RotationContinuity-MIT.txt](licenses/RotationContinuity-MIT.txt).
- Upstream license: [original text](https://raw.githubusercontent.com/papagina/RotationContinuity/758b0ce551c06372cab7022d4c0bdf331c89c696/LICENSE).
- License-file SHA-256: `b596cd3d430d997369778ce790516d2d10339a92cd22f002298e53c786522f04`.

## Scope

These notices cover the source adaptations identified above. Separately
installed dependencies and externally obtained models, datasets and simulator
assets retain their own license terms. These notices do not grant new rights
in those external materials.
