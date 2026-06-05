# Comparative-Evaluation-of-Anatomical-Prior-Based-U-Net-Architectures-for-Cardiac-CT-Segmentation

## Overview

This project investigates whether anatomical prior information can improve cardiac CT segmentation performance.

Five cardiovascular structures were segmented:

- Heart
- Aorta
- Inferior Vena Cava
- Superior Vena Cava
- Pulmonary Veins

Six additional structures were incorporated as anatomical pseudo-labels:

- Left Atrium
- Right Atrium
- Left Ventricle
- Right Ventricle
- Myocardium
- Coronary Arteries

The study compares:

- 2D U-Net
- 3D U-Net
- nnUNet
- 10 Modified Attention U-Net architectures

## Dataset

TotalSegmentator Dataset

https://github.com/wasserth/TotalSegmentator

1228 CT volumes were used.

## Architectures Evaluated

| Architecture | Description |
|-------------|-------------|
| 2D U-Net | Conventional 2D U-Net trained on axial CT slices. |
| 3D U-Net | Volumetric U-Net baseline trained on full CT volumes. |
| nnUNet | Self-configuring state-of-the-art segmentation framework using the 3D full-resolution configuration. |
| Modified-UNet 1 | Attention U-Net with multi-task pseudo-label supervision. |
| Modified-UNet 2 | Anatomical prior maps concatenated with CT inputs. |
| Modified-UNet 3 | Increased weighting assigned to pseudo-label supervision. |
| Modified-UNet 4 | Feature fusion block for CT and anatomical prior features. |
| Modified-UNet 5 | Dual-encoder architecture for independent CT and anatomical prior feature extraction. |
| Modified-UNet 6 | Dual encoders with attention-guided feature fusion. |
| Modified-UNet 7 | Deep supervision through auxiliary decoder outputs. |
| Modified-UNet 8 | Dice-Focal Loss replacing Dice Cross Entropy Loss. |
| Modified-UNet 9 | Increased pseudo-label supervision within the dual-encoder framework. |
| Modified-UNet 10 | Residual fusion block for enhanced feature refinement and reuse. |

## Quantitative Segmentation Performance

Models were ranked according to their mean DSC scores across the five target cardiovascular structures.

| Rank | Model | Mean DSC |
|------|--------|----------|
| 1 | nnUNet | 0.9270 |
| 2 | 2D U-Net | 0.8428 |
| 3 | Modified-UNet 6 | 0.7365 |
| 4 | Modified-UNet 8 | 0.7355 |
| 5 | Modified-UNet 9 | 0.7353 |
| 6 | Modified-UNet 5 | 0.7346 |
| 7 | Modified-UNet 10 | 0.7327 |
| 8 | Modified-UNet 2 | 0.7309 |
| 9 | Modified-UNet 7 | 0.7301 |
| 10 | Modified-UNet 4 | 0.7289 |
| 11 | Modified-UNet 3 | 0.7263 |
| 12 | Modified-UNet 1 | 0.7189 |
| 13 | 3D U-Net | 0.7049 |

### Key Observations

- All evaluated structures achieved their highest voxel-level recall using nnUNet.
- The aorta was the easiest structure to segment, achieving nearly 99% voxel recall and precision.
- The pulmonary veins remained the most challenging structure due to their smaller size and anatomical variability.
- Several proposed anatomical-prior-based architectures substantially improved upon the baseline 3D U-Net despite remaining below nnUNet performance.

