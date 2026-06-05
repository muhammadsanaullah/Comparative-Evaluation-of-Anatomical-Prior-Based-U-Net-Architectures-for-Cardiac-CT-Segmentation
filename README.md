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
