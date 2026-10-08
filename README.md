<div align="center">

<img src="docs/logo.svg" alt="Unsupervised Image Restoration & Super-Resolution logo" width="110" />

# Unsupervised Image Restoration & Super-Resolution

**Training-Free ×8 Super-Resolution for Ancient Kannada Palm-Leaf Manuscripts**

Recovers high-resolution detail from a single low-resolution scan with Deep Image Prior (DIP): an untrained network is fitted to that one image, so no training dataset is needed. Measured against nearest, bicubic and sharpened upsampling with PSNR and SSIM.

<a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white" alt="Python 3.8+" /></a>
<a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-DL_Framework-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" /></a>
<a href="https://github.com/DmitryUlyanov/deep-image-prior"><img src="https://img.shields.io/badge/Deep_Image_Prior-CVPR_2018-6A5ACD" alt="Deep Image Prior" /></a>
<a href="https://numpy.org"><img src="https://img.shields.io/badge/NumPy-Array_Ops-013243?logo=numpy&logoColor=white" alt="NumPy" /></a>
<br />
<a href="https://python-pillow.org"><img src="https://img.shields.io/badge/Pillow-Image_IO-3776AB" alt="Pillow" /></a>
<a href="https://scikit-image.org"><img src="https://img.shields.io/badge/scikit--image-PSNR%20%2F%20SSIM-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-image" /></a>
<a href="https://pandas.pydata.org"><img src="https://img.shields.io/badge/pandas-Metrics_Export-150458?logo=pandas&logoColor=white" alt="pandas" /></a>
<a href="https://colab.research.google.com"><img src="https://img.shields.io/badge/Google_Colab-GPU_Runtime-F9AB00?logo=googlecolab&logoColor=white" alt="Google Colab" /></a>
<a href="https://git-lfs.com"><img src="https://img.shields.io/badge/Git_LFS-Images_%26_Notebook-F64935?logo=git&logoColor=white" alt="Git LFS" /></a>

<p>
  <a href="#demo-video"><strong>Demo Video</strong></a> ·
  <a href="#results">Results</a> ·
  <a href="#overview">Overview</a> ·
  <a href="#repository-structure">Repository Structure</a> ·
  <a href="#method">Method</a> ·
  <a href="#pipeline">Pipeline</a> ·
  <a href="#metrics">Metrics</a> ·
  <a href="#end-to-end-quick-start">Quick Start</a>
</p>

</div>

---

## Demo Video

https://github.com/user-attachments/assets/88c0b924-acbc-4a16-8a57-5e249f093ba5

A 2-minute walkthrough: the no-training-data problem, the Deep Image Prior loop, the network configuration, the pipeline, PSNR during optimization, results on all 8 manuscripts, and zoomed visual comparisons.

## Overview

Digitised palm-leaf manuscripts are often low resolution, faded, and noisy, and there is no paired low/high-resolution dataset to train a supervised super-resolution model on. This project uses **Deep Image Prior** ([Ulyanov et al., CVPR 2018](https://arxiv.org/abs/1711.10925)), which needs no training data. The *structure of a convolutional network* acts as the image prior: an untrained network is optimised so that a **downsampled** version of its output matches the observed low-resolution image, and its full-resolution output is taken as the super-resolved result.

What the repository contains:
- A batch DIP super-resolution pipeline over a folder of palm-leaf images (`DEEP_IMAGE_PRIOR_kannada_palm_leaf_8image.ipynb`)
- A single-image reference script adapted from the official DIP super-resolution demo (`zebra.py`)
- Comparisons against classical **nearest**, **bicubic**, and **sharpened-bicubic** upsampling
- Per-image **PSNR** and **SSIM** metrics, best-iteration tracking, and runtime, exported to CSV
- Plots: PSNR per image, SSIM per image, and PSNR over optimisation iterations
- Compatibility patches for running the original DIP code on current Pillow and scikit-image versions

## Repository Structure

```text
unsupervised_imgRestoration_SR/
├── DEEP_IMAGE_PRIOR_kannada_palm_leaf_8image.ipynb   # main pipeline: batch ×8 DIP SR + metrics + plots
├── zebra.py                                         # single-image DIP SR demo (Colab export)
├── ancient_kannada_palm_leaf/                       # 15 palm-leaf manuscript images (1-img.jpg … 15-img.jpg)
├── sr_miniDataset/
│   └── test/
│       ├── high_res/                                # 99 HR reference images
│       └── low_res/                                 # 99 matching LR images
├── dip_metrics_8images.csv                          # PSNR / SSIM / runtime for 8 manuscript images
├── psnr-compare.png                                 # DIP vs bicubic PSNR per image
├── ssim.png                                         # SSIM per image
├── psnr_evolution.png                               # PSNR (LR & HR) over iterations
├── nature.jpg                                       # extra natural test image
├── Chain-of-Zoom.pdf                                # related-work reference paper
├── docs/logo.svg                                    # README logo
├── .gitattributes                                   # Git LFS tracking rules
└── README.md
```

## Requirements

Python 3.8+ and a CUDA GPU are recommended. The notebook was developed on **Google Colab** with a GPU runtime. On CPU, each image takes far longer than the roughly 8.5 minutes it takes on a GPU.

```bash
pip install torch torchvision numpy scipy matplotlib pillow scikit-image pandas
```

The pipeline relies on the model and utility code from the official [DmitryUlyanov/deep-image-prior](https://github.com/DmitryUlyanov/deep-image-prior) repository (`models/`, `utils/sr_utils.py`, `models/downsampler.py`). The notebook clones it automatically; see [Pipeline](#pipeline).

## Data in GitHub (Git LFS)

Images, the PDF, the CSV, and the notebook are stored with **Git LFS**.

- Tracked patterns: `*.jpg`, `*.png`, `*.pdf`, `*.csv`, `*.ipynb`
- Config file: `.gitattributes`

When cloning, run:

```bash
git lfs install
git clone https://github.com/mainak569/unsupervised_imgRestoration_SR.git
cd unsupervised_imgRestoration_SR
git lfs pull
```

If Git LFS is not installed, these files appear as small text pointer files instead of real images or notebooks.

## Data

### Ancient Kannada palm-leaf manuscripts
`ancient_kannada_palm_leaf/` holds 15 manuscript photographs. The reported experiment uses `1-img.jpg` to `8-img.jpg`.

For each image, the high-resolution (HR) ground truth is the image itself, and the low-resolution (LR) input is created by **downsampling it ×8** with `load_LR_HR_imgs_sr(...)`. This lets PSNR and SSIM be measured against a known reference.

### SR mini dataset
`sr_miniDataset/test/` contains 99 paired `high_res/` and `low_res/` PNGs. It is included for further experiments; the current notebook does not use it.

## Method

### Deep Image Prior for super-resolution

Given an LR image $x_0$, a network $f_\theta$ takes a fixed random noise tensor $z$ and is optimised so that

$$
\theta^\* = \arg\min_\theta \; \lVert d(f_\theta(z)) - x_0 \rVert^2, \qquad \hat{x}_{HR} = f_{\theta^\*}(z)
$$

where $d(\cdot)$ is a fixed Lanczos downsampler. The network learns natural image structure faster than it learns noise, so stopping the optimisation at the right point gives a sharp HR image without any training set.

### Configuration

| Setting | Value |
|---|---|
| Scale factor | ×8 (×4 also supported: 2000 iters, `reg_noise_std = 0.03`) |
| Network | `skip` encoder–decoder (U-Net-like), 5 scales, 128 down/up channels, 4 skip channels |
| Upsampling / padding | bilinear / reflection |
| Input | 32-channel uniform noise `z` at HR size, perturbed each step (`reg_noise_std = 0.05`) |
| Downsampler | `lanczos2` kernel, phase 0.5 |
| Loss | MSE between downsampled output and LR input (TV weight `0.0`) |
| Optimiser | Adam, `lr = 0.01` |
| Iterations | 4000, with PSNR-based early stopping (patience 200) |
| Preprocessing | images resized so the longest side is ≤ 512 px (avoids GPU out-of-memory) |

## Pipeline

Notebook: `DEEP_IMAGE_PRIOR_kannada_palm_leaf_8image.ipynb`

1. Mounts Google Drive and points `base_path` at the palm-leaf folder.
2. Clones the official DIP repo and applies compatibility patches to `utils/sr_utils.py`:
   - `from PIL import images` → `from PIL import Image`
   - `Image.ANTIALIAS` → `Image.Resampling.LANCZOS` (removed in Pillow 10)
   - `skimage.measure.compare_psnr` → `skimage.metrics.peak_signal_noise_ratio`
3. For each image:
   - resizes it to ≤ 512 px with `safe_open_and_resize`, builds the HR/LR pair, and forces RGB
   - computes the nearest, bicubic, and sharpened-bicubic baselines
   - optimises a fresh DIP network, logging PSNR for LR and HR every iteration and printing elapsed time and ETA
   - saves the SR output to `results/<name>_SR_DIP.png`
   - records the best and final PSNR, the best iteration, baseline PSNRs, SSIM, hyper-parameters, and runtime
4. Writes all rows to `dip_metrics.csv` and plots PSNR per image, SSIM per image, PSNR over iterations, and side-by-side views of ground truth, input noise, and DIP output.

`zebra.py` is the same algorithm for a single image (`data/sr/zebra_GT.png` from the DIP repo, ×4 by default). It is a Colab export and contains `!` shell lines, so run it in a notebook cell or remove those lines first.

## Metrics

Computed with `scikit-image` against the HR ground truth:

- **PSNR (dB)**: `peak_signal_noise_ratio`, tracked for both:
  - `PSNR_HR`: DIP output vs ground truth (the reported quality)
  - `PSNR_LR`: downsampled DIP output vs LR input (how well the network fits the data)
- **SSIM**: `structural_similarity` with `channel_axis=-1`, `data_range=1.0`, `win_size=7` (falls back to 3 for very small images)

CSV columns in `dip_metrics_8images.csv`:
`image_name, scale_factor, best_psnr_hr, best_iter, final_psnr_hr, final_psnr_lr, bicubic_psnr, nearest_psnr, sharp_psnr, lr, reg_noise_std, tv_weight, num_iter, runtime_sec, ssim_hr`

## Results

×8 super-resolution on 8 palm-leaf images (from `dip_metrics_8images.csv`):

| Image | Nearest | Bicubic | Sharp | **DIP (best)** | Δ vs Bicubic | SSIM | Best iter | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `1-img.jpg` | 23.86 | 24.61 | 24.63 | **25.18** | +0.57 | 0.563 | 3958 | 522 |
| `2-img.jpg` | 22.12 | 22.44 | 22.44 | **22.64** | +0.20 | 0.545 | 3857 | 531 |
| `3-img.jpg` | 15.84 | 16.48 | 16.50 | **16.85** | +0.37 | 0.537 | 3088 | 526 |
| `4-img.jpg` | 29.12 | 30.09 | 30.10 | **30.28** | +0.18 | 0.661 | 3979 | 531 |
| `5-img.jpg` | 26.89 | 27.26 | 27.26 | **27.41** | +0.15 | 0.779 | 3922 | 519 |
| `6-img.jpg` | 23.21 | 23.68 | 23.70 | **23.96** | +0.27 | 0.665 | 3845 | 532 |
| `7-img.jpg` | 25.52 | 26.28 | 26.30 | **27.14** | +0.86 | 0.702 | 2832 | 393 |
| `8-img.jpg` | 23.55 | 24.00 | 24.02 | **24.62** | +0.62 | 0.619 | 3654 | 520 |
| **Mean** | 23.76 | 24.35 | 24.37 | **24.76** | **+0.40** | **0.634** | — | 509 |

All PSNR values are in dB. DIP beats every classical baseline on **all 8 images** without any training data.

<p align="center">
  <img src="psnr-compare.png" alt="Best DIP PSNR vs bicubic PSNR per image" width="85%" />
</p>

<p align="center">
  <img src="psnr_evolution.png" alt="PSNR evolution over iterations" width="48%" />
  <img src="ssim.png" alt="SSIM per image" width="48%" />
</p>

`PSNR_LR` keeps rising to about 46 dB as the network fits the LR input more and more closely. `PSNR_HR` levels off near 24–25 dB after about 1000 iterations. This gap shows why the number of iterations and early stopping matter in DIP.

## End-to-End Quick Start

1. Open `DEEP_IMAGE_PRIOR_kannada_palm_leaf_8image.ipynb` in **Google Colab** and choose a **GPU** runtime.
2. Copy `ancient_kannada_palm_leaf/` to your Google Drive at `MyDrive/ancient_kannada_palm_leaf/`, or change `base_path` / `folder_path`.
3. Choose the scale with `factor = 8` (or `4`).
4. Run all cells. The notebook clones the DIP repo, patches it, and processes every image.
5. Collect the outputs:
   - `results/<name>_SR_DIP.png`: super-resolved images
   - `dip_metrics.csv`: per-image metrics (downloaded automatically at the end)

To run locally instead:

```bash
git clone https://github.com/DmitryUlyanov/deep-image-prior
cp -r deep-image-prior/{models,utils} .
sed -i "s/Image.ANTIALIAS/Image.Resampling.LANCZOS/" utils/sr_utils.py
jupyter notebook DEEP_IMAGE_PRIOR_kannada_palm_leaf_8image.ipynb   # then set base_path to ./ancient_kannada_palm_leaf/
```

## Known Limitations

- Each image is optimised from scratch: about 8.5 minutes per image for ×8 on a Colab GPU.
- The improvement over bicubic is modest (mean +0.40 dB), and SSIM is about 0.63 at ×8.
- `safe_open_and_resize` **overwrites the source images in place** when they are larger than 512 px. Work on a copy.
- Paths (`/content/drive/MyDrive/...`), the scale factor, and hyper-parameters are hardcoded in notebook cells.
- The early-stopping check uses PSNR against the HR ground truth, which is not available for truly unseen low-resolution scans.
- `sr_miniDataset/` and `nature.jpg` are included but not yet evaluated.

## Suggested Next Improvements

- Add a `requirements.txt` and vendor or pin the DIP `models/` and `utils/` code.
- Turn the notebook into a CLI script with arguments for the input folder, scale, and iterations.
- Replace ground-truth-based early stopping with a no-reference criterion, such as a running-variance or ES-WMV style rule.
- Evaluate on `sr_miniDataset/` and add denoising and inpainting variants of DIP.
- Compare against Chain-of-Zoom or other zero-shot and diffusion-based SR methods.

## References

- D. Ulyanov, A. Vedaldi, V. Lempitsky. *Deep Image Prior*. CVPR 2018. [arXiv:1711.10925](https://arxiv.org/abs/1711.10925) · [code](https://github.com/DmitryUlyanov/deep-image-prior)
- *Chain-of-Zoom* (included as `Chain-of-Zoom.pdf`): related work on extreme-scale super-resolution.

## Author

**Mainak Das** · GitHub: [@mainak569](https://github.com/mainak569)
Developed as part of **LUSIP (Learning Under Summer Internship Program)**.
