# 🧠 Unsupervised Image Restoration & Super-Resolution (Deep Image Prior)

This repository contains code and experimental notebooks for **unsupervised image restoration and super-resolution** using the **Deep Image Prior (DIP)** framework.  
The project explores noise removal and resolution enhancement on ancient *Kannada palm leaf manuscripts* and other natural images — without requiring supervised training data.

---

## 📂 Project Structure
```
final-lusip/
│
├── DEEP_IMAGE_PRIOR_kannada_palm_leaf_8image.ipynb     # Main Jupyter notebook
├── zebra.py                                            # Python implementation script
├── dip_metrics_8images.csv                             # PSNR/SSIM metrics
├── sr_miniDataset/                                     # Dataset (high_res & low_res)
├── ancient_kannada_palm_leaf/                          # Sample palm-leaf images
├── Chain-of-Zoom.pdf                                   # Paper reference
├── psnr-compare.png / ssim.png                         # Evaluation visuals
└── README.md                                           # You are here 🙂
```
---

## ⚙️ Setup Instructions

### 1. Clone the repository
git clone https://github.com/mainak569/unsupervised_imgRestoration_SR.git
cd unsupervised_imgRestoration_SR

### 2. Install dependencies
Use Python ≥3.8 and install required libraries:
pip install -r requirements.txt

If you don’t have a `requirements.txt`, typical packages are:
pip install numpy scipy matplotlib pillow torch torchvision tqdm opencv-python

### 3. (Optional) Create a virtual environment
python3 -m venv env
source env/bin/activate  # macOS/Linux
env\Scripts\activate     # Windows

---

## 💾 Dataset Notes (Git LFS Required)

This project includes many **high-resolution images**.  
To handle them properly, **Git LFS (Large File Storage)** must be installed.

### Install and configure Git LFS:
#### macOS
brew install git-lfs

#### Initialize LFS globally
git lfs install

#### Track large file types
git lfs track "*.jpg" "*.png" "*.pdf" "*.csv" "*.ipynb"

#### Add tracking configuration
git add .gitattributes
git commit -m "Add Git LFS tracking"

---

## 🚀 Run the Notebook

You can explore and run the restoration pipeline using:
jupyter notebook DEEP_IMAGE_PRIOR_kannada_palm_leaf_8image.ipynb

Or directly execute the Python script:
python zebra.py

---

## 📊 Outputs & Metrics

- **PSNR** and **SSIM** scores for 8 test images are provided in `dip_metrics_8images.csv`.
- Visual comparisons are stored in:
  - `psnr-compare.png`
  - `ssim.png`
  - `psnr_evolution.png`

---

## 🌐 Git & Push Notes

If pushing large data to GitHub:
### Increase buffer and prevent timeout
git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999

### Recommit and push
git add .
git commit -m "final LFS-optimized commit"
git push origin main --force

---

## 🧠 Reference

- Deep Image Prior: *Ulyanov et al., CVPR 2018*
- Paper Link: https://arxiv.org/abs/1711.10925
- Implementations inspired by official PyTorch examples and community repos.

---

## 👤 Author

**Mainak Das**  
📧 GitHub: [mainak569](https://github.com/mainak569)  
💡 Project developed as part of **LUSIP (Learning Under Summer Internship Program)**.

---
