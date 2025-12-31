# MFS-Fusion（ESWA2026)
MFS-Fusion: Mamba-Integrated Deep Multi-Modal Image Fusion Framework with Multi-Scale Fourier Enhancement and Spatial Calibration
（[Paper Link](https://www.sciencedirect.com/science/article/abs/pii/S095741742503670X)）
## Environment Setup

Create and activate a new conda environment:

```bash
conda create -n mfs python=3.10
conda activate mfs
```

Install required dependencies:

```bash
pip install numpy
pip install torch
pip install opencv-python
pip install torchvision
pip install matplotlib
pip install tensorboard
pip install einops
pip install mamba_ssm
pip install timm
```

## Data

Download the dataset from the provided link:

[https://pan.baidu.com/s/15LzwkowSbawPkJhN7FDP9g?pwd=5awc](https://pan.baidu.com/s/15LzwkowSbawPkJhN7FDP9g?pwd=5awc)

## Fast Test

To quickly test the model and generate results:

```bash
python test_unet.py
```

Results will be saved in `./results/VIS-IR/` directory.

**Note:** Make sure the pre-trained model weights are available at `./Model/Fusion/Fusion/models/Best.pth` for testing.

## Retrain

To retrain the model with different configurations or datasets:

```bash
python train.py
```

**Note:** Make sure to modify the `config.json` file to support training with your corresponding dataset.

### Configuration

The training configuration can be adjusted in `config.json`. Key parameters include:

- Dataset paths (`dataroot_A`, `dataroot_B`)
- Model parameters (`netG` section)
- Training hyperparameters (`train` section)
- Data loading settings (`datasets` section)

## Project Structure

```
MFS-Fusion/
├── train.py                 # Training script
├── test_unet.py            # Testing script
├── config.json             # Configuration file
├── Net/                    # Network architectures
│   ├── MyNet.py           # Main network implementation
│   ├── MVTNet.py          # MobileViT implementation
│   └── ...
├── models/                 # Model definitions and utilities
├── data/                   # Data loading utilities
├── utils/                  # Utility functions
├── Model/                  # Model weights and checkpoints
│   └── Fusion/Fusion/models/
│       └── Best.pth       # Pre-trained model weights
└── results/               # Output results
```

## Usage Examples

### Training
```bash
# Use default configuration
python train.py

# Use custom configuration
python train.py --opt your_config.json
```

### Testing
```bash
# Test with default settings
python test_unet.py
```

## Requirements

- Python 3.10
- PyTorch
- CUDA (recommended for GPU acceleration)
- See the environment setup section for the complete dependency list

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{chen2025mfs,
  title={MFS-Fusion: Mamba-Integrated Deep Multi-Modal Image Fusion Framework with Multi-Scale Fourier Enhancement and Spatial Calibration},
  author={Chen, Tao and Wang, Chuang and Zhang, Yuanpeng and Xia, Kaijian and Qian, Pengjiang},
  journal={Expert Systems with Applications},
  pages={130054},
  year={2025},
  publisher={Elsevier}
}
```
