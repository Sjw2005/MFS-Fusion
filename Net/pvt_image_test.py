import cv2
import torch
from Net.pvt import PyramidVisionTransformerV2

img = cv2.imread("your_image.png", cv2.IMREAD_GRAYSCALE)  # H,W
img = cv2.resize(img, (128, 128))
x1 = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0  # [1,1,H,W]

pre = torch.nn.Conv2d(1, 48, 3, 1, 1).eval()
x = pre(x1)  # [1,48,128,128]

model = PyramidVisionTransformerV2(
    in_chans=48, embed_dims=[48,96,192], num_heads=[1,2,4],
    mlp_ratios=[4,4,4], depths=[2,2,2], sr_ratios=[8,4,2],
    num_stages=3, linear=True
).eval()

# 同样建议手动逐stage测试，便于debug