import torch
from Net.MyNet import MyNet

model = MyNet(in_chans=1, hidden_chans=[48, 96, 192], out_chans=1)
x = torch.randn(2, 1, 256, 256)
y = torch.randn(2, 1, 256, 256)
out = model(x, y)
assert out.shape == (2, 1, 256, 256), f"输出形状错误: {out.shape}"
print("形状验证通过")