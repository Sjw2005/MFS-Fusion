import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt

from Net.MobileViT import mobile_vit_small
from Net.pvt import PyramidVisionTransformerV2, PyramidVisionTransformerV2_one
import torch.nn.functional as F
import os
# import onnx
from einops import rearrange
from mamba_ssm import Mamba


def _debug_tensor_stats(tag, tensor):
    if tensor is None:
        print(f"[DEBUG][{tag}] tensor is None")
        return
    with torch.no_grad():
        t = tensor.detach()
        finite_mask = torch.isfinite(t)
        finite_ratio = finite_mask.float().mean().item()
        nan_count = torch.isnan(t).sum().item()
        inf_count = torch.isinf(t).sum().item()
        if finite_mask.any():
            finite_t = t[finite_mask]
            t_min = finite_t.min().item()
            t_max = finite_t.max().item()
            t_mean = finite_t.mean().item()
        else:
            t_min = float('nan')
            t_max = float('nan')
            t_mean = float('nan')
        print(
            f"[DEBUG][{tag}] shape={tuple(t.shape)} min={t_min:.6f} max={t_max:.6f} "
            f"mean={t_mean:.6f} finite_ratio={finite_ratio:.6f} nan={int(nan_count)} inf={int(inf_count)}"
        )

def custom_complex_normalization(input_tensor, dim=-1):
    # 作用：对复数张量的实部与虚部分别做 softmax，再重新拼回应复数张量。
    # 输入 shape：通常与 attention 分数一致，例如 [B, head, C_per_head, C_per_head]。
    # real_part / imag_part shape 与 input_tensor 完全一致。
    # 说明：这里的“归一化”并不是严格的复数概率归一化，而是分别处理实部、虚部。
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)

    # 经过 softmax 后，real / imag 在 dim 维上的数值被约束到类似概率分布的范围。
    # 最终输出仍是复数张量，shape 不变。
    normalized_tensor = torch.complex(norm_real, norm_imag)

    return normalized_tensor


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    # 作用：构造一个最基础的 3x3 卷积层。
    # 输入 shape：[B, in_planes, H, W]
    # 输出 shape：[B, out_planes, H/stride, W/stride]
    # 由于 padding=1，若 stride=1，则空间尺寸 H、W 不变。
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    # 作用：常用卷积块 = 3x3 Conv + BN + ReLU。
    # 输入 shape：[B, in_planes, H, W]
    # 输出 shape：[B, out_planes, H/stride, W/stride]
    # 若 stride=1，则只改变通道数，不改变空间分辨率。
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    # 作用：构造一个 1x1 卷积层，主要负责通道映射/通道混合。
    # 输入 shape：[B, in_planes, H, W]
    # 输出 shape：[B, out_planes, H/stride, W/stride]
    # 若 stride=1，则空间尺寸不变，只调整通道表达。
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    # 作用：常用逐点卷积块 = 1x1 Conv + BN + ReLU。
    # 典型用途：通道压缩、通道升维、特征融合后的线性映射。
    # 输入 shape：[B, in_planes, H, W]
    # 输出 shape：[B, out_planes, H/stride, W/stride]
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )



class MambaLayer(nn.Module):
    # 作用：将二维特征图展平成序列后送入 Mamba，建模长程依赖，
    # 再还原回特征图；同时通过四个方向/翻转版本增强序列建模的方向鲁棒性。
    # 输入输出的总体 shape 都保持为 [B, dim, H, W]。
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.nin = conv1x1(dim, dim)
        self.nin2 = conv1x1(dim, dim)
        self.norm2 = nn.BatchNorm2d(dim) # LayerNorm
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.norm = nn.BatchNorm2d(dim) # LayerNorm
        self.relu = nn.ReLU(inplace=True)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )

    def forward(self, x):
        # MambaLayer 的核心流程：
        # 1) 先用 1x1 卷积做通道混合；
        # 2) 将 [B, C, H, W] 展平为 [B, H*W, C] 序列；
        # 3) 分别以原序、长度翻转、通道翻转、双翻转四种形式送入 Mamba；
        # 4) 四路结果平均后恢复为二维特征图，并叠加残差。
        # IN: x [B, C, H, W]
        B, C = x.shape[:2]
        x = self.nin(x)
        x = self.norm(x)
        x = self.relu(x)
        # After 1x1 conv + BN + ReLU:
        # x: [B, C, H, W] (shape unchanged, channel mixed by 1x1 conv)
        # act_x 作为局部残差，shape 仍为 [B, C, H, W]。
        act_x = x
        assert C == self.dim
        # n_tokens = H*W，表示把整张特征图看作多少个 token。
        n_tokens = x.shape[2:].numel()
        # img_dims = (H, W)，后续用于恢复回二维空间。
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        # Flatten spatial dims to token sequence:
        # x_flat: [B, H*W, C]

        # 下面四路输入的 shape 全都是 [B, H*W, C]：
        # x_flip_l：在通道/特征维上翻转；
        # x_flip_c：在 token 序列维上翻转；
        # x_flip_lc：同时翻转 token 维和通道维。
        x_flip_l = torch.flip(x_flat, dims=[2])
        x_flip_c = torch.flip(x_flat, dims=[1])
        x_flip_lc = torch.flip(x_flat, dims=[1,2])
        # 四次 Mamba 建模后输出 shape 仍保持 [B, H*W, C]。
        x_ori = self.mamba(x_flat)
        x_mamba_l = self.mamba(x_flip_l)
        x_mamba_c = self.mamba(x_flip_c)
        x_mamba_lc = self.mamba(x_flip_lc)
        # 将翻转后的结果翻回来，对齐到原始 token 顺序。
        x_ori_l = torch.flip(x_mamba_l, dims=[2])
        x_ori_c = torch.flip(x_mamba_c, dims=[1])
        x_ori_lc = torch.flip(x_mamba_lc, dims=[1,2])
        # 四路平均融合，shape 仍是 [B, H*W, C]。
        x_mamba = (x_ori+x_ori_l+x_ori_c+x_ori_lc)/4

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        # Sequence -> feature map:
        # out: [B, C, H, W]
        # 与局部卷积特征 act_x 做残差相加，shape 不变：[B, C, H, W]。
        out += act_x
        # 再经过一次 1x1 Conv + BN + ReLU 进行通道重整，输出 shape 仍为 [B, C, H, W]。
        out = self.nin2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        # OUT: [B, C, H, W]
        return out

class ECALayer(nn.Module):
    """轻量级通道注意力（Efficient Channel Attention）。
    用 1D 卷积替代 SE 中的双全连接层，避免降维信息损失，参数量更低。
    """
    def __init__(self, channel, gamma=2, b=1):
        super(ECALayer, self).__init__()
        # 根据通道数自适应计算卷积核大小
        import math
        t = int(abs((math.log2(channel) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)  # 最小核大小为 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)                          # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)           # [B, 1, C]
        y = self.conv(y)                               # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)         # [B, C, 1, 1]
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class FSDA(nn.Module):
    """频域幅值-相位解耦注意力（Frequency Spectrum Decoupled Attention）。
    将 FFT 输出显式分解为幅度谱和相位谱，分别通过 ECA 通道注意力增强，
    再通过极坐标重建回复数频谱，经 IFFT 回到空间域。
    
    替换原始 AttenFFT 中数学上不严谨的复数 Softmax 操作。
    
    Args:
        dim (int): 输入通道数（与 AttenFFT 接口一致）
    """
    def __init__(self, dim):
        super(FSDA, self).__init__()
        # 幅度谱增强分支
        self.mag_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.mag_attn = ECALayer(dim)  # 幅度谱通道注意力
        self.mag_proj = nn.Conv2d(dim, dim, kernel_size=1)

        # 相位谱增强分支
        self.pha_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.Identity(),
        )
        self.pha_attn = ECALayer(dim)  # 相位谱通道注意力
        self.pha_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 空间域特征图
        Returns:
            out: [B, C, H, W] 频域增强后的空间域特征图
        """
        # Step 1: FFT2 变换到频域
        if not torch.isfinite(x).all():
            _debug_tensor_stats('FSDA/input', x)
        x_fft = torch.fft.fft2(x, norm='ortho')  # [B, C, H, W] 复数

        # Step 2: 显式分解幅度谱和相位谱
        mag = torch.abs(x_fft)           # [B, C, H, W] 实数，幅度
        pha = torch.angle(x_fft)         # [B, C, H, W] 实数，相位（弧度）
        if not torch.isfinite(mag).all():
            _debug_tensor_stats('FSDA/mag', mag)
        if not torch.isfinite(pha).all():
            _debug_tensor_stats('FSDA/pha', pha)

        # Step 3: 幅度谱通道注意力增强（在实数域操作，数学严谨）
        mag_feat = self.mag_enhance(mag)         # Conv1×1 + BN + ReLU
        mag_feat = self.mag_attn(mag_feat)       # ECA 通道注意力
        mag_out = self.mag_proj(mag_feat) + mag  # 残差连接：增强 + 原始

        # Step 4: 相位谱通道注意力增强（在实数域操作，数学严谨）
        pha_feat = self.pha_enhance(pha)
        pha_feat = self.pha_attn(pha_feat)
        pha_out = self.pha_proj(pha_feat) + pha  # 残差连接
        if not torch.isfinite(mag_out).all():
            _debug_tensor_stats('FSDA/mag_out', mag_out)
        if not torch.isfinite(pha_out).all():
            _debug_tensor_stats('FSDA/pha_out', pha_out)

        # Step 5: 极坐标重建回复数频谱
        # F_out = mag_out * exp(j * pha_out)
        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        x_fft_enhanced = torch.complex(real, imag)

        # Step 6: IFFT2 回到空间域
        out = torch.fft.ifft2(x_fft_enhanced, norm='ortho').real  # 取实部
        if not torch.isfinite(out).all():
            _debug_tensor_stats('FSDA/out', out)

        return out


class AttenFFT(nn.Module):
    # 作用：在频域上做多头通道注意力，并辅以一条频域门控增强分支。
    # 该模块输入输出 shape 一致，均为 [B, dim, H, W]。
    # 它主要负责：
    # 1) 通过 FFT 将空间特征映射到频域；
    # 2) 在频域上计算通道间的相关性；
    # 3) 通过另一条频域权重分支学习“哪些频率需要增强/抑制”；
    # 4) 将两路频域增强结果融合回空间域特征。
    """
    输入 x: [B, dim, H, W]
    → DWConv3×3 生成 q_s, k_s, v_s（三个独立的 depthwise conv，均为 3×3）
    → 对 q_s, k_s, v_s 分别做 torch.fft.fft2 得到复数张量
    → 多头拆分后计算 Q×K^T（复数域点积）
    → 对复数注意力矩阵的实部和虚部分别做 Softmax（问题所在）
    → 加权求和后 IFFT2 → 取绝对值回到实数域
    → 同时有一个 weight 分支：FFT(x) → Conv1×1→BN→ReLU→Conv1×1→Sigmoid 生成频域权重
    → 两路输出 cat → Conv1×1(2*dim → dim) 得到最终输出
    """
    def __init__(self, dim, num_heads=2, bias=False, ):
        super(AttenFFT, self).__init__()
        # 总通道数dim会被分成num_heads份
        # 每个head处理一部分通道
        # 让不同的head学到不同的频域特征
        self.num_heads = num_heads

        # 三个deptwise conv生成q/k/v
        self.qkv1conv_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias) # v_s
        self.qkv1conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias) # k_s
        self.qkv1conv_5 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias) # q_s

        # 每个 head 一个可学习缩放参数，用于调节注意力矩阵的分布强弱。
        # shape: [num_heads, 1, 1]，会在 batch 维上自动广播。
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 频域权重分支：输入为 FFT 后特征的实部，输出为 [0, 1] 范围的频率权重图。
        # 输入 shape：[B, dim, H, W]
        # 输出 shape：[B, dim, H, W]
        self.weight = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.Sigmoid())

        # 用于融合两条分支输出：
        # cat 后通道数从 2*dim 压回 dim，空间分辨率不变。
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        # AttenFFT 的 forward 主要包含两条路径：
        # 主路径：q/k/v -> FFT -> 多头频域注意力 -> IFFT；
        # 辅路径：x -> FFT -> 学习频域 mask -> IFFT。
        # 最终把两路结果在通道维拼接，再用 1x1 卷积融合。
        # IN: x [b, c, h, w]
        b, c, h, w = x.shape
        # 通过三个 depthwise 3x3 卷积生成 q/k/v。
        # 因为 groups=dim，所以每个通道独立卷积，shape 不变。
        q_s = self.qkv1conv_5(x)
        k_s = self.qkv1conv_3(x)
        v_s = self.qkv1conv_1(x)
        # Depthwise conv branches keep shape:
        # q_s/k_s/v_s: [b, c, h, w]

        # 对 q/k/v 分别做二维 FFT：
        # shape 仍为 [b, c, h, w]，但数据类型由实数变为复数 complex。
        # 此时每个位置表示对应频率分量的幅值+相位信息。
        q_s = torch.fft.fft2(q_s.float())
        k_s = torch.fft.fft2(k_s.float())
        v_s = torch.fft.fft2(v_s.float())

        q_s = rearrange(q_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_s = rearrange(k_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # Rearranged to multi-head frequency tokens:
        # q_s/k_s/v_s: [b, head, c_per_head, h*w]

        # 在最后一维 H*W 上做归一化。
        # q_s/k_s shape 不变，仍为 [b, head, c_per_head, h*w]。
        # 可以理解为：每个通道的频率向量被规范化，便于后续计算相似度。
        q_s = torch.nn.functional.normalize(q_s, dim=-1)
        k_s = torch.nn.functional.normalize(k_s, dim=-1)
        # q_s @ k_s^T：
        # [b, head, c_per_head, h*w] @ [b, head, h*w, c_per_head]
        # -> [b, head, c_per_head, c_per_head]
        # 得到的是“每个 head 内部的通道-通道相关性矩阵”，而不是空间位置间相关性。
        # 再乘以可学习温度参数，shape 不变。
        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        # 对复数注意力矩阵做自定义归一化后，shape 仍为 [b, head, c_per_head, c_per_head]。
        attn_s = custom_complex_normalization(attn_s, dim=-1)


        # attn_s @ v_s：
        # [b, head, c_per_head, c_per_head] @ [b, head, c_per_head, h*w]
        # -> [b, head, c_per_head, h*w]
        # 表示使用通道相关性矩阵去重加权 value。
        # ifft2 后 shape 不变，仍是 [b, head, c_per_head, h*w]，数据回到复数空间表示；
        # 再经过 abs 取模，变成实数张量。
        outr = torch.abs(torch.fft.ifft2(attn_s @ v_s))
        # 将多头张量重新拼回标准特征图：
        # [b, head, c_per_head, h*w] -> [b, c, h, w]
        outr = rearrange(outr, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # Back to feature map:
        # outr: [b, c, h, w]

        # 辅助频域门控分支：
        # 1) torch.fft.fft2(x.float())：得到输入的复数频谱，shape [b, c, h, w]；
        # 2) .real：取实部后 shape 不变，作为权重网络输入；
        # 3) self.weight(...)：生成 [0,1] 范围频率权重图，shape [b, c, h, w]；
        # 4) 与原始复数频谱逐点相乘，shape 仍为 [b, c, h, w]；
        # 5) ifft2 + abs：恢复为实数空间域特征，得到 out_f_lr，shape [b, c, h, w]。
        out_f_lr = torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(x.float()).real) * torch.fft.fft2(x.float())))
        # 主分支 outr 与辅助分支 out_f_lr 在通道维拼接：
        # [b, c, h, w] cat [b, c, h, w] -> [b, 2c, h, w]
        # 再经 1x1 卷积融合回 [b, c, h, w]。
        outr = self.project_out(torch.cat((outr, out_f_lr), 1))
        # Concatenate then project 2c -> c:
        # OUT: [b, c, h, w]


        return outr


class ASI(nn.Module):
    # 作用：将输入通道均分成 4 份，分别使用不同膨胀率的卷积 + AttenFFT 进行多尺度频域增强，
    # 再把四路结果拼接融合，得到多尺度感受野下的增强特征。
    # 输入输出 shape 相同，都是 [B, dim, H, W]。
    def __init__(self, dim):
        super(ASI, self).__init__()
        # 每一路只处理总通道数的 1/4。
        self.dim = dim // 4
        # 四个不同膨胀率的卷积，对应不同感受野：
        # dilation=1/2/3/4，padding 与之匹配，因此空间尺寸 H、W 保持不变。
        # 每路输入输出 shape 都是 [B, dim/4, H, W]。
        self.conv1 = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv4 = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=4, dilation=4)

        # 原有方法
        # self.fft1 = AttenFFT(self.dim)
        # self.fft2 = AttenFFT(self.dim)
        # self.fft3 = AttenFFT(self.dim)
        # self.fft4 = AttenFFT(self.dim)

        # FSDA改进
        self.fft1 = FSDA(self.dim)
        self.fft2 = FSDA(self.dim)
        self.fft3 = FSDA(self.dim)
        self.fft4 = FSDA(self.dim)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=3,padding=1)


    def forward(self, x):
        # ASI 的思路：先拆通道，再并行做多尺度增强，最后再拼接回去。
        # IN: x [B, dim, H, W]
        x1,x2,x3,x4 = torch.chunk(x,4,dim=1)
        # Channel split into 4 equal groups:
        # each xi: [B, dim/4, H, W]

        # 四路分支分别进行：
        # 卷积提取局部/多尺度信息 -> AttenFFT 做频域增强 -> 与原分支残差相加。
        # 每一路 shape 始终保持 [B, dim/4, H, W]。
        x1 = self.fft1(self.conv1(x1)) + x1
        x2 = self.fft2(self.conv2(x2)) + x2
        x3 = self.fft3(self.conv3(x3)) + x3
        x4 = self.fft4(self.conv4(x4)) + x4

        outs = torch.cat((x1,x2,x3,x4),1) + x
        # Merge 4 groups back to full channel:
        # outs: [B, dim, H, W]

        # 3x3 卷积进一步融合四路信息，shape 保持 [B, dim, H, W]。
        out = self.project_out(outs)
        # OUT: [B, dim, H, W]

        return out


class sa_layer(nn.Module):
    # 作用：一个轻量级的混合注意力模块。
    # 它先按 group 重排特征，再将通道一分为二：
    # 一半做 channel attention，一半做 spatial attention，最后拼接并 channel shuffle。
    # 输入输出 shape 保持一致：[B, channel, H, W]。
    def __init__(self, channel, groups=4):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        # 作用：将分组后的通道重新交错，促进不同组之间的信息交换。
        # 输入 shape：[B, C, H, W]
        b, c, h, w = x.shape

        # -> [B, groups, C/groups, H, W]
        x = x.reshape(b, groups, -1, h, w)
        # -> [B, C/groups, groups, H, W]
        x = x.permute(0, 2, 1, 3, 4)

        # -> [B, C, H, W]
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        # IN: x [B, C, H, W]
        b, c, h, w = x.shape

        # 按组展开 batch 维，便于在每组内部做注意力。
        # [B, C, H, W] -> [B*groups, C/groups, H, W]
        x = x.reshape(b * self.groups, -1, h, w)
        # 每组通道再均分为两半：
        # x_0/x_1: [B*groups, C/(2*groups), H, W]
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        # 对 x_0 做全局平均池化：
        # [B*groups, C/(2*groups), H, W] -> [B*groups, C/(2*groups), 1, 1]
        xn = self.avg_pool(x_0)
        # 逐通道仿射变换后 shape 不变。
        xn = self.cweight * xn + self.cbias
        # 生成通道权重并回乘到原特征，恢复 shape：[B*groups, C/(2*groups), H, W]
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        # 对 x_1 做组归一化，shape 不变：[B*groups, C/(2*groups), H, W]
        xs = self.gn(x_1)
        # 逐元素仿射调制，shape 不变。
        xs = self.sweight * xs + self.sbias
        # 生成空间注意力响应后回乘原特征，shape 仍不变。
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        # 在通道维拼接两路结果：
        # [B*groups, C/(2*groups), H, W] cat 同 shape -> [B*groups, C/groups, H, W]
        out = torch.cat([xn, xs], dim=1)
        # 恢复原始 batch 组织方式：
        # [B*groups, C/groups, H, W] -> [B, C, H, W]
        out = out.reshape(b, -1, h, w)

        # 通道洗牌后 shape 保持 [B, C, H, W]。
        out = self.channel_shuffle(out, 2)

        return out


class MEF(nn.Module):
    # 作用：双分支模态特征融合模块。
    # 输入为两路特征 ir / vi，先通过空间注意力强调共有显著区域，
    # 再分别做 ASI 增强，最后拼接融合为单一路输出。
    # 返回三个张量：增强后的 ir、增强后的 vi、融合结果 out。
    def __init__(self, dim):
        super(MEF, self).__init__()

        self.spatial_attention = SpatialAttention()

        self.feir = ASI(dim)
        self.fevi = ASI(dim)

        self.cov = conv3x3_bn_relu(2 * dim, dim)

    def forward(self, ir, vi):
        # IN ir/vi: [B, dim, H, W]
        # 两路逐元素相乘，强调共同响应区域。
        # mul_fuse: [B, dim, H, W]
        mul_fuse = ir * vi
        sa = self.spatial_attention(mul_fuse)
        # Spatial attention map squeezes channels:
        # sa: [B, 1, H, W]
        vi = vi * sa + vi
        ir = ir * sa + ir
        # Broadcast multiply keeps branch shape:
        # ir/vi: [B, dim, H, W]

        # 两个分支分别经过 ASI 模块做多尺度频域增强，再叠加残差。
        # vi/ir shape 均保持 [B, dim, H, W]。
        vi = self.fevi(vi) + vi
        ir = self.feir(ir) + ir

        out = self.cov(torch.cat((vi, ir), 1))
        # Concat then fuse by 3x3 conv:
        # cat: [B, 2*dim, H, W] -> out: [B, dim, H, W]

        return ir, vi, out


class MyNet(nn.Module):
    # 作用：整个融合网络的主干。
    # 整体流程：
    # 1) 两路单通道输入先映射到浅层特征；
    # 2) 分别送入两个 PVT 主干提取三阶段多尺度特征；
    # 3) 每个尺度用 MEF 进行双模态交互与融合；
    # 4) 用 Decode 逐级上采样恢复分辨率；
    # 5) 用最后的增强头生成单通道融合结果。
    def __init__(self, in_chans=1, hidden_chans=[48, 96, 192], pool_ratio=[8, 6, 4], out_chans=1, linear=True):
        super(MyNet, self).__init__()

        """
        预处理卷积，得到48通道的特征图
        送入两个PVT，得到三个阶段的特征图
        In: x:[2 ,1 ,256 ,256]    y:[2 ,1 ,256 ,256]
        """
        self.pool_ratio = pool_ratio
        self.pre_x = nn.Conv2d(in_chans, hidden_chans[0], 3, 1, 1)
        self.pre_y = nn.Conv2d(in_chans, hidden_chans[0], 3, 1, 1)

        # x[2 ,48 ,256 ,256]    y[2 ,48 ,256 ,256] 同时保存短残差
        self.un_x = PyramidVisionTransformerV2(in_chans=48, embed_dims=hidden_chans,
                                               num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4],
                                               depths=[2, 2, 2], sr_ratios=[8, 4, 2], num_stages=3, linear=linear)
        self.un_y = PyramidVisionTransformerV2(in_chans=48, embed_dims=hidden_chans,
                                               num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4],
                                               depths=[2, 2, 2], sr_ratios=[8, 4, 2], num_stages=3, linear=linear)

        # 三个尺度分别对应一个融合模块：
        # stage0: 48 通道，stage1: 96 通道，stage2: 192 通道。
        self.fuse0 = MEF(dim=hidden_chans[0])
        self.fuse1 = MEF(dim=hidden_chans[1])
        self.fuse2 = MEF(dim=hidden_chans[2])
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fes = [self.fuse0,self.fuse1,self.fuse2]
        # 最终重建头：
        # 输入 [B, 48, H, W]，先做卷积增强，再通过 ESI 和 sa_layer 细化特征，
        # 最后映射到 out_chans（默认 1 通道）输出。
        self.last = nn.Sequential(
            nn.Conv2d(in_channels=hidden_chans[0], out_channels=hidden_chans[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_chans[0]),
            nn.GELU(),
            # self.upsample2,
            ESI(hidden_chans[0]),
            sa_layer(hidden_chans[0]),
            nn.Conv2d(in_channels=hidden_chans[0], out_channels=out_chans, kernel_size=3, padding=1, bias=True),
        )



        self.dec = Decode(hidden_chans[0],hidden_chans[1],hidden_chans[2])

    def forward(self, x,y):
        # MyNet forward：
        # 输入两张单通道图像 x、y，输出一张单通道融合图。
        fuses = []
        # Input from dataloader (after collate):
        # x/y: [B, 1, H, W], in this project usually H=W=256


        h, w = x.shape[2], x.shape[3]
        # img_size 记录原图大小，当前代码里未继续使用，但通常用于后续恢复分辨率。
        img_size = (h, w)
        x = self.pre_x(x)
        y = self.pre_y(y)
        # Stem conv (1 -> 48 channels, resolution unchanged):
        # x/y: [B, 48, H, W]
        short_x = x
        short_y = y
        # short_x/short_y are full-resolution residual features:
        # [B, 48, H, W]


        B = x.shape[0]
        # 逐阶段提取与融合多尺度特征。
        for i in range(self.un_x.num_stages):
            # Stage i shape expectation with H=W=256:
            # i=0 -> [B, 48, 128, 128]
            # i=1 -> [B, 96,  64,  64]
            # i=2 -> [B, 192, 32,  32]
            patch_embedx = getattr(self.un_x, f"patch_embed{i + 1}")
            blockx = getattr(self.un_x, f"block{i + 1}")
            normx = getattr(self.un_x, f"norm{i + 1}")

            patch_embedy = getattr(self.un_y, f"patch_embed{i + 1}")
            blocky = getattr(self.un_y, f"block{i + 1}")
            normy = getattr(self.un_y, f"norm{i + 1}")

            x, H, W = patch_embedx(x)
            # OverlapPatchEmbed does stride=2 downsampling and channel projection:
            # x: [B, H*W, C_i], where C_i is [48, 96, 192] by stage
            # 这里 H、W 分别是当前 stage 下采样后的空间尺寸。
            for blkx in blockx:
                # 每个 Transformer block 仅更新 token 特征，shape 保持 [B, H*W, C_i]。
                x = blkx(x, H, W)
            # Transformer blocks keep token shape:
            # x: [B, H*W, C_i]
            x = normx(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            # Tokens -> feature map:
            # x: [B, C_i, H, W]

            y, H, W = patch_embedy(y)
            # y branch has the same stage shape as x branch:
            # y: [B, H*W, C_i]
            for blky in blocky:
                # y 分支与 x 分支完全对称，shape 也保持 [B, H*W, C_i]。
                y = blky(y, H, W)
            y = normy(y)
            y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            # y: [B, C_i, H, W]

            x,y,out = self.fes[i](x,y)
            # MEF returns branch-refined features and fused feature:
            # x/y/out: [B, C_i, H, W]

            fuses.append(out)
            # fuses[0]: [B, 48, 128,128], fuses[1]: [B, 96, 64,64], fuses[2]: [B, 192, 32,32]


        # 解码器接收三个尺度的融合特征：
        # fuses[0]=浅层 [B,48,H/2,W/2]
        # fuses[1]=中层 [B,96,H/4,W/4]
        # fuses[2]=深层 [B,192,H/8,W/8]
        out = self.dec(fuses[0],fuses[1],fuses[2])
        # Decoder restores to full resolution:
        # out: [B, 48, H, W]
        if not torch.isfinite(out).all():
            print('[DEBUG] Non-finite tensor detected after decoder in MyNet.forward')
            _debug_tensor_stats('MyNet/dec_out', out)
        # 与两路浅层短残差相加后送入最后预测头，shape 仍为 [B, 48, H, W]。
        out = self.last(out+short_x+short_y)
        out = torch.sigmoid(out)
        if not torch.isfinite(out).all():
            print('[DEBUG] Non-finite tensor detected at final output in MyNet.forward')
            _debug_tensor_stats('MyNet/final_out', out)
        # Residual add keeps [B, 48, H, W], last head maps to single channel:
        # final out: [B, 1, H, W]


        return out



class Decode(nn.Module):
    # 作用：将三层金字塔融合特征逐级上采样，恢复回高分辨率特征图。
    # 输入通常是：
    # x1=[B,48,H/2,W/2]，x2=[B,96,H/4,W/4]，x3=[B,192,H/8,W/8]
    # 输出为 [B,48,H,W]。
    def __init__(self, in1,in2,in3):
        super(Decode, self).__init__()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv_up4 = nn.Sequential(
        #     nn.Conv2d(in_channels=in4, out_channels=in3, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(in3),
        #     nn.GELU(),
        #     self.upsample2
        # )
        # conv_up3：最深层特征 x3 先做通道压缩 192->96，再上采样 2 倍，
        # 输出尺寸与 x2 对齐，便于后续拼接。
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(in_channels=in3, out_channels=in2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in2),
            nn.GELU(),
            self.upsample2,
            ESI(in2),
            sa_layer(in2)
        )
        # conv_up2：拼接 up3 与 x2 后，通道数 2*in2 -> in1，再上采样到与 x1 对齐。
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(in_channels=in2*2, out_channels=in1, kernel_size=1, bias=False),
            nn.BatchNorm2d(in1),
            nn.GELU(),
            self.upsample2,
            ESI(in1),
            sa_layer(in1)
        )
        # conv_up1：拼接 up2 与 x1 后，再次融合并上采样到原图分辨率。
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(in_channels=in1*2, out_channels=in1, kernel_size=1, bias=False),
            nn.BatchNorm2d(in1),
            nn.GELU(),
            self.upsample2,
            ESI(in1),
            sa_layer(in1)
        )


        self.p_1 = nn.Sequential(
            nn.Conv2d(in_channels=in1, out_channels=in1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in1),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=in1, out_channels=in1, kernel_size=3, padding=1, bias=True),
        )



    def forward(self,x1,x2,x3):
        # Decode forward：从深层到浅层逐步恢复分辨率。
        # IN:
        # x1: [B, in1, H/2, W/2]   (default [B,48,128,128])
        # x2: [B, in2, H/4, W/4]   (default [B,96,64,64])
        # x3: [B, in3, H/8, W/8]   (default [B,192,32,32])

        # up4 = self.conv_up4(x4)
        up3 = self.conv_up3(x3)
        # conv_up3: 192 -> 96, then upsample x2:
        # up3: [B, 96, H/4, W/4] (default [B,96,64,64])
        up2 = self.conv_up2(torch.cat((up3,x2),1))
        # cat(up3,x2): [B, 192, H/4, W/4] -> conv_up2 -> [B,48,H/2,W/2]
        # 此时 up2 已经与最浅层特征 x1 的空间尺寸一致。
        up1 = self.conv_up1(torch.cat((up2,x1), 1))
        # cat(up2,x1): [B, 96, H/2, W/2] -> conv_up1 -> [B,48,H,W]



        # OUT: [B, in1, H, W] (default [B,48,256,256])
        return up1


class ESI(nn.Module):
    # 作用：多分支局部卷积 + Mamba 序列建模增强模块。
    # 它使用不同卷积核尺寸提取不同感受野下的特征，
    # 每一路再用 MambaLayer 建模长程关系，最后拼接融合。
    # 输入输出 shape 相同：[B, dim, H, W]。
    def __init__(self, dim):
        super(ESI, self).__init__()
        # 每一路分支输出 dim/4 个通道，四路拼接后回到原始 dim 通道。
        self.dim = dim // 4
        # 四个并行卷积分支：
        # 1x1 更偏通道映射；3x3/5x5/7x7 提供不同大小的局部感受野。
        # 每个分支输出 shape 都是 [B, dim/4, H, W]。
        self.conv1 = nn.Conv2d(dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(dim, self.dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(dim, self.dim, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(dim, self.dim, kernel_size=7, stride=1, padding=3)

        self.mmb1 = MambaLayer(self.dim)
        self.mmb2 = MambaLayer(self.dim)
        self.mmb3 = MambaLayer(self.dim)
        self.mmb4 = MambaLayer(self.dim)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        # IN: x [B, dim, H, W]

        # 四个分支先做不同感受野卷积，再交给 MambaLayer 建模。
        # 每一路输出均为 [B, dim/4, H, W]。
        x1 = self.mmb1(self.conv1(x))
        x2 = self.mmb2(self.conv2(x))
        x3 = self.mmb3(self.conv3(x))
        x4 = self.mmb4(self.conv4(x))
        # Each branch compresses channels to dim/4 and preserves spatial size:
        # x1/x2/x3/x4: [B, dim/4, H, W]

        outs = torch.cat((x1, x2, x3, x4), 1) + x
        # Concatenate 4 branches back to dim channels:
        # outs: [B, dim, H, W]

        # 最后用 3x3 卷积进一步融合多分支信息，shape 保持不变。
        out = self.project_out(outs)
        # OUT: [B, dim, H, W]

        return out


class SpatialAttention(nn.Module):
    # 作用：生成单通道空间注意力图。
    # 它通过对通道维做 max 聚合，提取“哪里更显著”，
    # 再用卷积 + Sigmoid 输出 [B,1,H,W] 的空间权重。
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # IN: x [B, C, H, W]
        # 在通道维做最大池化，压缩成单通道空间响应图。
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Channel squeeze by max-pooling over C:
        # max_out: [B, 1, H, W]
        x = max_out
        x = self.conv1(x)
        # OUT attention map: [B, 1, H, W]
        return self.sigmoid(x)




if __name__ == '__main__':
    # 这里是一个简单的独立测试入口：
    # 1) 构建 MyNet；
    # 2) 生成两张随机输入；
    # 3) 用 thop 统计模型 FLOPs 与参数量。
    import torch
    import torchvision
    from thop import profile

    model = MyNet().cuda()

    a = torch.randn(1, 1, 128, 128).cuda()
    b = torch.randn(1, 1, 128, 128).cuda()
    flops, params = profile(model, (a,b))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
