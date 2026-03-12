# MFS-Fusion 核心代码全模块汇总

> **项目全称**：MFS-Fusion — Mamba-Integrated Deep Multi-Modal Image Fusion Framework with Multi-Scale Fourier Enhancement and Spatial Calibration  
> **任务**：多模态图像融合（VIS-IR 可见光-红外 / MEF 多曝光 等）  
> **输入**：两张单通道灰度图 `[B, 1, 256, 256]`（如 IR + Visible）  
> **输出**：融合图像 `[B, 1, 256, 256]`

---

## 目录

- [MFS-Fusion 核心代码全模块汇总](#mfs-fusion-核心代码全模块汇总)
  - [目录](#目录)
  - [1. 整体架构总览](#1-整体架构总览)
  - [2. 主干网络 A — MyNet（PVTv2 骨干）](#2-主干网络-a--mynetpvtv2-骨干)
    - [2.1 初始化参数](#21-初始化参数)
    - [2.2 网络组件与通道数](#22-网络组件与通道数)
    - [2.3 PVTv2 配置](#23-pvtv2-配置)
    - [2.4 Forward 流程](#24-forward-流程)
  - [3. 主干网络 B — MVTNet（MobileViT 骨干）](#3-主干网络-b--mvtnetmobilevit-骨干)
    - [3.1 初始化参数](#31-初始化参数)
    - [3.2 网络组件与通道数](#32-网络组件与通道数)
    - [3.3 MobileViT-Small 各层通道配置](#33-mobilevit-small-各层通道配置)
    - [3.4 MVTNet Decode 通道（4 级解码器）](#34-mvtnet-decode-通道4-级解码器)
  - [4. PVTv2 金字塔视觉 Transformer](#4-pvtv2-金字塔视觉-transformer)
    - [4.1 初始化参数](#41-初始化参数)
    - [4.2 实际使用配置（MyNet 中）](#42-实际使用配置mynet-中)
    - [4.3 每个 Stage 内部结构](#43-每个-stage-内部结构)
  - [5. MobileViT 移动端混合架构](#5-mobilevit-移动端混合架构)
    - [5.1 MobileViT-Small 完整架构](#51-mobilevit-small-完整架构)
  - [6. 注意力模块](#6-注意力模块)
    - [6.1 Spatial Reduction Attention（空间缩减注意力）](#61-spatial-reduction-attention空间缩减注意力)
    - [6.2 AttenFFT（傅里叶频域注意力）](#62-attenfft傅里叶频域注意力)
    - [6.3 SpatialAttention（空间注意力）](#63-spatialattention空间注意力)
    - [6.4 sa\_layer（通道-空间联合注意力 + 通道混洗）](#64-sa_layer通道-空间联合注意力--通道混洗)
    - [6.5 Attention\_conv（跨模态交叉注意力）](#65-attention_conv跨模态交叉注意力)
  - [7. 特征提取与处理模块](#7-特征提取与处理模块)
    - [7.1 ASI（自适应空间实例模块）](#71-asi自适应空间实例模块)
    - [7.2 ESI（增强空间实例模块 — Mamba 多核融合）](#72-esi增强空间实例模块--mamba-多核融合)
    - [7.3 MambaLayer（状态空间模型层）](#73-mambalayer状态空间模型层)
    - [7.4 MEF（多尺度空间注意力融合模块）](#74-mef多尺度空间注意力融合模块)
    - [7.5 Decode（渐进式上采样解码器）](#75-decode渐进式上采样解码器)
  - [8. PVTv2 内部组件详解](#8-pvtv2-内部组件详解)
    - [8.1 OverlapPatchEmbed（重叠 Patch 嵌入）](#81-overlappatchembed重叠-patch-嵌入)
    - [8.2 Block（Transformer Block）](#82-blocktransformer-block)
    - [8.3 Mlp（带 DWConv 的前馈网络）](#83-mlp带-dwconv-的前馈网络)
    - [8.4 DWConv（深度可分离卷积）](#84-dwconv深度可分离卷积)
  - [9. MobileViT 内部组件详解](#9-mobilevit-内部组件详解)
    - [9.1 ConvLayer / InvertedResidual](#91-convlayer--invertedresidual)
    - [9.2 MobileViTBlock](#92-mobilevitblock)
    - [9.3 TransformerEncoder（标准 Transformer 编码器）](#93-transformerencoder标准-transformer-编码器)
    - [9.4 MultiHeadAttention](#94-multiheadattention)
  - [10. interact.py 交互模块](#10-interactpy-交互模块)
    - [10.1 FMultiOrderDWConv](#101-fmultiorderdwconv)
    - [10.2 MultiOrderAvgPool / MultiOrderMaxPool](#102-multiorderavgpool--multiordermaxpool)
    - [10.3 TransformerBlock](#103-transformerblock)
    - [10.4 pure 模块](#104-pure-模块)
  - [11. 损失函数](#11-损失函数)
    - [11.1 总损失构成](#111-总损失构成)
    - [11.2 各分项损失](#112-各分项损失)
    - [11.3 Sobel 算子](#113-sobel-算子)
  - [12. 训练流水线](#12-训练流水线)
    - [12.1 训练配置（config.json）](#121-训练配置configjson)
    - [12.2 网络加载流程](#122-网络加载流程)
    - [12.3 训练循环](#123-训练循环)
  - [13. 全局通道数速查表](#13-全局通道数速查表)
    - [13.1 MyNet（PVTv2 骨干）完整通道流](#131-mynetpvtv2-骨干完整通道流)
    - [13.2 MambaLayer 内部参数](#132-mambalayer-内部参数)
    - [13.3 sa\_layer 参数表](#133-sa_layer-参数表)

---

## 1. 整体架构总览

```
┌──────────────────────────── MFS-Fusion 主网络（MyNet）──────────────────────────────┐
│                                                                                      │
│  输入 x (IR): [B,1,256,256]               输入 y (VIS): [B,1,256,256]               │
│       │                                          │                                   │
│  ┌────▼────┐                                ┌────▼────┐                              │
│  │ pre_x   │  Conv3×3, 1→48                │ pre_y   │  Conv3×3, 1→48              │
│  └────┬────┘  → [B,48,256,256]              └────┬────┘  → [B,48,256,256]           │
│       │  保存 short_x 残差                        │  保存 short_y 残差                │
│       │                                          │                                   │
│  ═════╪══════════ PVTv2 双流编码器 ═══════════════╪═══════════════════════════════    │
│       │                                          │                                   │
│  ┌────▼──────────┐  Stage 0               ┌─────▼──────────┐                        │
│  │ patch_embed1  │  Conv3×3,s=2,48→48     │ patch_embed1   │                        │
│  │ block1 ×2     │  SR-Attn(head=1,sr=8)  │ block1 ×2      │                        │
│  │ norm1         │  → [B,48,128,128]       │ norm1          │ → [B,48,128,128]       │
│  └────┬──────────┘                         └─────┬──────────┘                        │
│       └──────────────┬───────────────────────────┘                                   │
│                ┌─────▼─────┐                                                         │
│                │  MEF(48)  │  空间注意力 + ASI + 双流融合                              │
│                └─────┬─────┘  → fuses[0]: [B,48,128,128]                             │
│       ┌──────────────┼───────────────────────────┐                                   │
│  ┌────▼──────────┐  │  Stage 1               ┌───▼──────────┐                        │
│  │ patch_embed2  │  Conv3×3,s=2,48→96       │ patch_embed2  │                        │
│  │ block2 ×2     │  SR-Attn(head=2,sr=4)    │ block2 ×2     │                        │
│  │ norm2         │  → [B,96,64,64]           │ norm2         │ → [B,96,64,64]        │
│  └────┬──────────┘                           └───┬──────────┘                        │
│       └──────────────┬───────────────────────────┘                                   │
│                ┌─────▼─────┐                                                         │
│                │  MEF(96)  │  → fuses[1]: [B,96,64,64]                               │
│       ┌──────────────┼───────────────────────────┐                                   │
│  ┌────▼──────────┐   │  Stage 2              ┌───▼──────────┐                        │
│  │ patch_embed3  │  Conv3×3,s=2,96→192      │ patch_embed3  │                        │
│  │ block3 ×2     │  SR-Attn(head=4,sr=2)    │ block3 ×2     │                        │
│  │ norm3         │  → [B,192,32,32]          │ norm3         │ → [B,192,32,32]       │
│  └────┬──────────┘                           └───┬──────────┘                        │
│       └──────────────┬───────────────────────────┘                                   │
│                ┌─────▼─────┐                                                         │
│                │ MEF(192)  │  → fuses[2]: [B,192,32,32]                              │
│                └─────┬─────┘                                                         │
│                                                                                      │
│  ═══════════════════ 解码器（Decode）════════════════════════════════════════════      │
│                                                                                      │
│  fuses[2] ──► conv_up3: Conv1×1(192→96)+BN+GELU+↑2+ESI(96)+sa(96)                  │
│               → up3: [B,96,64,64]                                                    │
│                      ↓                                                               │
│  cat(up3, fuses[1]) → [B,192,64,64]                                                 │
│  ──► conv_up2: Conv1×1(192→48)+BN+GELU+↑2+ESI(48)+sa(48)                           │
│               → up2: [B,48,128,128]                                                  │
│                      ↓                                                               │
│  cat(up2, fuses[0]) → [B,96,128,128]                                                │
│  ──► conv_up1: Conv1×1(96→48)+BN+GELU+↑2+ESI(48)+sa(48)                            │
│               → up1: [B,48,256,256]                                                  │
│                                                                                      │
│  ═══════════════════ 输出头（Last）════════════════════════════════════════════       │
│                                                                                      │
│  up1 + short_x + short_y → [B,48,256,256]                                           │
│  ──► Conv3×3(48→48)+BN+GELU → ESI(48) → sa_layer(48) → Conv3×3(48→1)              │
│  ──► 最终输出: [B,1,256,256]                                                         │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 主干网络 A — MyNet（PVTv2 骨干）

**文件**：`Net/MyNet.py`  
**类名**：`class MyNet(nn.Module)`

### 2.1 初始化参数

```python
def __init__(self, in_chans=1, hidden_chans=[48, 96, 192], pool_ratio=[8, 6, 4], out_chans=1, linear=True)
```

### 2.2 网络组件与通道数

| 组件 | 操作 | 输入通道 | 输出通道 | 卷积核 / 说明 |
|:---|:---|:---:|:---:|:---|
| `pre_x` | `nn.Conv2d` | 1 | 48 | kernel=3, stride=1, pad=1 |
| `pre_y` | `nn.Conv2d` | 1 | 48 | kernel=3, stride=1, pad=1 |
| `un_x` | `PyramidVisionTransformerV2` | 48 | [48,96,192] | 3 stages, embed_dims=[48,96,192] |
| `un_y` | `PyramidVisionTransformerV2` | 48 | [48,96,192] | 3 stages（与 un_x 结构相同、参数独立）|
| `fuse0` | `MEF` | 48 | 48 | Stage0 双流融合 |
| `fuse1` | `MEF` | 96 | 96 | Stage1 双流融合 |
| `fuse2` | `MEF` | 192 | 192 | Stage2 双流融合 |
| `dec` | `Decode` | (48,96,192) | 48 | 渐进式上采样解码器 |
| `last` | Sequential | 48 | 1 | Conv3×3(48→48)+BN+GELU → ESI(48) → sa(48) → Conv3×3(48→1) |

### 2.3 PVTv2 配置

```python
PyramidVisionTransformerV2(
    in_chans=48, embed_dims=[48, 96, 192],
    num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4],
    depths=[2, 2, 2], sr_ratios=[8, 4, 2],
    num_stages=3, linear=True
)
```

### 2.4 Forward 流程

```python
def forward(self, x, y):
    # x: [B,1,256,256], y: [B,1,256,256]
    x = self.pre_x(x)    # → [B,48,256,256]
    y = self.pre_y(y)    # → [B,48,256,256]
    short_x, short_y = x, y   # 全分辨率残差

    for i in range(3):  # 3 个 PVTv2 stage
        # OverlapPatchEmbed: stride=2 下采样 + 通道升维
        x, H, W = patch_embed_x(x)   # [B, H*W, C_i]
        for blk in block_x: x = blk(x, H, W)   # Transformer Blocks
        x = norm_x(x) → reshape → [B, C_i, H, W]
        
        y, H, W = patch_embed_y(y)
        for blk in block_y: y = blk(y, H, W)
        y = norm_y(y) → reshape → [B, C_i, H, W]
        
        x, y, out = self.fes[i](x, y)   # MEF 融合
        fuses.append(out)
    
    # fuses[0]:[B,48,128,128], fuses[1]:[B,96,64,64], fuses[2]:[B,192,32,32]
    out = self.dec(fuses[0], fuses[1], fuses[2])   # → [B,48,256,256]
    out = self.last(out + short_x + short_y)       # → [B,1,256,256]
    return out
```

---

## 3. 主干网络 B — MVTNet（MobileViT 骨干）

**文件**：`Net/MVTNet.py`  
**类名**：`class MyNet(nn.Module)`（与 MyNet.py 中同名，通过 `select_network.py` 配置切换）

### 3.1 初始化参数

```python
def __init__(self, in_chans=1, hidden_chans=[32, 64, 96, 128], out_chans=1, linear=True)
```

### 3.2 网络组件与通道数

| 组件 | 操作 | 输入通道 | 输出通道 | 说明 |
|:---|:---|:---:|:---:|:---|
| `p_x` / `p_y` | Conv3×3 | 1 | 3 | 灰度 → 伪 RGB |
| `pre_x` / `pre_y` | Conv3×3 | 3 | 32 | 提取初始特征 |
| `un_x` / `un_y` | `mobile_vit_small()` | 3 | 多尺度 | 预训练 MobileViT-Small |
| `fuse0` | MEF | 32 | 32 | Layer1 后融合 |
| `fuse1` | MEF | 64 | 64 | Layer2 后融合 |
| `fuse2` | MEF | 96 | 96 | Layer3 后融合 |
| `fuse3` | MEF | 128 | 128 | Layer4 后融合 |
| `dec` | Decode | (32,64,96,128) | 32 | 4 级解码器 |
| `last` | Sequential | 32 | 1 | Conv3×3(32→32)+BN+GELU → ESI(32) → sa(32) → Conv3×3(32→1) |

### 3.3 MobileViT-Small 各层通道配置

| 层 | 类型 | 输出通道 | Transformer 通道 | FFN 维度 | heads | blocks |
|:--|:--|:---:|:---:|:---:|:---:|:---:|
| conv_1 | Conv3×3, s=2 | 16 | — | — | — | — |
| layer_1 | MV2 Block | 32 | — | — | — | ×1 |
| layer_2 | MV2 Block | 64 | — | — | — | ×3, s=2 |
| layer_3 | MobileViTBlock | 96 | 144 | 288 | 4 | ×2, s=2 |
| layer_4 | MobileViTBlock | 128 | 192 | 384 | 4 | ×4, s=2 |
| layer_5 | MobileViTBlock | 160 | 240 | 480 | 4 | ×3, s=2 |

### 3.4 MVTNet Decode 通道（4 级解码器）

| 步骤 | 输入通道 | 输出通道 | 操作 |
|:---|:---:|:---:|:---|
| conv_up4 | 128 | 96 | Conv1×1(128→96)+BN+GELU+↑2+ESI(96)+sa(96) |
| conv_up3 | 96×2=192 | 64 | cat(up4,fuses[2]) → Conv1×1(192→64)+BN+GELU+↑2+ESI(64)+sa(64) |
| conv_up2 | 64×2=128 | 32 | cat(up3,fuses[1]) → Conv1×1(128→32)+BN+GELU+↑2+ESI(32)+sa(32) |
| conv_up1 | 32×2=64 | 32 | cat(up2,fuses[0]) → Conv1×1(64→32)+BN+GELU+↑2+ESI(32)+sa(32) |

---

## 4. PVTv2 金字塔视觉 Transformer

**文件**：`Net/pvt.py`  
**类名**：`class PyramidVisionTransformerV2(nn.Module)`

### 4.1 初始化参数

```python
def __init__(self, img_size=224, patch_size=16, in_chans=1,
             embed_dims=[48, 96, 192, 384],
             num_heads=[1, 2, 4, 8],
             mlp_ratios=[4, 4, 4, 4],
             depths=[3, 4, 6, 3],
             sr_ratios=[8, 4, 2, 1],
             num_stages=4, linear=True, ...)
```

### 4.2 实际使用配置（MyNet 中）

| Stage | in_chans | embed_dim | num_heads | mlp_ratio | depth | sr_ratio | Patch 大小 | stride | 输出分辨率(H=256) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 0 | 48 | 48 | 1 | 4 | 2 | 8 | 3×3 | 2 | [B,48,128,128] |
| 1 | 48 | 96 | 2 | 4 | 2 | 4 | 3×3 | 2 | [B,96,64,64] |
| 2 | 96 | 192 | 4 | 4 | 2 | 2 | 3×3 | 2 | [B,192,32,32] |

### 4.3 每个 Stage 内部结构

```
输入 [B, C_prev, H, W]
  │
  ├─ OverlapPatchEmbed: Conv3×3(C_prev→C_i, stride=2) + LayerNorm
  │     → [B, H/2 × W/2, C_i]  (token 序列)
  │
  ├─ Block × depth[i]:
  │     ├─ LayerNorm → Attention(SR) → DropPath → 残差加
  │     └─ LayerNorm → Mlp(DWConv) → DropPath → 残差加
  │     → [B, N, C_i]
  │
  └─ LayerNorm → reshape → [B, C_i, H/2, W/2]
```

---

## 5. MobileViT 移动端混合架构

**文件**：`Net/MobileViT.py`  
**类名**：`class MobileViT(nn.Module)`

### 5.1 MobileViT-Small 完整架构

```
输入 [B, 3, H, W]
  │
  ├─ conv_1: Conv3×3(3→16, s=2) + BN + SiLU    → [B,16, H/2, W/2]
  ├─ layer_1: InvertedResidual(16→32, exp=4)     → [B,32, H/2, W/2]
  ├─ layer_2: InvertedResidual×3(32→64, s=2)     → [B,64, H/4, W/4]
  ├─ layer_3: InvRes(64→96,s=2) + MobileViTBlock → [B,96, H/8, W/8]
  ├─ layer_4: InvRes(96→128,s=2) + MobileViTBlock→ [B,128,H/16,W/16]
  └─ layer_5: InvRes(128→160,s=2) + MobileViTBlock→[B,160,H/32,W/32]
```

---

## 6. 注意力模块

### 6.1 Spatial Reduction Attention（空间缩减注意力）

**文件**：`Net/pvt.py` → `class Attention`

PVTv2 中的核心注意力机制，通过对 K/V 进行空间下采样减少计算量。

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1, linear=False):
        self.q = nn.Linear(dim, dim)              # Q 投影: dim → dim
        self.kv = nn.Linear(dim, dim * 2)          # K/V 投影: dim → 2×dim
        self.proj = nn.Linear(dim, dim)            # 输出投影: dim → dim
        
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                # 空间下采样: [B,C,H,W] → [B,C,H/sr,W/sr]
                self.norm = nn.LayerNorm(dim)
        else:  # linear=True
            self.pool = nn.AdaptiveAvgPool2d(7)  # 固定池化到 7×7
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
```

**数据流（以 Stage0 为例，dim=48, heads=1, sr_ratio=8）**：

| 步骤 | 操作 | 张量形状 |
|:---|:---|:---|
| 输入 | token 序列 | [B, 16384, 48] (128×128=16384) |
| Q 投影 | Linear(48→48) | [B, 16384, 48] → [B, 1, 16384, 48] |
| K/V 空间缩减 | Conv(48→48, k=8, s=8) | [B, 48, 128, 128] → [B, 48, 16, 16] |
| K/V 展平 | reshape + Linear | [B, 256, 48] → K,V: [B, 1, 256, 48] |
| 注意力 | Q×Kᵀ×scale | [B, 1, 16384, 256] |
| 输出 | attn×V → proj | [B, 16384, 48] |

**复杂度**：从 O(N²) 降低到 O(N × N/sr²)，sr=8 时降低 64 倍。

### 6.2 AttenFFT（傅里叶频域注意力）

**文件**：`Net/MyNet.py` → `class AttenFFT`

在频域进行多头自注意力计算，捕获全局频率特征。

```python
class AttenFFT(nn.Module):
    def __init__(self, dim, num_heads=2, bias=False):
        # Q/K/V 生成：逐通道深度可分离卷积
        self.qkv1conv_1 = nn.Conv2d(dim, dim, 3, pad=1, groups=dim)  # V 分支
        self.qkv1conv_3 = nn.Conv2d(dim, dim, 3, pad=1, groups=dim)  # K 分支
        self.qkv1conv_5 = nn.Conv2d(dim, dim, 3, pad=1, groups=dim)  # Q 分支
        
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 可学习温度系数
        
        # 频域权重分支
        self.weight = nn.Sequential(
            nn.Conv2d(dim, dim, 1),     # 1×1 conv: dim→dim
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),     # 1×1 conv: dim→dim
            nn.Sigmoid()
        )
        
        self.project_out = nn.Conv2d(dim * 2, dim, 1)  # 融合投影: 2×dim → dim
```

**数据流（以 dim=12 / ASI 中 dim//4 为例, heads=2）**：

| 步骤 | 操作 | 张量形状 |
|:---|:---|:---|
| 输入 x | 特征图 | [B, 12, H, W] |
| DWConv → q_s/k_s/v_s | 3×3 depthwise conv | [B, 12, H, W] |
| FFT2 变换 | torch.fft.fft2 | [B, 12, H, W] (复数) |
| 多头拆分 | rearrange | [B, 2, 6, H×W] |
| 频域注意力 | q·kᵀ × temperature → softmax(complex) | [B, 2, 6, 6] |
| 加权求和 | attn × v → IFFT2 → abs | [B, 12, H, W] |
| 频域加权分支 | weight(FFT(x).real) × FFT(x) → IFFT → abs | [B, 12, H, W] |
| 融合输出 | cat → Conv1×1(24→12) | [B, 12, H, W] |

### 6.3 SpatialAttention（空间注意力）

**文件**：`Net/MyNet.py` → `class SpatialAttention`

通过通道维度 Max Pooling 压缩后用 7×7 大核卷积生成空间注意力图。

```python
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        self.conv1 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        # 输入 1 通道 → 输出 1 通道
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [B, C, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # → [B, 1, H, W]
        x = self.conv1(max_out)                           # → [B, 1, H, W]
        return self.sigmoid(x)                            # → [B, 1, H, W] (注意力图)
```

### 6.4 sa_layer（通道-空间联合注意力 + 通道混洗）

**文件**：`Net/MyNet.py` → `class sa_layer`

将特征分组后分别做通道注意力和空间注意力，再通过通道混洗融合。

```python
class sa_layer(nn.Module):
    def __init__(self, channel, groups=4):
        self.groups = groups  # 默认 4 组
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 可学习参数（以 channel=48 为例）
        self.cweight = nn.Parameter(zeros(1, 48//(2×4)=6, 1, 1))  # 通道注意力权重
        self.cbias = nn.Parameter(ones(1, 6, 1, 1))                # 通道注意力偏置
        self.sweight = nn.Parameter(zeros(1, 6, 1, 1))             # 空间注意力权重
        self.sbias = nn.Parameter(ones(1, 6, 1, 1))                # 空间注意力偏置
        self.gn = nn.GroupNorm(6, 6)                                # GroupNorm(6 groups, 6 channels)
```

**数据流（channel=48, groups=4）**：

| 步骤 | 操作 | 张量形状 |
|:---|:---|:---|
| 输入 | x | [B, 48, H, W] |
| 分组 reshape | B×groups 个子组 | [B×4, 12, H, W] |
| 拆分 | chunk(2, dim=1) | x_0/x_1: [B×4, 6, H, W] |
| 通道注意力 | AvgPool → cw×xn+cb → Sigmoid × x_0 | [B×4, 6, H, W] |
| 空间注意力 | GN → sw×xs+sb → Sigmoid × x_1 | [B×4, 6, H, W] |
| 合并 | cat → reshape | [B, 48, H, W] |
| 通道混洗 | channel_shuffle(groups=2) | [B, 48, H, W] |

### 6.5 Attention_conv（跨模态交叉注意力）

**文件**：`Net/interact.py` → `class Attention_conv`

IR 和 VIS 双模态之间的交叉注意力机制，使用池化降低 K/V 分辨率。

```python
class Attention_conv(nn.Module):
    def __init__(self, dim=48, num_heads=3, bias=True):
        # VIS 分支
        self.q_vi = nn.Conv2d(dim, dim, 1)                   # Q: 1×1 conv, dim→dim
        self.kv_vi = nn.Conv2d(dim, dim * 2, 1)              # KV: 1×1 conv, dim→2×dim
        self.project_out_vi = nn.Conv2d(dim, dim, 1)         # 输出: 1×1 conv
        self.MultiOrderPool_vi = MultiOrderAvgPool(dim)       # 1×1+5×5_DW 池化
        # IR 分支（结构相同）
        self.q_ir = nn.Conv2d(dim, dim, 1)
        self.kv_ir = nn.Conv2d(dim, dim * 2, 1)
        self.project_out_ir = nn.Conv2d(dim, dim, 1)
        self.MultiOrderPool_ir = MultiOrderMaxPool(dim)
```

**交叉注意力机制**：
- VIS 的 Q 与 IR 的 K/V 交互 → 增强 VIS 中的 IR 信息
- IR 的 Q 与 VIS 的 K/V 交互 → 增强 IR 中的 VIS 信息

---

## 7. 特征提取与处理模块

### 7.1 ASI（自适应空间实例模块）

**文件**：`Net/MyNet.py` → `class ASI`

将通道均分为 4 组，每组使用不同膨胀率的卷积 + FFT 注意力，捕获多尺度感受野。

```python
class ASI(nn.Module):
    def __init__(self, dim):
        self.dim = dim // 4      # 每组通道数
        # 4 组膨胀卷积 (以 dim=48 为例, 每组 12 通道)
        self.conv1 = nn.Conv2d(12, 12, k=3, dilation=1, pad=1)   # 感受野 3×3
        self.conv2 = nn.Conv2d(12, 12, k=3, dilation=2, pad=2)   # 感受野 7×7
        self.conv3 = nn.Conv2d(12, 12, k=3, dilation=3, pad=3)   # 感受野 13×13（有效）
        self.conv4 = nn.Conv2d(12, 12, k=3, dilation=4, pad=4)   # 感受野 17×17（有效）
        # 对应 4 组 FFT 注意力
        self.fft1 = AttenFFT(12)
        self.fft2 = AttenFFT(12)
        self.fft3 = AttenFFT(12)
        self.fft4 = AttenFFT(12)
        self.project_out = nn.Conv2d(dim, dim, k=3, pad=1)       # 融合: dim→dim
```

**数据流（dim=48）**：

```
输入 x: [B, 48, H, W]
  │
  ├─ chunk(4, dim=1) → x1/x2/x3/x4: [B, 12, H, W]
  │
  ├─ x1 = AttenFFT(Conv3×3_d1(x1)) + x1    ← 小感受野 + 频域注意力
  ├─ x2 = AttenFFT(Conv3×3_d2(x2)) + x2    ← 中感受野 + 频域注意力
  ├─ x3 = AttenFFT(Conv3×3_d3(x3)) + x3    ← 大感受野 + 频域注意力
  ├─ x4 = AttenFFT(Conv3×3_d4(x4)) + x4    ← 超大感受野 + 频域注意力
  │
  ├─ cat(x1,x2,x3,x4) + x → [B, 48, H, W]  (全局残差)
  └─ Conv3×3(48→48) → [B, 48, H, W]
```

### 7.2 ESI（增强空间实例模块 — Mamba 多核融合）

**文件**：`Net/MyNet.py` → `class ESI`

将特征用不同大小卷积核抽取后分别送入 Mamba 层进行状态空间建模。

```python
class ESI(nn.Module):
    def __init__(self, dim):
        self.dim = dim // 4      # 每分支通道 (以 dim=48 为例, 每分支 12)
        # 4 种不同核大小的投影卷积
        self.conv1 = nn.Conv2d(48, 12, k=1, pad=0)   # 1×1 conv: dim→dim/4
        self.conv2 = nn.Conv2d(48, 12, k=3, pad=1)   # 3×3 conv: dim→dim/4
        self.conv3 = nn.Conv2d(48, 12, k=5, pad=2)   # 5×5 conv: dim→dim/4
        self.conv4 = nn.Conv2d(48, 12, k=7, pad=3)   # 7×7 conv: dim→dim/4
        # 每分支一个独立的 MambaLayer
        self.mmb1 = MambaLayer(12)   # d_model=12
        self.mmb2 = MambaLayer(12)
        self.mmb3 = MambaLayer(12)
        self.mmb4 = MambaLayer(12)
        self.project_out = nn.Conv2d(dim, dim, k=3, pad=1)   # 融合: dim→dim
```

**数据流（dim=48）**：

```
输入 x: [B, 48, H, W]
  │
  ├─ conv1(x): [B,48,H,W] → [B,12,H,W] → MambaLayer → x1: [B,12,H,W]
  ├─ conv2(x): [B,48,H,W] → [B,12,H,W] → MambaLayer → x2: [B,12,H,W]
  ├─ conv3(x): [B,48,H,W] → [B,12,H,W] → MambaLayer → x3: [B,12,H,W]
  ├─ conv4(x): [B,48,H,W] → [B,12,H,W] → MambaLayer → x4: [B,12,H,W]
  │
  ├─ cat(x1,x2,x3,x4) + x → [B, 48, H, W]  (全局残差)
  └─ Conv3×3(48→48) → [B, 48, H, W]
```

### 7.3 MambaLayer（状态空间模型层）

**文件**：`Net/MyNet.py` → `class MambaLayer`

基于 Mamba（Selective State Space Model）的序列建模层，采用**四方向双向扫描**策略。

```python
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        self.nin = Conv1×1(dim→dim)        # 输入投影
        self.norm = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.mamba = Mamba(
            d_model=dim,    # 模型维度
            d_state=16,     # SSM 状态扩展因子
            d_conv=4,       # 局部卷积宽度
            expand=2        # Block 扩展因子
        )
        self.nin2 = Conv1×1(dim→dim)       # 输出投影
        self.norm2 = nn.BatchNorm2d(dim)
        self.relu2 = nn.ReLU(inplace=True)
```

**数据流（dim=12）**：

| 步骤 | 操作 | 张量形状 |
|:---|:---|:---|
| 输入 | x | [B, 12, H, W] |
| 1×1 conv + BN + ReLU | 通道混合 | [B, 12, H, W] |
| 展平 | reshape + transpose | [B, H×W, 12] |
| 四方向 Mamba 扫描 | | |
| → 正向 | mamba(x_flat) | [B, H×W, 12] |
| → 空间翻转 | mamba(flip(dim=2)) → flip_back | [B, H×W, 12] |
| → 通道翻转 | mamba(flip(dim=1)) → flip_back | [B, H×W, 12] |
| → 双翻转 | mamba(flip(dim=[1,2])) → flip_back | [B, H×W, 12] |
| 平均融合 | (4 路求和) / 4 | [B, H×W, 12] |
| 恢复 + 残差 | reshape + skip | [B, 12, H, W] |
| 1×1 conv + BN + ReLU | 输出投影 | [B, 12, H, W] |

### 7.4 MEF（多尺度空间注意力融合模块）

**文件**：`Net/MyNet.py` → `class MEF`

双流特征的核心融合模块，用于在编码器每个 Stage 末尾融合 IR/VIS 分支。

```python
class MEF(nn.Module):
    def __init__(self, dim):
        self.spatial_attention = SpatialAttention()  # 7×7 空间注意力
        self.feir = ASI(dim)                         # IR 分支 ASI
        self.fevi = ASI(dim)                         # VIS 分支 ASI
        self.cov = conv3x3_bn_relu(2 * dim, dim)    # 融合卷积: 2×dim → dim
```

**数据流（dim=48）**：

```
ir: [B,48,H,W]   vi: [B,48,H,W]
        │                │
        └──── × ────────┘  逐元素乘 → mul_fuse: [B,48,H,W]
              │
        SpatialAttention → sa: [B,1,H,W]
              │
        ┌─────┴──────┐
  vi = vi×sa + vi    ir = ir×sa + ir     ← 空间注意力加权 + 残差
        │                   │
  vi = ASI(vi) + vi     ir = ASI(ir) + ir  ← ASI 多尺度细化 + 残差
        │                   │
        └──── cat ─────────┘  → [B,96,H,W]
              │
        Conv3×3(96→48) + BN + ReLU → out: [B,48,H,W]

返回: ir, vi, out
```

### 7.5 Decode（渐进式上采样解码器）

**文件**：`Net/MyNet.py` → `class Decode`（MyNet 版本，3 级）

```python
class Decode(nn.Module):
    def __init__(self, in1=48, in2=96, in3=192):
        self.conv_up3 = Sequential(
            Conv1×1(192→96) + BN + GELU + Upsample×2 + ESI(96) + sa_layer(96)
        )
        self.conv_up2 = Sequential(
            Conv1×1(192→48) + BN + GELU + Upsample×2 + ESI(48) + sa_layer(48)
            # 输入 192 = cat(96, 96)
        )
        self.conv_up1 = Sequential(
            Conv1×1(96→48) + BN + GELU + Upsample×2 + ESI(48) + sa_layer(48)
            # 输入 96 = cat(48, 48)
        )
```

**数据流**：

| 步骤 | 输入 | 操作 | 输出 |
|:---|:---|:---|:---|
| up3 | x3: [B,192,32,32] | Conv1×1(192→96)+BN+GELU+↑2×+ESI+sa | [B,96,64,64] |
| up2 | cat(up3,x2): [B,192,64,64] | Conv1×1(192→48)+BN+GELU+↑2×+ESI+sa | [B,48,128,128] |
| up1 | cat(up2,x1): [B,96,128,128] | Conv1×1(96→48)+BN+GELU+↑2×+ESI+sa | [B,48,256,256] |

---

## 8. PVTv2 内部组件详解

### 8.1 OverlapPatchEmbed（重叠 Patch 嵌入）

**文件**：`Net/pvt.py` → `class OverlapPatchEmbed`

用步长为 2 的 3×3 卷积实现空间下采样 + 通道升维，保留局部空间上下文。

```python
class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size=3, stride=2, in_chans, embed_dim):
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2,
                              padding=1)   # 3×3 conv, stride=2: 空间减半 + 通道变换
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.proj(x)        # [B, C_in, H, W] → [B, C_out, H/2, W/2]
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # → [B, H/2×W/2, C_out] (token 序列)
        x = self.norm(x)
        return x, H, W
```

**各 Stage 通道变化**：

| Stage | in_chans | embed_dim | 结果 |
|:---:|:---:|:---:|:---|
| 0 | 48 | 48 | Conv3×3(48→48, s=2): 分辨率减半 |
| 1 | 48 | 96 | Conv3×3(48→96, s=2): 通道×2 + 分辨率减半 |
| 2 | 96 | 192 | Conv3×3(96→192, s=2): 通道×2 + 分辨率减半 |

### 8.2 Block（Transformer Block）

**文件**：`Net/pvt.py` → `class Block`

标准 Pre-Norm Transformer Block，包含 SR-Attention 和 DWConv-MLP。

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, sr_ratio=1, linear=False):
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, sr_ratio=sr_ratio, linear=linear)
        self.drop_path = DropPath(...)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden)
    
    def forward(self, x, H, W):
        x = x + drop_path(attn(norm1(x), H, W))   # 注意力残差
        x = x + drop_path(mlp(norm2(x), H, W))     # FFN 残差
        return x  # [B, N, C]
```

**MLP 隐藏层维度**：

| Stage | dim | mlp_ratio | hidden_features |
|:---:|:---:|:---:|:---:|
| 0 | 48 | 4 | 192 |
| 1 | 96 | 4 | 384 |
| 2 | 192 | 4 | 768 |

### 8.3 Mlp（带 DWConv 的前馈网络）

**文件**：`Net/pvt.py` → `class Mlp`

```python
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, linear=False):
        self.fc1 = nn.Linear(in_features, hidden_features)    # 升维
        self.dwconv = DWConv(hidden_features)                   # 深度可分离卷积
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)    # 降维
        if linear:
            self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, H, W):
        x = self.fc1(x)         # [B,N,C] → [B,N,hidden]
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)  # token→图→DWConv→token
        x = self.act(x)
        x = self.fc2(x)         # [B,N,hidden] → [B,N,C]
        return x
```

### 8.4 DWConv（深度可分离卷积）

**文件**：`Net/pvt.py` → `class DWConv`

```python
class DWConv(nn.Module):
    def __init__(self, dim=768):
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        # groups=dim: 每通道独立卷积，参数量 = dim × 3 × 3
    
    def forward(self, x, H, W):
        x = x.transpose(1,2).view(B, C, H, W)   # token → 图
        x = self.dwconv(x)                        # 3×3 DWConv
        x = x.flatten(2).transpose(1,2)           # 图 → token
        return x
```

---

## 9. MobileViT 内部组件详解

### 9.1 ConvLayer / InvertedResidual

**文件**：`Net/MobileViT.py`

**ConvLayer**：标准 Conv2d + BN + SiLU 封装。

**InvertedResidual**（MobileNetV2 倒残差块）：

```python
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        hidden_dim = make_divisible(in_channels * expand_ratio, 8)
        # expand_ratio=4 时 (以 32→64, stride=2 为例):
        #   exp_1x1: Conv1×1(32→128) + BN + SiLU         ← 升维
        #   conv_3x3: DWConv3×3(128→128, s=2) + BN + SiLU ← 深度卷积
        #   red_1x1: Conv1×1(128→64) + BN                  ← 降维(无激活)
```

### 9.2 MobileViTBlock

**文件**：`Net/MobileViT.py` → `class MobileViTBlock`

CNN + Transformer 混合块，先提取局部特征，再在 Patch 级别做全局 Transformer。

```python
class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, transformer_dim, ffn_dim,
                 n_transformer_blocks=2, head_dim=32, patch_h=2, patch_w=2):
        # 局部表示
        conv_3x3_in:  Conv3×3(in_ch→in_ch) + BN + SiLU
        conv_1x1_in:  Conv1×1(in_ch→transformer_dim) (无 BN/Act)
        
        # 全局表示
        global_rep:  TransformerEncoder × n_transformer_blocks + LayerNorm
        
        # 投影 & 融合
        conv_1x1_out: Conv1×1(transformer_dim→in_ch) + BN + SiLU
        fusion:       Conv3×3(2×in_ch→in_ch) + BN + SiLU   ← cat(res, fm)
```

**数据流（以 layer_3: in_ch=96, transformer_dim=144 为例）**：

```
输入 x: [B, 96, H, W]
  │ res = x
  │
  ├─ local_rep:
  │    Conv3×3(96→96) + BN + SiLU → Conv1×1(96→144)
  │    → fm: [B, 144, H, W]
  │
  ├─ unfolding:
  │    [B,144,H,W] → [B×patch_area, num_patches, 144]
  │    patch_h=2, patch_w=2 → patch_area=4
  │
  ├─ TransformerEncoder × 2:
  │    LayerNorm → MultiHeadAttention(144, heads=4) → 残差
  │    LayerNorm → FFN(144→288→144) → 残差
  │
  ├─ folding:
  │    [B×4, N, 144] → [B, 144, H, W]
  │
  ├─ conv_proj: Conv1×1(144→96) + BN + SiLU → [B, 96, H, W]
  │
  └─ fusion: cat(res, fm) → Conv3×3(192→96) + BN + SiLU → [B, 96, H, W]
```

### 9.3 TransformerEncoder（标准 Transformer 编码器）

**文件**：`Net/transformer.py` → `class TransformerEncoder`

```python
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim, num_heads=8, 
                 attn_dropout=0.0, dropout=0.0, ffn_dropout=0.0):
        self.pre_norm_mha = Sequential(
            LayerNorm(embed_dim),
            MultiHeadAttention(embed_dim, num_heads),
            Dropout(dropout)
        )
        self.pre_norm_ffn = Sequential(
            LayerNorm(embed_dim),
            Linear(embed_dim → ffn_latent_dim),   # 升维
            SiLU(),
            Dropout(ffn_dropout),
            Linear(ffn_latent_dim → embed_dim),   # 降维
            Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.pre_norm_mha(x)     # MHA 残差
        x = x + self.pre_norm_ffn(x)     # FFN 残差
        return x
```

### 9.4 MultiHeadAttention

**文件**：`Net/transformer.py` → `class MultiHeadAttention`

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0):
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)   # QKV 联合投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)         # 输出投影
        self.head_dim = embed_dim // num_heads
        self.scaling = head_dim ** -0.5
    
    def forward(self, x_q):
        # [N, P, C] → QKV拆分 → 多头注意力 → 输出投影
        qkv = self.qkv_proj(x_q)  # [N,P,3C]
        # → Q,K,V: [N, heads, P, head_dim]
        # attn = softmax((Q×Kᵀ) × scale)
        # out = attn × V → [N, P, C]
        return self.out_proj(out)
```

---

## 10. interact.py 交互模块

**文件**：`Net/interact.py`

### 10.1 FMultiOrderDWConv

门控深度可分离卷积 FFN：

```python
class FMultiOrderDWConv(nn.Module):
    def __init__(self, embed_dims):
        self.dp = Sequential(
            Conv2d(embed_dims, embed_dims, 1),                          # 1×1 pointwise
            Conv2d(embed_dims, embed_dims, 3, pad=1, groups=embed_dims), # 3×3 depthwise
            GELU()
        )
    
    def forward(self, x):
        return x * self.dp(x)    # 门控乘法
```

### 10.2 MultiOrderAvgPool / MultiOrderMaxPool

特征池化模块，用于降低 K/V 分辨率：

```python
class MultiOrderAvgPool(nn.Module):
    def __init__(self, embed_dims):
        self.DW_conv = Sequential(
            Conv2d(embed_dims, embed_dims, 1),                          # 1×1 pointwise
            Conv2d(embed_dims, embed_dims, 5, pad=2, groups=embed_dims) # 5×5 depthwise
        )
    
    def forward(self, x, x_size):
        return F.adaptive_avg_pool2d(self.DW_conv(x), output_size=x_size)
```

### 10.3 TransformerBlock

双模态融合 Transformer Block（interact.py 中）：

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1):
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.ffn = FMultiOrderDWConv(dim)         # 门控 FFN
        self.cov = nn.Conv2d(2 * dim, dim, 1)     # 融合: 2×dim → dim
    
    def forward(self, ir, vi, x_size):
        ir_a, vi_a = self.pc(ir, vi)    # Pixel Calibration: sigmoid 门控
        x = self.cov(cat(ir_a, vi_a))   # 通道融合
        x = x + self.ffn(self.norm2(x)) # 门控 FFN + 残差
        return x
```

### 10.4 pure 模块

```python
class pure(nn.Module):
    def __init__(self, dim):
        self.pre_process = Conv3×3(dim→dim)
        self.pre_process1 = Conv3×3(dim→dim)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        x = conv1(x)
        xdp = conv2(x)
        x_max = xdp * sigmoid(maxpool(xdp))   # 通道注意力门控
        return x_max + x
```

---

## 11. 损失函数

**文件**：`models/loss_vif.py` → `class fusion_loss_vif`

### 11.1 总损失构成

```python
class fusion_loss_vif(nn.Module):
    def forward(self, image_A, image_B, image_fused):
        loss_l1 = 5 × L_Intensity(A, B, fused)       # 强度损失
        loss_gradient = 20 × L_Grad(A, B, fused)      # 梯度损失
        loss_edge = EdgeLoss(A, B, fused)              # 边缘损失
        loss_SSIM = 10 × (1 - L_SSIM(A, B, fused))   # 结构相似性损失
        total = loss_l1 + loss_gradient + loss_edge + loss_SSIM
        return total, loss_gradient, loss_l1
```

### 11.2 各分项损失

| 损失 | 权重 | 公式 / 说明 |
|:---|:---:|:---|
| **L_Intensity** | ×5 | `L1(fused, max(A, B))`，保留双模态最大亮度 |
| **L_Grad** | ×20 | `L1(Sobel(fused), max(Sobel(A), Sobel(B)))`，保留最强边缘 |
| **EdgeLoss** | ×1 | Laplacian 金字塔边缘 + CharbonnierLoss |
| **L_SSIM** | ×10 | 加权 SSIM：`w_A × SSIM(A,fused) + w_B × SSIM(B,fused)`，权重由梯度均值自适应计算 |

### 11.3 Sobel 算子

```python
class Sobelxy(nn.Module):
    # 水平 Sobel 核 3×3
    kernelx = [[-1,0,1], [-2,0,2], [-1,0,1]]
    # 垂直 Sobel 核 3×3
    kernely = [[1,2,1], [0,0,0], [-1,-2,-1]]
    # 输出 = |conv(x, kernelx)| + |conv(x, kernely)|
```

---

## 12. 训练流水线

**文件**：`train.py` + `models/model_plain.py`

### 12.1 训练配置（config.json）

| 参数 | 值 | 说明 |
|:---|:---|:---|
| task | Fusion | 融合任务 |
| model | plain2 | 双输入模型 |
| net_type | un | 使用 MyNet 网络 |
| n_channels | 1 | 灰度图 |
| embed_dim | 48 | 基础特征通道 |
| H_size | 256 | 训练图像尺寸 |
| batch_size | 2 | 批大小 |
| G_lossfn_type | vif | VIF 融合损失 |
| G_optimizer_lr | 5e-5 | 学习率 |
| E_decay | 0.999 | EMA 指数移动平均 |

### 12.2 网络加载流程

```
config.json (net_type="un")
  → select_network.py: define_G()
    → from Net.MyNet import MyNet as Net
    → netG = Net()
  → model_plain.py: ModelPlain.__init__()
    → self.netG = define_G(opt)
```

### 12.3 训练循环

```python
for epoch in range(300):
    for i, train_data in enumerate(train_loader):
        # train_data: {'A': [B,1,256,256], 'B': [B,1,256,256]}
        model.feed_data(train_data)
        model.optimize_parameters(epoch)
        # 内部: fused = netG(A, B) → loss = fusion_loss_vif(A, B, fused) → backward
    
    # 测试
    for test_data in test_loader:
        A, B, fused = model.test()
        fusion_loss = test_loss(A, B, fused)
        if fusion_loss < best_loss:
            model.save(epoch)  # 保存 Best.pth
```

---

## 13. 全局通道数速查表

### 13.1 MyNet（PVTv2 骨干）完整通道流

| 位置 | 层名 | Conv 类型 | 输入通道 | 输出通道 | 空间分辨率 |
|:---|:---|:---|:---:|:---:|:---|
| **Stem** | pre_x / pre_y | Conv3×3 | 1 | 48 | 256×256 |
| **PVT Stage0** | patch_embed1 | Conv3×3, s=2 | 48 | 48 | 128×128 |
| | block1 (×2) | SR-Attn + MLP | 48 | 48 | 128×128 |
| | SR-Attn.sr | Conv8×8, s=8 | 48 | 48 | 16×16 (K/V) |
| | MLP.fc1 | Linear | 48 | 192 | — |
| | MLP.dwconv | DWConv3×3 | 192 | 192 | 128×128 |
| | MLP.fc2 | Linear | 192 | 48 | — |
| **MEF(48)** | spatial_attn | Conv7×7 | 1 | 1 | 128×128 |
| | ASI.conv1~4 | Conv3×3 d=1/2/3/4 | 12 | 12 | 128×128 |
| | ASI.fft.qkv | DWConv3×3 | 12 | 12 | 128×128 |
| | ASI.fft.weight | Conv1×1→Conv1×1 | 12 | 12 | 128×128 |
| | ASI.fft.proj_out | Conv1×1 | 24 | 12 | 128×128 |
| | ASI.project_out | Conv3×3 | 48 | 48 | 128×128 |
| | MEF.cov | Conv3×3 | 96 | 48 | 128×128 |
| **PVT Stage1** | patch_embed2 | Conv3×3, s=2 | 48 | 96 | 64×64 |
| | block2 (×2) | SR-Attn + MLP | 96 | 96 | 64×64 |
| | SR-Attn.sr | Conv4×4, s=4 | 96 | 96 | 16×16 (K/V) |
| | MLP.fc1 | Linear | 96 | 384 | — |
| | MLP.dwconv | DWConv3×3 | 384 | 384 | 64×64 |
| | MLP.fc2 | Linear | 384 | 96 | — |
| **MEF(96)** | ASI.conv1~4 | Conv3×3 d=1/2/3/4 | 24 | 24 | 64×64 |
| | ASI.fft 各分支 | DWConv3×3 | 24 | 24 | 64×64 |
| | ASI.project_out | Conv3×3 | 96 | 96 | 64×64 |
| | MEF.cov | Conv3×3 | 192 | 96 | 64×64 |
| **PVT Stage2** | patch_embed3 | Conv3×3, s=2 | 96 | 192 | 32×32 |
| | block3 (×2) | SR-Attn + MLP | 192 | 192 | 32×32 |
| | SR-Attn.sr | Conv2×2, s=2 | 192 | 192 | 16×16 (K/V) |
| | MLP.fc1 | Linear | 192 | 768 | — |
| | MLP.dwconv | DWConv3×3 | 768 | 768 | 32×32 |
| | MLP.fc2 | Linear | 768 | 192 | — |
| **MEF(192)** | ASI.conv1~4 | Conv3×3 d=1/2/3/4 | 48 | 48 | 32×32 |
| | ASI.fft 各分支 | DWConv3×3 | 48 | 48 | 32×32 |
| | ASI.project_out | Conv3×3 | 192 | 192 | 32×32 |
| | MEF.cov | Conv3×3 | 384 | 192 | 32×32 |
| **Decode** | conv_up3 | Conv1×1 | 192 | 96 | ↑ 64×64 |
| | ESI(96).conv1~4 | Conv 1/3/5/7 | 96 | 24 | 64×64 |
| | ESI(96).mamba | Mamba(d=24) | 24 | 24 | 64×64 |
| | ESI(96).proj | Conv3×3 | 96 | 96 | 64×64 |
| | conv_up2 | Conv1×1 | 192 | 48 | ↑ 128×128 |
| | ESI(48).conv1~4 | Conv 1/3/5/7 | 48 | 12 | 128×128 |
| | ESI(48).mamba | Mamba(d=12) | 12 | 12 | 128×128 |
| | ESI(48).proj | Conv3×3 | 48 | 48 | 128×128 |
| | conv_up1 | Conv1×1 | 96 | 48 | ↑ 256×256 |
| | ESI(48) + sa(48) | 同上 | — | — | 256×256 |
| **Output** | last.conv1 | Conv3×3 | 48 | 48 | 256×256 |
| | last.ESI | ESI(48) | 48 | 48 | 256×256 |
| | last.sa | sa_layer(48) | 48 | 48 | 256×256 |
| | last.conv2 | Conv3×3 | 48 | 1 | 256×256 |

### 13.2 MambaLayer 内部参数

| 参数 | 说明 | ESI(48)中值 | ESI(96)中值 |
|:---|:---|:---:|:---:|
| d_model | 模型维度 | 12 | 24 |
| d_state | SSM 状态数 | 16 | 16 |
| d_conv | 局部卷积宽 | 4 | 4 |
| expand | 扩展因子 | 2 | 2 |
| 内部扩展维度 | d_model × expand | 24 | 48 |
| nin / nin2 | Conv1×1 | 12→12 | 24→24 |

### 13.3 sa_layer 参数表

| channel | groups | 子组通道 | cweight/cbias 形状 | GN groups |
|:---:|:---:|:---:|:---|:---:|
| 48 | 4 | 6 | [1, 6, 1, 1] | 6 |
| 96 | 4 | 12 | [1, 12, 1, 1] | 12 |

---

> **总结**：MFS-Fusion 采用双流编码器-融合-解码器架构。编码器使用 PVTv2（3 stage，通道 48→96→192）进行多尺度特征提取；每个 Stage 末尾通过 MEF 模块（空间注意力 + ASI 多尺度膨胀卷积 + FFT 频域注意力）融合 IR/VIS 双流特征；解码器通过渐进式上采样恢复分辨率，每级嵌入 ESI（多核 Mamba 状态空间模型）和 sa_layer（通道-空间联合注意力）进行精细化。整体设计融合了 Transformer 全局建模、Mamba 线性复杂度长程依赖、FFT 频域处理和多尺度空间校准等多种先进机制。
