# MFS-Fusion 改进实验操作指南：P1 — CMA 替换 MEF 逐元素乘法 + IDC 替换 ESI 多核卷积

> **改进优先级**：P1  
> **对应缺陷**：缺陷 B（跨模态交互深度不足）+ 落差 3（ESI 大核卷积效率问题）  
> **前置依赖**：P0（FSDA 替换 AttenFFT）已完成消融实验并确认有效  
> **改进目标**：  
> - **CMA 替换 MEF**：用跨模态双向通道注意力替换逐元素乘法，实现深层跨模态信息交互  
> - **IDC 替换 ESI**：用 Inception Depthwise Convolution 替换 ESI 中的多核标准卷积，在保持多尺度感受野的同时大幅降低参数量和 FLOPs  
> **预期收益**：SCD 指标提升 2–5%（尤其在医学图像融合中），总 FLOPs 降低 20–25%，推理延迟降低约 20%

---

## 一、实验前准备：确认基线状态

P1 改进在 P0 的基础上进行。在动手修改代码之前，必须明确当前的基线是什么，以及 P1 的两项改进如何在消融实验中被隔离评估。

### 1.1 确认 P0 基线

P1 实验的"基线"根据 P0 的结果分为两种情况。

**情况一：P0 改进有效（FSDA 优于 AttenFFT）**。以 FSDA 已集成的模型作为 P1 的基线。此时 P1 的消融实验需要包含"仅 FSDA"和"FSDA + CMA"以及"FSDA + IDC"的对比组，以隔离各改进项的独立贡献。

**情况二：P0 改进无效（FSDA 未优于 AttenFFT）**。将原始 AttenFFT 恢复为默认配置，以原始模型作为 P1 的基线。P1 的改进独立于 P0 推进。

无论选择哪种情况，都需要确保基线模型的指标已完整记录（参见 P0 指南第一节的指标清单），且基线结果可复现。

### 1.2 记录当前 MEF 和 ESI 的运行时行为

在基线模型上添加临时调试钩子，分别打印 MEF 模块和 ESI 模块在 forward 过程中的关键中间张量统计信息。

**MEF 模块调试**：在 `MEF.forward` 中，打印逐元素乘法 `mul_fuse = ir * vi` 的统计信息，以及 SpatialAttention 输出的注意力图 `sa` 的分布。这些数据将用于与 CMA 替换后的跨模态注意力矩阵进行行为对比。

```python
# 在 MEF.forward 中，逐元素乘法后添加：
print(f"[DEBUG MEF] mul_fuse - mean: {mul_fuse.mean():.4f}, std: {mul_fuse.std():.4f}")
print(f"[DEBUG MEF] mul_fuse - min: {mul_fuse.min():.4f}, max: {mul_fuse.max():.4f}")

# 在 SpatialAttention 输出后添加：
print(f"[DEBUG MEF] sa_map - mean: {sa.mean():.4f}, std: {sa.std():.4f}")
print(f"[DEBUG MEF] sa_map - coverage(>0.5): {(sa > 0.5).float().mean():.4f}")
```

**ESI 模块调试**：在 `ESI.forward` 中，打印各核大小分支（1×1、3×3、5×5、7×7）输出特征的 L2 范数，观察不同分支对最终输出的贡献权重分布。

```python
# 在 ESI.forward 中，各分支卷积后添加：
for i, (name, feat) in enumerate(zip(['1x1','3x3','5x5','7x7'], [x1,x2,x3,x4])):
    print(f"[DEBUG ESI] {name} branch - L2 norm: {feat.norm():.4f}, mean: {feat.mean():.4f}")
```

运行几个 batch 的推理，保存打印结果。特别关注 ESI 中 5×5 和 7×7 分支的特征范数是否显著高于 1×1 和 3×3 分支——如果差异不大，说明大核卷积的收益有限，IDC 替换的收益/成本比更高。

---

## 二、改进 A 详解：CMA 替换 MEF 的逐元素乘法融合

### 2.1 当前 MEF 的问题分析

MEF 模块的核心融合逻辑是 `mul_fuse = ir * vi`，即 IR 和 VIS 特征的逐元素乘法。这种交互方式存在三个层面的局限性。

第一，交互维度单一。逐元素乘法仅在空间维度上捕获共激活模式（两个模态在哪些位置同时有强响应），完全忽略了通道维度的跨模态依赖（IR 的哪些通道特征与 VIS 的哪些通道特征互补）。

第二，信息流单向。`mul_fuse` 被送入 SpatialAttention 生成统一的注意力图 `sa`，然后对 IR 和 VIS 施加相同的空间加权。两个模态接收到的注意力信号是相同的，不存在差异化的跨模态信息注入。

第三，梯度路径不完整。在反向传播中，IR 编码器只接收来自 `ir * vi` 中 `ir` 的梯度（经 `vi` 缩放），而不直接接收来自 VIS 分支内容特征的梯度信号。这限制了两个编码器学习真正互补表示的能力。

### 2.2 CMA 模块的设计原理

CMA（Cross-Modal Attention）通过通道级双向交叉注意力解决上述三个问题。其核心机制如下。

**通道注意力矩阵**：将 IR 特征经全局平均池化后作为 Query（形状 [B, C, 1]），VIS 特征经全局平均池化后生成 Key 和 Value（形状 [B, C, 1]），计算 Q × K^T 得到 [C, C] 的通道相关矩阵，经 Softmax 归一化后乘以 Value，完成 VIS→IR 的通道级信息注入。反向（IR→VIS）同理。

**gamma 可学习缩放**：CMA 引入可学习标量 `gamma`（初始化为 0），控制交叉注意力的强度。训练初期 gamma 接近 0，网络行为接近原始 MEF；随着训练推进，gamma 逐渐增大，交叉注意力逐步介入。这种渐进式引入策略确保训练稳定性。

**与原始 SpatialAttention 的互补关系**：CMA 在通道维度建模跨模态依赖，而空间注意力在空间维度建模显著性区域。两者正交互补，可以串联使用。

### 2.3 CMA 模块的完整实现代码

在 `Net/MyNet.py` 中新增以下类，建议插入在 `class MEF` 定义之前：

```python
class CrossModalAttention(nn.Module):
    """跨模态通道注意力（Cross-Modal Attention）。
    通过 Q×K^T 通道相关矩阵实现一个模态向另一个模态的信息注入。
    使用可学习的 gamma 参数控制交叉注意力强度，确保训练初期的稳定性。
    
    Args:
        dim (int): 输入通道数
    """
    def __init__(self, dim):
        super(CrossModalAttention, self).__init__()
        self.query_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.key_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.value_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 初始化为 0，渐进式引入
        self.softmax = nn.Softmax(dim=-1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_query, x_kv):
        """
        Args:
            x_query: [B, C, H, W] 接收信息的模态特征（如 IR）
            x_kv:    [B, C, H, W] 提供信息的模态特征（如 VIS）
        Returns:
            out: [B, C, H, W] 经跨模态注意力增强后的 x_query
        """
        B, C, H, W = x_query.shape

        # 通道级 Query/Key/Value（通过全局平均池化压缩空间维度）
        q = self.query_conv(self.avg_pool(x_query))  # [B, C, 1, 1]
        k = self.key_conv(self.avg_pool(x_kv))       # [B, C, 1, 1]
        v = self.value_conv(x_kv)                     # [B, C, H, W]（保持空间维度）

        # 计算通道注意力矩阵
        q = q.view(B, C, -1)           # [B, C, 1]
        k = k.view(B, C, -1)           # [B, C, 1]
        attn = torch.bmm(q, k.transpose(1, 2))  # [B, C, C]
        attn = attn / (C ** 0.5)        # 缩放
        attn = self.softmax(attn)        # [B, C, C] 归一化

        # 加权求和：通道级信息注入
        v_flat = v.view(B, C, -1)       # [B, C, H*W]
        out = torch.bmm(attn, v_flat)   # [B, C, H*W]
        out = out.view(B, C, H, W)

        # gamma 控制的渐进式残差
        out = self.gamma * out + x_query

        return out
```

### 2.4 改进后的 MEF 模块实现

将原始 MEF 模块的融合逻辑替换为 CMA 双向交叉注意力。在 `Net/MyNet.py` 中新增 `class ImprovedMEF`：

```python
class ImprovedMEF(nn.Module):
    """改进的多尺度空间注意力融合模块。
    用 CMA 双向通道注意力替换逐元素乘法，实现深层跨模态信息交互。
    保留 ASI/FSDA 多尺度频域细化分支。
    
    Args:
        dim (int): 输入通道数
        use_fsda (bool): 是否使用 FSDA（P0 改进）替代 AttenFFT
    """
    def __init__(self, dim, use_fsda=True):
        super(ImprovedMEF, self).__init__()
        # 跨模态双向通道注意力（替换 mul_fuse + SpatialAttention）
        self.cma_vi2ir = CrossModalAttention(dim)  # VIS 信息注入 IR
        self.cma_ir2vi = CrossModalAttention(dim)  # IR 信息注入 VIS

        # 空间注意力保留（但改为对各模态独立施加，而非共享）
        self.sa_ir = SpatialAttention()
        self.sa_vi = SpatialAttention()

        # ASI/FSDA 多尺度细化（根据 P0 结果选择）
        self.feir = ASI(dim, use_fsda=use_fsda)
        self.fevi = ASI(dim, use_fsda=use_fsda)

        # 融合卷积
        self.cov = conv3x3_bn_relu(2 * dim, dim)

    def forward(self, ir, vi):
        """
        Args:
            ir: [B, C, H, W] 红外特征
            vi: [B, C, H, W] 可见光特征
        Returns:
            ir_out, vi_out, fused: 各 [B, C, H, W]
        """
        # Step 1: 双向跨模态通道注意力（替换 mul_fuse = ir * vi）
        ir_cross = self.cma_vi2ir(ir, vi)   # VIS → IR 信息注入
        vi_cross = self.cma_ir2vi(vi, ir)   # IR → VIS 信息注入

        # Step 2: 独立空间注意力（替换共享的 SpatialAttention）
        sa_ir = self.sa_ir(ir_cross)
        sa_vi = self.sa_vi(vi_cross)
        ir_sa = ir_cross * sa_ir + ir_cross   # 注意力加权 + 残差
        vi_sa = vi_cross * sa_vi + vi_cross

        # Step 3: ASI/FSDA 多尺度频域细化 + 残差
        ir_out = self.feir(ir_sa) + ir
        vi_out = self.fevi(vi_sa) + vi

        # Step 4: 拼接融合
        out = self.cov(torch.cat([ir_out, vi_out], dim=1))

        return ir_out, vi_out, out
```

### 2.5 将 ImprovedMEF 集成到 MyNet

在 `class MyNet` 的 `__init__` 方法中，找到 MEF 实例化代码：

```python
self.fuse0 = MEF(hidden_chans[0])   # MEF(48)
self.fuse1 = MEF(hidden_chans[1])   # MEF(96)
self.fuse2 = MEF(hidden_chans[2])   # MEF(192)
```

替换为：

```python
self.fuse0 = ImprovedMEF(hidden_chans[0], use_fsda=True)
self.fuse1 = ImprovedMEF(hidden_chans[1], use_fsda=True)
self.fuse2 = ImprovedMEF(hidden_chans[2], use_fsda=True)
```

**接口兼容性验证**：ImprovedMEF 的 forward 签名与原始 MEF 完全一致——输入 `(ir, vi)`，输出 `(ir_out, vi_out, out)`——因此 MyNet 的 forward 方法无需修改。

**消融实验控制**：建议通过配置参数控制使用哪种 MEF：

```python
class MyNet(nn.Module):
    def __init__(self, in_chans=1, hidden_chans=[48, 96, 192], pool_ratio=[8, 6, 4],
                 out_chans=1, linear=True, use_fsda=True, use_cma=True):
        # ...
        FuseModule = ImprovedMEF if use_cma else MEF
        self.fuse0 = FuseModule(hidden_chans[0]) if not use_cma else ImprovedMEF(hidden_chans[0], use_fsda)
        self.fuse1 = FuseModule(hidden_chans[1]) if not use_cma else ImprovedMEF(hidden_chans[1], use_fsda)
        self.fuse2 = FuseModule(hidden_chans[2]) if not use_cma else ImprovedMEF(hidden_chans[2], use_fsda)
```

---

## 三、改进 B 详解：IDC 替换 ESI 的多核标准卷积

### 3.1 当前 ESI 的效率问题分析

ESI 模块使用四种核大小（1×1、3×3、5×5、7×7）的标准卷积将输入通道从 `dim` 投影到 `dim//4`。以 `dim=48` 为例，各分支的参数量和计算量如下。

1×1 卷积：参数量 = 48 × 12 × 1 = 576，FLOPs（128×128 输入）= 576 × 128 × 128 ≈ 9.4M。

3×3 卷积：参数量 = 48 × 12 × 9 = 5,184，FLOPs = 5,184 × 128 × 128 ≈ 84.9M。

5×5 卷积：参数量 = 48 × 12 × 25 = 14,400，FLOPs = 14,400 × 128 × 128 ≈ 235.9M。

7×7 卷积：参数量 = 48 × 12 × 49 = 28,224，FLOPs = 28,224 × 128 × 128 ≈ 462.4M。

5×5 和 7×7 分支的参数量和计算量分别是 3×3 分支的 2.8 倍和 5.4 倍。ESI 在解码器的每一级都被调用（包括最高分辨率 256×256），还在输出头中额外调用一次。大核卷积在高分辨率特征图上的计算开销极为显著。

### 3.2 IDC 的设计原理

IDC（Inception Depthwise Convolution）来自 InceptionNeXt 论文的核心思想：并非所有通道都需要大核空间卷积。IDC 将输入通道拆分为四个子组，对每个子组施加不同复杂度的操作。

**Identity 分支**（约 25% 通道）：不做任何空间卷积，直接透传。这些通道已包含足够的特征信息，无需额外空间混合。

**方形核 DWConv 分支**（约 25% 通道）：使用 `square_kernel_size × square_kernel_size`（如 3×3）的 Depthwise Conv，捕获局部正方形感受野。

**水平带状核 DWConv 分支**（约 25% 通道）：使用 `1 × band_kernel_size`（如 1×11）的 Depthwise Conv，捕获水平方向的长距离依赖。

**垂直带状核 DWConv 分支**（约 25% 通道）：使用 `band_kernel_size × 1`（如 11×1）的 Depthwise Conv，捕获垂直方向的长距离依赖。

水平和垂直带状核的组合等效于一个 `band_kernel_size × band_kernel_size` 的大核感受野，但参数量从 O(k²) 降为 O(k)。以 k=11 为例，标准 11×11 DWConv 参数量为 121/通道，而 1×11 + 11×1 的组合仅为 22/通道，降低 82%。

### 3.3 IDC 模块的完整实现代码

在 `Net/MyNet.py` 中新增以下类，建议插入在 `class ESI` 定义之前：

```python
class InceptionDWConv2d(nn.Module):
    """Inception Depthwise Convolution（IDC）。
    将通道拆分为四组：Identity / 方形核 DWConv / 水平带状核 DWConv / 垂直带状核 DWConv，
    在保持大感受野的同时大幅降低参数量和 FLOPs。
    
    替换 ESI 中的 1×1/3×3/5×5/7×7 标准卷积。
    
    Args:
        in_channels (int): 输入通道数（与 ESI 的 dim 对应）
        square_kernel_size (int): 方形核大小，默认 3
        band_kernel_size (int): 带状核大小，默认 11（等效 11×11 感受野）
        branch_ratio (float): Identity 分支占比，默认 0.25
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.25):
        super(InceptionDWConv2d, self).__init__()
        
        gc = int(in_channels * branch_ratio)  # Identity 分支通道数
        remaining = in_channels - gc
        gc_sq = remaining // 3                # 方形核分支通道数
        gc_bh = remaining // 3                # 水平带状核分支通道数
        gc_bv = remaining - gc_sq - gc_bh     # 垂直带状核分支通道数（余数归入此分支）
        
        self.gc = gc
        self.gc_sq = gc_sq
        self.gc_bh = gc_bh
        self.gc_bv = gc_bv
        
        # 方形核 DWConv
        self.dwconv_sq = nn.Conv2d(
            gc_sq, gc_sq, kernel_size=square_kernel_size,
            padding=square_kernel_size // 2, groups=gc_sq
        )
        # 水平带状核 DWConv
        self.dwconv_bh = nn.Conv2d(
            gc_bh, gc_bh, kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2), groups=gc_bh
        )
        # 垂直带状核 DWConv
        self.dwconv_bv = nn.Conv2d(
            gc_bv, gc_bv, kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0), groups=gc_bv
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        # 通道拆分
        x_id, x_sq, x_bh, x_bv = torch.split(
            x, [self.gc, self.gc_sq, self.gc_bh, self.gc_bv], dim=1
        )
        
        # 各分支处理
        # x_id: Identity，直接透传
        x_sq = self.dwconv_sq(x_sq)
        x_bh = self.dwconv_bh(x_bh)
        x_bv = self.dwconv_bv(x_bv)
        
        # 拼接恢复
        out = torch.cat([x_id, x_sq, x_bh, x_bv], dim=1)
        return out
```

### 3.4 改进后的 ESI 模块实现

用 IDC 替换 ESI 中的四组标准卷积。新模块 `ImprovedESI` 将 IDC 的四组输出分别送入 MambaLayer，保持长程依赖建模能力。

```python
class ImprovedESI(nn.Module):
    """改进的增强空间实例模块。
    用 IDC 替换原始的 1×1/3×3/5×5/7×7 标准卷积，
    通过 Inception 式通道拆分 + DWConv 实现高效多尺度特征提取。
    MambaLayer 长程依赖建模保持不变。
    
    Args:
        dim (int): 输入通道数
        square_kernel_size (int): IDC 方形核大小，默认 3
        band_kernel_size (int): IDC 带状核大小，默认 11
    """
    def __init__(self, dim, square_kernel_size=3, band_kernel_size=11):
        super(ImprovedESI, self).__init__()
        self.dim = dim
        self.group_dim = dim // 4
        
        # IDC 多尺度特征提取（替换 4 组标准卷积）
        self.idc = InceptionDWConv2d(
            in_channels=dim,
            square_kernel_size=square_kernel_size,
            band_kernel_size=band_kernel_size,
            branch_ratio=0.25
        )
        
        # 通道降维投影（IDC 输出 dim 通道 → 分为 4 组，每组 dim//4）
        # IDC 是 DWConv 不改变通道数，需要在 IDC 后加 1×1 Conv 对齐到 dim//4
        self.proj1 = nn.Conv2d(dim, self.group_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(dim, self.group_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(dim, self.group_dim, kernel_size=1)
        self.proj4 = nn.Conv2d(dim, self.group_dim, kernel_size=1)
        
        # 每分支一个独立的 MambaLayer（保持不变）
        self.mmb1 = MambaLayer(self.group_dim)
        self.mmb2 = MambaLayer(self.group_dim)
        self.mmb3 = MambaLayer(self.group_dim)
        self.mmb4 = MambaLayer(self.group_dim)
        
        # 融合投影（保持不变）
        self.project_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
    
    def forward(self, x):
        """
        Args:
            x: [B, dim, H, W]
        Returns:
            out: [B, dim, H, W]
        """
        # IDC 多尺度空间特征提取
        x_idc = self.idc(x)  # [B, dim, H, W]（通道数不变，但不同子组经历不同空间操作）
        
        # 分 4 组投影 + MambaLayer
        x1 = self.mmb1(self.proj1(x_idc))  # [B, dim//4, H, W]
        x2 = self.mmb2(self.proj2(x_idc))  # [B, dim//4, H, W]
        x3 = self.mmb3(self.proj3(x_idc))  # [B, dim//4, H, W]
        x4 = self.mmb4(self.proj4(x_idc))  # [B, dim//4, H, W]
        
        # 拼接 + 全局残差 + 融合投影
        out = torch.cat([x1, x2, x3, x4], dim=1) + x  # [B, dim, H, W]
        out = self.project_out(out)
        
        return out
```

**设计说明**：上述实现方案 A 使用 IDC 作为统一的空间操作，然后通过 4 个独立的 1×1 Conv 投影到 4 组 MambaLayer。另一种可选方案 B 是保持 ESI 的"4 组并行"结构，但将每组的标准卷积替换为对应的 IDC 子操作。

**方案 B（可选，更保守的替换策略）**：

```python
class ImprovedESI_B(nn.Module):
    """方案 B：逐分支替换，保持原始 ESI 的并行结构。
    - 1×1 分支：保持不变（已是最轻量）
    - 3×3 分支：替换为 3×3 DWConv + 1×1 Conv（深度可分离卷积）
    - 5×5 分支：替换为 1×5 DWConv + 5×1 DWConv + 1×1 Conv（带状核分解）
    - 7×7 分支：替换为 1×7 DWConv + 7×1 DWConv + 1×1 Conv（带状核分解）
    """
    def __init__(self, dim):
        super(ImprovedESI_B, self).__init__()
        self.dim = dim
        group_dim = dim // 4
        
        # 1×1 分支：保持不变
        self.conv1 = nn.Conv2d(dim, group_dim, kernel_size=1)
        
        # 3×3 分支：标准卷积 → 深度可分离卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),  # DWConv
            nn.Conv2d(dim, group_dim, kernel_size=1),                   # Pointwise
        )
        
        # 5×5 分支：标准卷积 → 带状核分解
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 5), padding=(0, 2), groups=dim),
            nn.Conv2d(dim, dim, kernel_size=(5, 1), padding=(2, 0), groups=dim),
            nn.Conv2d(dim, group_dim, kernel_size=1),
        )
        
        # 7×7 分支：标准卷积 → 带状核分解
        self.conv4 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=dim),
            nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0), groups=dim),
            nn.Conv2d(dim, group_dim, kernel_size=1),
        )
        
        # MambaLayer 和融合投影保持不变
        self.mmb1 = MambaLayer(group_dim)
        self.mmb2 = MambaLayer(group_dim)
        self.mmb3 = MambaLayer(group_dim)
        self.mmb4 = MambaLayer(group_dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
    
    def forward(self, x):
        x1 = self.mmb1(self.conv1(x))
        x2 = self.mmb2(self.conv2(x))
        x3 = self.mmb3(self.conv3(x))
        x4 = self.mmb4(self.conv4(x))
        out = torch.cat([x1, x2, x3, x4], dim=1) + x
        out = self.project_out(out)
        return out
```

**两种方案的对比**：

方案 A（IDC 统一操作）结构更简洁，参数更少，但 4 个 MambaLayer 的输入来自同一个 IDC 输出的不同投影，多样性可能不如方案 B。方案 B（逐分支替换）保持了原始 ESI "不同空间操作→不同 Mamba" 的设计哲学，结构更保守但更易理解。建议在消融实验中同时测试两种方案。

### 3.5 将 ImprovedESI 集成到 Decode 和 Last

ESI 在网络中有两个使用位置：Decode（解码器的每一级）和 Last（输出头）。

**Decode 中的替换**：找到 `class Decode` 的 `__init__` 方法，将所有 `ESI(dim)` 替换为 `ImprovedESI(dim)` 或 `ImprovedESI_B(dim)`。

**Last 中的替换**：找到 `class MyNet` 中 `self.last` 的定义，将其中的 `ESI(hidden_chans[0])` 替换为 `ImprovedESI(hidden_chans[0])`。

**消融实验控制**：

```python
class MyNet(nn.Module):
    def __init__(self, ..., use_idc=True, idc_variant='A'):
        # ...
        if use_idc:
            ESIModule = ImprovedESI if idc_variant == 'A' else ImprovedESI_B
        else:
            ESIModule = ESI
        # 在 Decode 和 Last 中使用 ESIModule
```

---

## 四、代码改动总览

### 4.1 需要修改的文件清单

| 文件路径 | 修改类型 | 涉及的类/函数 | 改动说明 |
|---------|---------|-------------|---------|
| `Net/MyNet.py` | **新增** | `class CrossModalAttention` | 跨模态通道注意力模块 |
| `Net/MyNet.py` | **新增** | `class ImprovedMEF` | CMA 双向注意力融合模块 |
| `Net/MyNet.py` | **新增** | `class InceptionDWConv2d` | IDC 多尺度高效卷积模块 |
| `Net/MyNet.py` | **新增** | `class ImprovedESI` / `ImprovedESI_B` | IDC 版本的 ESI 模块 |
| `Net/MyNet.py` | **适配** | `class MyNet.__init__` | 替换 MEF→ImprovedMEF，添加控制参数 |
| `Net/MyNet.py` | **适配** | `class Decode.__init__` | 替换 ESI→ImprovedESI |
| `Net/MyNet.py` | **适配** | `class MyNet.__init__`（last 部分） | 替换 ESI→ImprovedESI |

### 4.2 不需要修改的文件

`Net/pvt.py`、`Net/MobileViT.py`、`Net/interact.py`、`models/loss_vif.py`、`train.py`、`models/model_plain.py`、`config.json` 均不需要修改。MyNet 的 forward 方法也不需要修改（接口兼容）。

---

## 五、修改后的完整性验证

### 5.1 形状一致性验证

```python
import torch
from Net.MyNet import MyNet

# 测试 CMA + IDC 全改进版本
model = MyNet(in_chans=1, hidden_chans=[48, 96, 192], out_chans=1,
              use_fsda=True, use_cma=True, use_idc=True)
x = torch.randn(2, 1, 256, 256)
y = torch.randn(2, 1, 256, 256)
out = model(x, y)
assert out.shape == (2, 1, 256, 256), f"输出形状错误: {out.shape}"
print("形状验证通过")
```

### 5.2 梯度流验证

重点验证 CMA 的 gamma 参数和 IDC 的各分支 DWConv 是否都能接收到梯度：

```python
model = MyNet(in_chans=1, hidden_chans=[48, 96, 192], out_chans=1,
              use_fsda=True, use_cma=True, use_idc=True)
x = torch.randn(2, 1, 256, 256)
y = torch.randn(2, 1, 256, 256)
out = model(x, y)
loss = out.mean()
loss.backward()

# 验证 CMA 参数
for name, param in model.named_parameters():
    if 'cma' in name or 'gamma' in name:
        assert param.grad is not None, f"CMA 参数 {name} 梯度为 None"
        assert not torch.isnan(param.grad).any(), f"CMA 参数 {name} 梯度含 NaN"
        print(f"[OK CMA] {name}: grad norm = {param.grad.norm():.6f}")

# 验证 IDC 参数
for name, param in model.named_parameters():
    if 'idc' in name or 'dwconv' in name:
        assert param.grad is not None, f"IDC 参数 {name} 梯度为 None"
        assert not torch.isnan(param.grad).any(), f"IDC 参数 {name} 梯度含 NaN"
        print(f"[OK IDC] {name}: grad norm = {param.grad.norm():.6f}")

# 特别关注 gamma 的初始梯度方向
for name, param in model.named_parameters():
    if 'gamma' in name:
        print(f"[GAMMA] {name}: value = {param.data.item():.6f}, grad = {param.grad.item():.6f}")

print("梯度流验证通过")
```

### 5.3 参数量与 FLOPs 对比

```python
from thop import profile

configs = {
    '原始模型': dict(use_fsda=False, use_cma=False, use_idc=False),
    '仅 P0(FSDA)': dict(use_fsda=True, use_cma=False, use_idc=False),
    '仅 CMA': dict(use_fsda=False, use_cma=True, use_idc=False),
    '仅 IDC': dict(use_fsda=False, use_cma=False, use_idc=True),
    'P0+P1(全部)': dict(use_fsda=True, use_cma=True, use_idc=True),
}

dummy_x = torch.randn(1, 1, 256, 256)
dummy_y = torch.randn(1, 1, 256, 256)

for name, kwargs in configs.items():
    model = MyNet(in_chans=1, hidden_chans=[48, 96, 192], out_chans=1, **kwargs)
    flops, params = profile(model, inputs=(dummy_x, dummy_y), verbose=False)
    print(f"{name}: {params/1e6:.2f}M params, {flops/1e9:.2f}G FLOPs")
```

**预期结果**：CMA 的引入会略微增加参数量（每个 MEF 增加 3 组 1×1 Conv + gamma，约 3 × dim² × 3 ≈ 额外约 0.1–0.3M），但 IDC 替换 ESI 带来的参数节省（预计 0.5–1.0M）远大于 CMA 的增量，整体净参数量应减少。FLOPs 方面，CMA 的通道注意力计算量很小（[C,C] 矩阵乘法），而 IDC 的 DWConv 相比标准卷积的 FLOPs 降低约 40–60%，整体 FLOPs 应显著下降。

---

## 六、消融实验设计

P1 包含两项独立改进（CMA 和 IDC），消融实验需要分别验证各自的贡献以及组合效应。

### 6.1 实验组 D：CMA 主效果验证（必做）

| 实验编号 | 频域模块 | 融合模块 | 解码器 | 说明 |
|---------|---------|---------|-------|------|
| D0 | FSDA | 原始 MEF | 原始 ESI | P0 基线 |
| D1 | FSDA | ImprovedMEF(CMA) | 原始 ESI | 仅替换融合模块 |
| D2 | FSDA | ImprovedMEF(CMA+独立SA) | 原始 ESI | CMA + 独立空间注意力 |

**核心比较**：D1 vs D0 验证 CMA 替换逐元素乘法的收益；D2 vs D1 验证独立空间注意力（IR/VIS 各用独立的 SA）相对于共享空间注意力的增益。

**重点关注指标**：SCD（相关性差异之和）和 FMI（特征互信息），这两个指标最能反映跨模态互补信息的保留程度。在医学图像融合数据集（PET-MRI、SPECT-MRI）上的效果尤其值得关注。

### 6.2 实验组 E：IDC 主效果验证（必做）

| 实验编号 | 频域模块 | 融合模块 | 解码器 | 说明 |
|---------|---------|---------|-------|------|
| E0 | FSDA | 原始 MEF | 原始 ESI | P0 基线 |
| E1 | FSDA | 原始 MEF | ImprovedESI(方案A) | IDC 统一操作 |
| E2 | FSDA | 原始 MEF | ImprovedESI_B(方案B) | 逐分支替换 |

**核心比较**：E1/E2 vs E0 验证 IDC 替换 ESI 是否能在不损失精度的前提下降低计算量；E1 vs E2 对比两种 IDC 集成方案的精度-效率权衡。

**重点关注指标**：除融合质量指标外，特别关注参数量、FLOPs 和推理延迟的变化。IDC 的核心价值在于效率提升，因此即使精度持平（各指标变化 ±0.5% 以内），只要效率显著改善，IDC 替换即可认为成功。

### 6.3 实验组 F：CMA + IDC 组合效应（必做）

| 实验编号 | 频域模块 | 融合模块 | 解码器 | 说明 |
|---------|---------|---------|-------|------|
| F0 | FSDA | 原始 MEF | 原始 ESI | P0 基线 |
| F1 | FSDA | ImprovedMEF(CMA) | 原始 ESI | 仅 CMA |
| F2 | FSDA | 原始 MEF | ImprovedESI(最优方案) | 仅 IDC |
| F3 | FSDA | ImprovedMEF(CMA) | ImprovedESI(最优方案) | CMA + IDC |

**核心比较**：F3 vs F0 验证 P1 整体收益；F3 vs F1/F2 检验 CMA 和 IDC 是否存在正向协同效应（F3 的提升是否大于 F1 和 F2 各自提升之和）。

### 6.4 实验组 G：CMA 内部消融（推荐做）

| 实验编号 | 配置 | 说明 |
|---------|------|------|
| G1 | 单向 CMA（仅 VIS→IR） | 验证单向信息注入的效果 |
| G2 | 单向 CMA（仅 IR→VIS） | 对比两个方向的贡献差异 |
| G3 | 双向 CMA（完整版） | 验证双向交互的协同效应 |
| G4 | 双向 CMA + gamma 固定为 1（跳过渐进式引入） | 验证 gamma 渐进机制的必要性 |

实现方式：在 ImprovedMEF 的 forward 中通过条件判断控制。例如 G1 实验中，`vi_cross = vi`（跳过 IR→VIS 注入）。G4 实验中，将 CMA 的 `self.gamma` 初始化为 1 而非 0。

### 6.5 实验组 H：IDC 超参数消融（可选做）

| 实验编号 | band_kernel_size | 说明 |
|---------|:---:|------|
| H1 | 7 | 较小带状核（等效 7×7 感受野） |
| H2 | 11 | 默认带状核（等效 11×11 感受野） |
| H3 | 15 | 较大带状核（等效 15×15 感受野） |

观察带状核大小对融合质量和效率的影响。预期 band_kernel_size=11 是精度-效率的最佳平衡点（与原始 ESI 的 7×7 标准卷积感受野接近）。

---

## 七、训练策略与超参数

### 7.1 训练配置

所有参数与 P0 基线完全一致：学习率 5e-5、batch size 2、训练图像尺寸 256×256、EMA 衰减 0.999、损失函数 `fusion_loss_vif`（权重 λ₁=10, λ₂=20, λ₃=5, λ₄=1）、训练轮数 300 epochs。

随机种子同样固定为 3 个（42, 123, 2024），每组实验跑 3 次取均值和标准差。

### 7.2 CMA 特有的训练注意事项

**gamma 的收敛监控**：在训练过程中每隔 10 个 epoch 记录各 CMA 模块中 gamma 参数的值。正常情况下，gamma 应从 0 缓慢增长到 0.1–0.5 的范围。如果 gamma 收敛到接近 0（< 0.01），说明交叉注意力对当前损失函数的贡献不显著，需要审视 CMA 是否与损失函数的监督信号匹配。如果 gamma 增长过快（训练前 10 epoch 即超过 1.0），可能导致训练不稳定，建议添加 gamma 的上界裁剪（`self.gamma = nn.Parameter(torch.zeros(1).clamp(max=2.0))`）。

```python
# 在训练循环中添加 gamma 监控
if epoch % 10 == 0:
    for name, param in model.named_parameters():
        if 'gamma' in name:
            print(f"[Epoch {epoch}] {name} = {param.data.item():.4f}")
```

**CMA 通道注意力矩阵的分析**：在验证阶段保存 CMA 中 `attn` 矩阵（[C, C]），可视化其热力图。理想情况下，该矩阵应显示出非对角线上的显著元素，表明 CMA 学习到了有意义的跨模态通道关联。如果矩阵接近单位矩阵，说明 CMA 退化为恒等映射，交叉注意力未发挥作用。

### 7.3 IDC 特有的训练注意事项

**带状核权重的初始化**：IDC 的带状核 DWConv 默认使用 PyTorch 的 Kaiming 初始化。由于带状核的参数量较少，初始化的随机性对训练初期的影响可能较大。如果观察到训练初期 loss 震荡，可以尝试将 IDC 的带状核权重初始化为正态分布 N(0, 0.01)，或使用零初始化配合残差连接。

**各分支梯度范数的均衡性**：在训练过程中监控 IDC 四个分支（Identity、方形核、水平带状核、垂直带状核）的梯度范数。如果某个分支的梯度范数显著大于其他分支，可能表明该分支主导了优化方向，其他分支的贡献被边缘化。

---

## 八、评估与结果分析

### 8.1 定量指标记录模板

每个实验配置需要在 MSRS、TNO、RoadScene 三个数据集上填写以下表格：

| 指标 | P0基线(D0) | +CMA(D1) | +IDC-A(E1) | +IDC-B(E2) | +CMA+IDC(F3) | Δ(CMA) | Δ(IDC) | Δ(组合) |
|------|-----------|----------|----------|----------|-------------|--------|--------|---------|
| EN | | | | | | | | |
| MI | | | | | | | | |
| FMI | | | | | | | | |
| QAB/F | | | | | | | | |
| SCD | | | | | | | | |
| AG | | | | | | | | |
| SF | | | | | | | | |
| SD | | | | | | | | |
| VIF | | | | | | | | |
| MS-SSIM | | | | | | | | |
| Params (M) | | | | | | | | |
| FLOPs (G) | | | | | | | | |
| 推理延迟 (ms) | | | | | | | | |

### 8.2 CMA 特有的定性分析

除标准融合结果可视化外，CMA 改进需要额外进行以下定性分析。

**跨模态注意力矩阵可视化**：保存 CMA 中 VIS→IR 方向和 IR→VIS 方向的 [C, C] 注意力矩阵，绘制热力图。观察两个方向的注意力矩阵是否对称、是否存在明显的通道对应关系。在 VIS-IR 融合中，预期 IR 的热敏通道与 VIS 的亮度通道之间存在强关联；在 PET-MRI 融合中，预期 PET 的代谢活性通道与 MRI 的解剖结构通道之间存在互补关联。

**gamma 收敛轨迹绘制**：将各 MEF 层中 CMA 的 gamma 值绘制为训练 epoch 的函数曲线。不同尺度的 MEF（48、96、192 通道）的 gamma 收敛速度和终值可能不同，这反映了跨模态交互在不同特征尺度上的重要性差异。

**融合结果对比——聚焦跨模态互补区域**：选择 IR 和 VIS 信息差异最大的场景（如夜间行人场景：IR 中行人清晰、VIS 中行人模糊）。对比 CMA 前后的融合结果，观察 CMA 是否有效地将 IR 中的行人信息注入到 VIS 分支，使最终融合图像中行人的可见性和清晰度得到提升。

### 8.3 IDC 特有的定性分析

**各分支特征响应可视化**：保存 IDC 四个分支（Identity、方形核、水平带状核、垂直带状核）的输出特征图，按通道取均值后显示为热力图。观察不同分支是否捕获了预期的空间模式——方形核分支应对局部纹理有强响应，水平带状核分支应对水平边缘和纹理有强响应，垂直带状核分支应对垂直边缘有强响应。

**效率对比表**：单独制作一张详细的效率对比表，逐模块分解参数量和 FLOPs：

| 模块位置 | 原始 ESI Params | IDC-A Params | IDC-B Params | 原始 ESI FLOPs | IDC-A FLOPs | IDC-B FLOPs |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| Decode Level 3 (96ch, 64×64) | | | | | | |
| Decode Level 2 (48ch, 128×128) | | | | | | |
| Decode Level 1 (48ch, 256×256) | | | | | | |
| Last (48ch, 256×256) | | | | | | |
| **合计** | | | | | | |

---

## 九、常见问题与排查清单

### 9.1 CMA 相关问题

**问题一：gamma 始终为 0 或接近 0**

可能原因：CMA 的梯度在反向传播中被其他更强的损失信号淹没。排查方向：检查 `gamma` 参数的梯度值；如果梯度绝对值极小（< 1e-6），尝试将 gamma 的学习率乘以 10（使用参数组差异化学习率）。

```python
optimizer = torch.optim.Adam([
    {'params': [p for n, p in model.named_parameters() if 'gamma' not in n], 'lr': 5e-5},
    {'params': [p for n, p in model.named_parameters() if 'gamma' in n], 'lr': 5e-4},
])
```

**问题二：CMA 导致训练 loss 震荡或不收敛**

可能原因：双向交叉注意力在训练初期引入了过强的跨模态干扰。排查方向：先尝试单向 CMA（仅 VIS→IR），确认单向版本能正常训练后再启用双向；或者将 gamma 初始化为更小的值（如 -1，经过训练后 gamma 仍可增长到正值）。

**问题三：CMA 在医学图像融合中效果好但在 VIS-IR 中效果差（或反之）**

可能原因：不同融合任务中跨模态互补性的强度不同。VIS-IR 融合中两个模态的信息重叠度较高，CMA 的通道注意力可能学到接近恒等映射；而 PET-MRI 中两个模态的信息高度互补，CMA 能发挥更大作用。这不一定是问题——CMA 的 gamma 自适应机制应能自动调节注意力强度。如果需要更好的泛化性，可以考虑为不同任务分别调优 gamma 的初始化值。

### 9.2 IDC 相关问题

**问题一：IDC 替换后精度显著下降（> 1%）**

可能原因一：band_kernel_size 过小，大感受野覆盖不足。解决方案：尝试增大 band_kernel_size（从 11 增加到 15 或 21）。

可能原因二：Identity 分支占比过高，有效空间操作的通道数不足。解决方案：将 `branch_ratio` 从 0.25 减小到 0.125，让更多通道参与空间卷积。

可能原因三：DWConv 的表达能力不如标准卷积。解决方案：在 IDC 后添加一个 1×1 Conv（Pointwise Convolution）进行通道混合，补偿 DWConv 缺乏通道间交互的缺陷。

**问题二：IDC 替换后推理延迟未显著改善**

可能原因：MambaLayer 才是 ESI 的计算瓶颈，而非卷积分支。排查方向：使用 `torch.profiler` 或 `torch.cuda.Event` 对 ESI/ImprovedESI 的各子模块进行单独计时，确认计算瓶颈位置。如果 MambaLayer 占总延迟 > 80%，IDC 替换的延迟收益自然有限。

**问题三：带状核分支的梯度消失**

可能原因：长带状核（如 1×11）的有效感受野在训练初期可能难以建立有效的梯度信号。解决方案：在带状核 DWConv 后添加 BatchNorm + ReLU 激活，帮助梯度传播；或使用渐进式核大小策略（训练前 50 epoch 使用 1×7，之后切换到 1×11）。

### 9.3 CMA + IDC 组合相关问题

**问题：CMA 和 IDC 同时引入后效果不如单独引入**

可能原因：CMA 增强的跨模态特征在经过 IDC 的高效卷积时信息损失过大（IDC 的 Identity 分支直接透传的通道未经过 CMA 增强的特征可能质量较差）。排查方向：检查 IDC 的 Identity 分支是否恰好覆盖了 CMA 注入信息最多的通道。解决方案：在 IDC 前添加一个通道混洗（Channel Shuffle）操作，确保 CMA 的增强信息分散到所有分支。

---

## 十、实验时间线建议

| 阶段 | 预计耗时 | 产出 |
|------|---------|------|
| P0 基线确认与 MEF/ESI 运行时分析 | 1 天 | 调试日志、基线确认 |
| CMA 模块编码与形状/梯度验证 | 1 天 | ImprovedMEF 通过验证 |
| IDC 模块编码与形状/梯度验证 | 1 天 | ImprovedESI/ImprovedESI_B 通过验证 |
| 实验组 D（CMA 主效果，3 配置 × 3 种子） | 5–7 天 | 9 组训练结果 |
| 实验组 E（IDC 主效果，3 配置 × 3 种子） | 5–7 天 | 9 组训练结果（可与 D 并行） |
| 实验组 F（CMA+IDC 组合，4 配置 × 3 种子） | 5–7 天 | 12 组训练结果 |
| 实验组 G（CMA 内部消融，4 配置 × 1 种子） | 3–4 天 | 4 组训练结果 |
| 实验组 H（IDC 超参数消融，可选） | 2–3 天 | 3 组训练结果 |
| 结果整理、可视化、分析 | 2–3 天 | 完整实验报告 |
| **合计** | **约 25–35 天** | |

**并行策略建议**：实验组 D 和 E 可以在不同 GPU 上并行执行，因为两者修改的模块不同（D 修改编码器融合模块，E 修改解码器）。实验组 F 需要在 D 和 E 完成后确定最优配置再执行。

**早期终止决策**：如果实验组 D 表明 CMA 在所有数据集上均无显著提升（各指标变化 < 0.3%），建议暂停 CMA 相关的后续消融（G 组），将计算资源集中于 IDC 的优化。反之亦然。

---

## 十一、后续衔接：为 P2 改进做准备

P1 完成后，下一步将进入 P2 优先级的改进。

**P2 改进项一：HFP 插入解码器**（对应缺陷 C——编码-解码频域不对称）。在 Decode 的每一级上采样后、ESI/ImprovedESI 之前插入 HFP（High Frequency Perception）模块，通过 DCT 频域高频提取 + 通道/空间双路注意力，显式恢复上采样过程中丢失的高频细节。

**P2 改进项二：EMA 替换 SpatialAttention**（对应落差 1——SA 简化实现）。将 MEF/ImprovedMEF 中的 SpatialAttention 替换为 EMA（Efficient Multi-scale Attention），通过分组结构在 1×1 和 3×3 两种尺度上建模空间语义。

P2 的改进以 P1 的最优配置为基线。与 P0→P1 的衔接逻辑相同：如果 P1 的某项改进无效，则回退到上一个有效版本作为基线，独立推进 P2。

在全部改进项完成后，需要进行一次"全组合消融"实验，系统评估各改进项的独立贡献和组合效应，确认最终发布版本的模型配置。
