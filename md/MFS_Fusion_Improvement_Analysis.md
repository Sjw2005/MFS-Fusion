# MFS-Fusion 深度缺陷诊断与改进方案

> **分析对象**：MFS-Fusion: Mamba-Integrated Deep Multi-Modal Image Fusion Framework  
> **发表刊物**：Expert Systems With Applications 299 (2026) 130054  
> **分析日期**：2026-03-11  
> **分析定位**：首席研究员视角下的理论缺陷诊断、代码审计与创新改进蓝图

---

## 第一部分：论文与代码的联合缺陷诊断（痛点剖析）

### 1.1 理论局限性分析

#### 缺陷 A：频域注意力的数学不严谨性

MFS-Fusion 的核心创新之一是频域多头自注意力（FMHSA），论文在 Eq.(5)-(6) 中将 Q/K/V 通过 FFT 映射到频域后进行点积注意力计算。然而，论文回避了一个根本性的数学问题：标准 Softmax 归一化定义在实数域上，而 FFT 输出是复数张量。代码中的 `custom_complex_normalization` 对复数的实部和虚部分别做 Softmax，这在数学上破坏了复数的幅值-相位耦合关系。Softmax 的概率归一化语义在复数域中不成立——对实部和虚部独立归一化后重新组合的复数，其幅值和相位分布已经失去了物理意义上的频谱一致性。

**根本原因**：作者将空间域注意力机制直接移植到频域，未针对复数运算的特殊性设计适配的归一化方案。更合理的做法应当是：要么在幅度谱和相位谱上分别建模（如 FSDA 模块所做的那样），要么将复数注意力矩阵取绝对值后在实数域完成归一化再映射回复数空间。

#### 缺陷 B：跨模态交互深度不足

论文声称 MS-FEM 实现了"高效跨模态信息交互"，但从代码的 MEF 模块实现来看，两个模态之间的交互仅限于逐元素乘法（`mul_fuse = ir * vi`）生成一个空间注意力图，随后对两个分支分别加权。这种交互方式本质上是单向的、浅层的——它仅捕获了两个模态在空间位置上的共激活模式，而未建模通道维度的跨模态依赖关系，也未实现真正的双向注意力交互（即 IR 的 Query 对 VIS 的 Key/Value 进行检索，反之亦然）。

尽管代码中 `interact.py` 存在 `Attention_conv` 类实现了跨模态交叉注意力，但在论文主网络 `MyNet.py` 的 forward 流程中并未调用该模块，说明这一更深层次的交互机制实际上被弃用或仅为实验残留。论文中也未提及任何交叉注意力机制的使用。

**根本原因**：作者可能出于计算效率考虑，选择了更轻量的逐元素乘法方案，但这导致跨模态信息的互补性未被充分挖掘，尤其在 PET-MRI 融合中（论文自述"PET 图像的功能信息强度偶尔不足"），这一缺陷尤为突出。

#### 缺陷 C：解码器缺乏频域处理的结构不对称

论文的编码端通过 MS-FEM 在频域进行了精心设计的全局建模，但解码端的 CS-SRM 仅在空间域工作（ESI 多核 Mamba + sa_layer 通道-空间注意力）。这造成了编码-解码之间的信息域不对称：编码器提取并融合的频域特征在经过解码器时缺乏对应的频域细化机制，可能导致高频细节（边缘、纹理）在逐级上采样过程中逐步退化。

**根本原因**：解码器设计遵循了传统 U-Net 的纯空间域范式，未将频域增强的设计哲学延伸到解码阶段。

#### 缺陷 D：Mamba 四方向扫描的信息稀释

CS-SRM 中的 MambaLayer 采用四方向扫描策略（正向、空间翻转、通道翻转、双翻转），最终通过简单平均融合四个方向的输出。这种等权平均假设所有方向的贡献相等，而在实际场景中，不同方向的扫描对不同空间结构（水平边缘、垂直边缘、对角线纹理）的建模效果差异显著。简单平均可能稀释某些方向上有价值的长程依赖信息。

**根本原因**：缺乏对多方向输出的自适应加权机制，未利用内容相关的动态融合策略。

### 1.2 代码实现落差（Code-to-Paper Gap）

#### 落差 1：空间注意力（SA）的简化实现

论文 Fig.2 描述的空间注意力组件使用 MaxPooling 沿通道维度压缩后接 7×7 卷积。代码实现与论文一致，但该设计本身是 CBAM 中空间注意力的简化版——CBAM 原文证明了同时使用 AvgPool 和 MaxPool 的双通道输入优于单一 MaxPool。MFS-Fusion 放弃了 AvgPool 分支，损失了通道维度上的全局平均统计信息，这直接影响空间注意力图的质量。

#### 落差 2：sa_layer 硬编码分组数

sa_layer（通道-空间联合注意力）中 `groups` 参数硬编码为 4，无论输入通道数如何变化。当通道数为 48 时，每组 12 个通道尚可接受；但当通道数为 192 时，每组 48 个通道的分组粒度过粗，可能无法捕获细粒度的通道间依赖。同时，sa_layer 内部的 GroupNorm 组数等于 `channel // (2 * groups)`，在某些通道配置下可能产生不合理的归一化分组。

#### 落差 3：ESI 多核卷积的效率问题

ESI 模块对输入特征分别使用 1×1、3×3、5×5、7×7 四种核大小的标准卷积进行投影，每个分支将通道数从 `dim` 压缩到 `dim//4`。问题在于 5×5 和 7×7 的标准卷积参数量显著高于小核卷积，但 ESI 在解码器的每一级（包括最高分辨率 256×256）都被调用，导致大核卷积在高分辨率特征图上的计算开销极大。论文 Table 7 报告的 77.70G FLOPs 中，ESI 的贡献不容忽视。

#### 落差 4：AttenFFT 中 Q/K/V 的生成方式

论文 Eq.(4) 声称使用"多尺度卷积"在频域中提取 Q/K/V 的表示，暗示 Q/K/V 应各自使用不同尺度（膨胀率）的卷积核。但代码实现中，`qkv1conv_1`、`qkv1conv_3`、`qkv1conv_5` 均为 3×3 的 Depthwise Conv，仅是三个独立的同尺度卷积——这与论文中"multi-scale convolution to extract Q, K, and V respectively"的描述不符。Q/K/V 实际上是在相同尺度下提取的，论文声称的"多尺度感知不同频率分量的能力"在代码中未得到体现。

#### 落差 5：损失函数的权重敏感性

论文 Eq.(16) 中四个损失项的权重（λ₁=10, λ₂=20, λ₃=5, λ₄=1）通过网格搜索确定，但论文未提供不同权重组合下的性能对比或敏感性分析。从代码 `loss_vif.py` 来看，Texture Loss 权重（20）远高于 Edge Loss 权重（1），可能导致模型过度关注梯度保持而忽略边缘连续性，这与论文在医学图像融合中"边缘清晰度不足"的自述缺陷一致。

---

## 第二部分：前沿方法引入与模块匹配（破局策略）

### 2.1 创新方法一：频域幅值-相位解耦注意力

**问题对标**：缺陷 A（频域注意力数学不严谨）+ 落差 4（Q/K/V 非多尺度）

**核心思路**：用 FSDA（Frequency Spectrum Dynamic Aggregation，模块 #69）的幅值-相位解耦策略替换当前 AttenFFT 中数学上不严谨的复数 Softmax。FSDA 将频谱显式分解为幅度谱和相位谱，分别通过独立的 SE 通道注意力网络进行动态建模，避免了在复数域上强行应用 Softmax 的问题。幅度谱控制各频率分量的能量分布，相位谱保持结构一致性，两者通过极坐标重建回复数频谱，再经 IFFT 回到空间域。

**为何优于原方法**：FSDA 在数学上严格尊重了频域信号的幅值-相位双极性质，通过 SE 机制实现通道级自适应加权，比原方法对实部/虚部分别做 Softmax 更具物理合理性。同时，FSDA 的残差设计（`ori_mag + mag`）确保了增强过程的渐进性和稳定性。

**精准模块匹配**：

| 模块名称 | 替换/插入位置 | 预期作用 |
|---------|------------|--------|
| **FSDA（#69）** | 替换 `Net/MyNet.py` 中 `AttenFFT` 类的核心频域注意力计算（`forward` 方法中的 FFT→attention→IFFT 流程） | 以幅值-相位解耦的方式正确建模频域特征，消除复数 Softmax 的数学缺陷，同时通过 SE 通道注意力实现频率分量的自适应加权 |
| **ECA（#3）** | 在 FSDA 的幅度谱/相位谱处理分支中，替代 SE 中的双全连接层，使用 1D 卷积实现更轻量的通道注意力 | 降低 FSDA 引入的参数开销，使频域注意力更加高效，避免 SE 的降维信息损失 |

### 2.2 创新方法二：跨模态双向交叉注意力融合

**问题对标**：缺陷 B（跨模态交互深度不足）

**核心思路**：在 MEF 模块中引入 CMA（Cross-Modal Attention，模块 #52）机制，将当前浅层的逐元素乘法替换为通道级双向交叉注意力。CMA 使 IR 特征的 Query 能够检索 VIS 特征的 Key/Value（反之亦然），通过 `gamma` 可学习缩放参数控制交叉注意力的强度，实现真正的双向深层跨模态信息交互。

**为何优于原方法**：逐元素乘法仅捕获逐像素的共激活关系，而 CMA 在通道维度上建模跨模态的全局相关性（attention map 为 [C, C] 的通道-通道相关矩阵），能够学习到"IR 的哪些通道特征与 VIS 的哪些通道特征互补"这一更高阶的跨模态依赖。这对于 PET-MRI 等功能-结构互补性强的融合任务尤为关键。

**精准模块匹配**：

| 模块名称 | 替换/插入位置 | 预期作用 |
|---------|------------|--------|
| **CMA（#52）** | 替换 `MEF` 类中 `mul_fuse = ir * vi` 及后续的 SpatialAttention 逻辑 | 通过 Q×K^T 通道注意力矩阵实现 IR→VIS 和 VIS→IR 的双向信息注入，弥补逐元素乘法无法捕获通道间跨模态依赖的缺陷 |
| **EMA（#13）** | 替换 MEF 中现有的 `SpatialAttention`（仅 MaxPool 的 7×7 Conv） | EMA 通过分组结构同时建模 1×1 和 3×3 两种尺度的空间语义，并引入跨空间学习进行双分支特征聚合，提供比单一 MaxPool 更丰富的空间注意力信息 |

### 2.3 创新方法三：解码器频域-空间协同细化

**问题对标**：缺陷 C（解码器缺乏频域处理）+ 缺陷 D（Mamba 方向信息稀释）+ 落差 3（ESI 大核效率问题）

**核心思路**：对解码器进行双重改进。首先，用 IDC（Inception Depthwise Convolution，模块 #67）替换 ESI 中效率低下的多核标准卷积，通过通道拆分 + 方形核/带状核并行 + Identity 映射的方式，在保持多尺度感受野的同时大幅降低参数量和 FLOPs。其次，在解码器的每一级上采样后引入轻量级 HFP（High Frequency Perception，模块 #66）模块，通过 DCT 频域高频提取 + 通道/空间双路注意力的方式，在解码阶段显式恢复高频细节，弥补编码-解码之间的频域处理不对称。

**为何优于原方法**：IDC 用 1×k 和 k×1 带状卷积替代 k×k 标准卷积，将参数量从 O(k²) 降为 O(k)，论文 InceptionNeXt 证明其在 ImageNet 上精度不降反升。HFP 将频域增强延伸到解码阶段，使整个网络在频域处理上实现编码-解码对称，从结构层面保障高频信息在逐级上采样中的保真度。

**精准模块匹配**：

| 模块名称 | 替换/插入位置 | 预期作用 |
|---------|------------|--------|
| **IDC（#67）** | 替换 `ESI` 类中的 `self.conv1`~`self.conv4`（1×1/3×3/5×5/7×7 标准卷积） | 通过 Inception 式通道拆分 + 带状卷积实现多尺度特征提取，参数量降低约 60%，FLOPs 降低约 40%，同时保持大感受野覆盖 |
| **HFP（#66）** | 在 Decode 类的每一级 `conv_up` 后、ESI 之前插入 | 通过 DCT 高频提取 + 通道/空间注意力显式恢复上采样过程中丢失的高频细节，弥补解码端频域处理缺失的结构不对称 |

---

## 第三部分：详细的改进方案与原理解释（执行蓝图）

### 3.1 改进一：FSDA 替换 AttenFFT 的频域注意力核心

#### 改进前后的机制对比

**改进前（AttenFFT）**：输入 x → DWConv 生成 q_s/k_s/v_s → FFT2 变换为复数 → 多头拆分 → 复数域 Q×K^T → 对复数的实部/虚部分别 Softmax → 复数加权求和 → IFFT → 取绝对值。

问题链条：Softmax(real) 和 Softmax(imag) 独立归一化后重组为复数，其幅值分布不再反映原始频谱的能量结构。例如，若某频率分量的实部 attention 权重为 0.8 而虚部权重为 0.2，重组后的复数幅值为 √(0.64+0.04)≈0.82，与两个独立 Softmax 的概率语义完全脱节。

**改进后（FSDA 集成）**：输入 x → FFT2 → 显式分离幅度谱 `mag = |F|` 和相位谱 `pha = angle(F)` → 幅度谱经 Conv1×1→BN→ReLU→SELayer→Conv1×1 通道注意力增强 → 相位谱经相同结构独立增强 → 残差连接 `mag_out = ori_mag + enhanced_mag` → 极坐标重建 `F_out = mag_out × exp(j × pha_out)` → IFFT2 回到空间域。

**积极变化**：幅度谱和相位谱在各自的实数子空间中独立处理，数学上严格合法；SE 通道注意力在实数域上的 Sigmoid 门控具有明确的概率语义（每个频率通道的重要性权重）；残差连接确保增强过程的渐进性，避免频谱坍缩。

#### 代码修改建议

```python
# === 伪代码：改进后的 SFE_Block（替换 ASI 中的 AttenFFT） ===
class ImprovedSFE(nn.Module):
    def __init__(self, dim, num_groups=4):
        super().__init__()
        group_dim = dim // num_groups
        # 多尺度膨胀卷积保持不变
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(group_dim, group_dim, 3, dilation=d, padding=d)
            for d in [1, 2, 3, 4]
        ])
        # 用 FSDA 替换 AttenFFT
        self.freq_enhancers = nn.ModuleList([
            FSDA(group_dim) for _ in range(num_groups)
        ])
        self.project_out = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        chunks = x.chunk(4, dim=1)
        outputs = []
        for i, (chunk, dconv, freq) in enumerate(
            zip(chunks, self.dilated_convs, self.freq_enhancers)
        ):
            spatial_feat = dconv(chunk) + chunk      # 空间域多尺度提取 + 残差
            freq_feat = freq(spatial_feat)            # FSDA 频域增强
            outputs.append(freq_feat + chunk)         # 全局残差
        out = torch.cat(outputs, dim=1) + x
        return self.project_out(out)
```

FSDA 内部的 SE 通道注意力可进一步替换为 ECA（模块 #3）以降低参数量：将 `nn.Linear(channel, channel//reduction)` + `nn.Linear(channel//reduction, channel)` 替换为 `nn.Conv1d(1, 1, kernel_size=k, padding=k//2)`，其中 k 根据通道数自适应计算。

### 3.2 改进二：CMA + EMA 重构 MEF 的跨模态融合

#### 改进前后的机制对比

**改进前（MEF）**：`mul_fuse = ir * vi` → `SpatialAttention(mul_fuse)` → `sa = sigmoid(conv7×7(maxpool(mul_fuse)))` → `vi = vi * sa + vi; ir = ir * sa + ir` → ASI 分别细化 → `cat(ir, vi)` → Conv3×3 融合。

问题：空间注意力图由两个模态的逐像素乘积生成，它编码的是"两个模态在哪些空间位置同时有强响应"，但忽略了"一个模态的某些通道特征对另一个模态的哪些通道特征有互补价值"这一更高层次的跨模态关系。

**改进后（CMA + EMA）**：首先通过 CMA 实现通道级跨模态注意力——IR 特征作为 Query，VIS 特征生成 Key/Value，通过 `Q × K^T` 计算 [C, C] 通道相关矩阵，再乘以 Value 完成 VIS→IR 的信息注入；反向同理。然后用 EMA 替换原始 SpatialAttention，EMA 通过分组结构在 1×1 和 3×3 两种尺度上建模空间语义，并通过跨空间学习（矩阵点积捕获像素级成对关系）聚合双分支特征。

**积极变化**：在梯度传播层面，CMA 的通道注意力矩阵 `attn = Softmax(Q × K^T)` 为两个模态的特征建立了显式的梯度通路——当融合损失反向传播时，IR 编码器不仅接收来自自身特征的梯度，还接收经过 CMA 矩阵调制的、来自 VIS 分支的梯度信号，促使两个编码器学习真正互补的特征表示。

#### 代码修改建议

```python
# === 伪代码：改进后的 MEF 模块 ===
class ImprovedMEF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 跨模态通道注意力（双向）
        self.cma_ir2vi = CrossModelAtt(feature_dim=dim, height=1, width=1)
        self.cma_vi2ir = CrossModelAtt(feature_dim=dim, height=1, width=1)
        # 高效多尺度空间注意力
        self.ema = EMA(channels=dim, factor=8)
        # ASI 多尺度细化保持不变
        self.feir = ImprovedSFE(dim)  # 使用改进后的 SFE
        self.fevi = ImprovedSFE(dim)
        self.cov = conv3x3_bn_relu(2 * dim, dim)

    def forward(self, ir, vi):
        # Step 1: 双向跨模态通道注意力
        ir_enhanced = self.cma_vi2ir(ir, vi)   # VIS信息注入IR
        vi_enhanced = self.cma_ir2vi(vi, ir)   # IR信息注入VIS

        # Step 2: EMA 空间注意力替代原始 SpatialAttention
        ir_spatial = self.ema(ir_enhanced)
        vi_spatial = self.ema(vi_enhanced)

        # Step 3: SFE 多尺度频域细化
        ir_out = self.feir(ir_spatial) + ir
        vi_out = self.fevi(vi_spatial) + vi

        # Step 4: 拼接融合
        out = self.cov(torch.cat([ir_out, vi_out], dim=1))
        return ir_out, vi_out, out
```

### 3.3 改进三：IDC + HFP 重构解码器

#### 改进前后的机制对比

**改进前（ESI）**：输入 x [B,48,H,W] → 四条并行路径：Conv1×1(48→12) / Conv3×3(48→12) / Conv5×5(48→12) / Conv7×7(48→12) → 每条路径接 MambaLayer(12) → cat → Conv3×3 融合。5×5 卷积参数量为 48×12×25=14,400，7×7 卷积为 48×12×49=28,224。在 256×256 分辨率下，仅 ESI 的 7×7 分支即贡献约 1.85G FLOPs。

**改进后（IDC + HFP）**：用 IDC 替换 ESI 的多核卷积投影。IDC 将输入通道拆分为四部分：Identity 分支（不卷积）、3×3 方形 DWConv 分支、1×11 水平 DWConv 分支、11×1 垂直 DWConv 分支。DWConv 的参数量仅为 `channel × k`（而非 `in × out × k²`），大幅降低计算成本。随后每个分支仍接入 MambaLayer 进行长程依赖建模。

同时，在每级上采样后插入 HFP 模块：通过 DCT 将特征转换到频域 → 用高通掩码过滤低频分量 → 分别在通道和空间维度上对高频特征进行注意力加权 → IDCT 回到空间域 → 与原始特征残差融合。

**积极变化**：IDC 的 Identity 分支避免了对所有通道进行空间混合的冗余计算（InceptionNeXt 论文证明并非所有通道都需要空间卷积），带状卷积（1×k / k×1）将参数从 O(k²) 降为 O(k)。HFP 在解码端显式补偿上采样丢失的高频信息，与编码端的 FSDA 形成频域处理的编码-解码对称结构。

#### 代码修改建议

```python
# === 伪代码：改进后的解码器单级 ===
class ImprovedDecodeStage(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                     align_corners=True)
        # HFP 高频感知（解码端频域补偿）
        self.hfp = HFP(
            in_channels=out_ch,
            ratio=(0.25, 0.25),  # 保留75%高频
            patch=(8, 8),
            isdct=True  # 启用DCT模式
        )
        # IDC 替代原始多核卷积
        self.idc = InceptionDWConv2d(
            in_channels=out_ch,
            square_kernel_size=3,
            band_kernel_size=11,
            branch_ratio=0.25
        )
        # Mamba 长程依赖建模（保持四方向扫描）
        self.mamba = MambaLayer(out_ch)
        # SCSA 替代 sa_layer（更精细的通道-空间协同）
        self.attn = SCSA(dim=out_ch)

    def forward(self, x):
        x = self.reduce(x)
        x = self.upsample(x)
        x = x + self.hfp(x)        # 频域高频恢复
        x = x + self.idc(x)        # 高效多尺度空间提取
        x = self.mamba(x)          # 长程依赖建模
        x = self.attn(x)           # 通道-空间联合注意力
        return x
```

### 3.4 附加改进：SCSA 替代 sa_layer

原始 sa_layer 的硬编码 `groups=4` 和固定的通道混洗策略在不同通道数下表现不稳定。建议用 SCSA（模块 #54）替代，其 SMSA 子模块通过四分支多尺度深度卷积在空间维度上显式建模不同语义子特征，PCSA 子模块通过渐进式通道语义注意力缓解多语义空间建模带来的通道差异。SCSA 的分组策略基于通道数自适应调整（`assert dim % 4 == 0`），比 sa_layer 的固定分组更灵活。

### 3.5 潜在收益预估

#### 精度维度

FSDA 替换 AttenFFT 预计在 QAB/F（边缘保真度）和 FMI（特征互信息）指标上带来约 1-3% 的提升，因为频域注意力的数学修正将更准确地保留高频边缘信息。CMA 的引入预计在 SCD（相关性差异之和）上带来 2-5% 的提升，尤其在 PET-MRI 和 SPECT-MRI 这类功能-结构互补性强的任务中，效果可能更为显著——解决论文自述的"PET 图像功能信息强度偶尔不足"问题。HFP 在解码端的引入预计在 AG（平均梯度）和 SF（空间频率）上带来 1-2% 的提升，增强融合图像的纹理清晰度。

#### 效率维度

IDC 替换 ESI 的多核标准卷积预计将总 FLOPs 从 77.70G 降低至约 55-60G（降幅 20-25%），参数量从 5.77M 降低至约 4.5-5.0M。HFP 的引入会增加约 3-5G FLOPs，但 IDC 节省的计算量远超此开销，整体仍为净节省。推理延迟预计从 144.45ms 降低至约 110-120ms。

#### 泛化维度

CMA 跨模态注意力 + FSDA 频域解耦的组合为模型提供了更强的跨任务适应能力。频域处理的幅值-相位解耦对不同成像模态的频谱差异具有天然的适应性（CT 主导高频结构，PET 主导低频功能信号），而 CMA 的通道级跨模态建模可自动学习不同任务中模态间的互补关系，减少人工调整损失函数权重的需求。预计在 TNO（灰度-灰度融合）和 RoadScene（RGB-灰度融合）的跨数据集泛化测试中，MS-SSIM 提升 1-2%。

#### 不确定性说明

上述预估基于各模块原论文的报告数据和工程经验推断，实际效果取决于超参数调优、训练策略调整以及模块间的组合效应。特别是 FSDA 与 CMA 的联合使用可能存在梯度竞争，建议通过渐进式训练策略（先固定编码器训练解码器，再端到端微调）来缓解。HFP 的 DCT 变换要求输入尺寸为 2 的幂次（或至少可被 patch 大小整除），需在不同分辨率的特征图上验证数值稳定性。

---

## 总结：改进优先级排序

| 优先级 | 改进项 | 对应缺陷 | 预期收益/成本比 | 实施难度 |
|:---:|------|--------|:----------:|:----:|
| **P0** | FSDA 替换 AttenFFT | 缺陷A + 落差4 | 高（修正数学错误，提升频域建模质量） | 中 |
| **P1** | CMA 替换 MEF 的逐元素乘法 | 缺陷B | 高（直接增强跨模态互补性提取） | 中 |
| **P1** | IDC 替换 ESI 多核卷积 | 落差3 | 高（显著降低 FLOPs，无精度损失） | 低 |
| **P2** | HFP 插入解码器 | 缺陷C | 中（弥补频域不对称，增强高频保真度） | 中 |
| **P2** | EMA 替换 SpatialAttention | 落差1 | 中（丰富空间注意力信息来源） | 低 |
| **P3** | SCSA 替换 sa_layer | 落差2 | 中（自适应分组，多语义建模） | 低 |

P0 和 P1 级改进建议优先实施并进行消融实验验证；P2 和 P3 级改进可作为进一步优化手段。所有改进均遵循"即插即用"原则，不改变网络的整体编码-融合-解码架构，确保与现有训练流水线的兼容性。
