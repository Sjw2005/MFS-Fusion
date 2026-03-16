# MFS-Fusion 改进实验操作指南：P0 — FSDA 替换 AttenFFT

> **改进优先级**：P0（最高优先级）  
> **对应缺陷**：缺陷 A（频域注意力数学不严谨）+ 落差 4（Q/K/V 非多尺度）  
> **改进目标**：用幅值-相位解耦注意力（FSDA）替换 AttenFFT 中数学上不严谨的复数 Softmax  
> **预期收益**：QAB/F 和 FMI 指标提升 1–3%，频域建模质量显著改善

---

## 一、实验前准备：建立可靠的基线

在动手修改任何代码之前，必须先跑通原始模型并记录完整的基线指标。所有后续消融实验的意义都建立在基线数据的可复现性之上。

### 1.1 基线训练与测试

首先确认当前代码库能够正常运行，按照 `config.json` 的默认配置（`net_type="un"`，即 MyNet + PVTv2 骨干）完成一次完整训练。训练完成后，在所有评估数据集上运行推理，记录以下指标：

**融合质量指标（逐数据集记录）**：EN（信息熵）、MI（互信息）、FMI（特征互信息）、QAB/F（基于梯度的边缘保真度）、SCD（相关性差异之和）、AG（平均梯度）、SF（空间频率）、SD（标准差）、VIF（视觉信息保真度）、MS-SSIM（多尺度结构相似度）。

**效率指标**：参数量（Params，单位 M）、浮点运算量（FLOPs，单位 G，输入尺寸 256×256）、单张推理延迟（ms，GPU 型号需固定记录）。

**测试数据集**：至少覆盖 MSRS（VIS-IR）、TNO（VIS-IR）、RoadScene（VIS-IR）三个数据集。如果条件允许，同时记录 MEF（多曝光）和医学图像融合（PET-MRI / SPECT-MRI / CT-MRI）的基线数据。

### 1.2 定位待修改模块的运行时行为

在基线模型上添加临时的调试钩子（hook），打印 AttenFFT 模块在 forward 过程中的中间张量统计信息。这一步的目的是理解当前频域注意力的实际数值行为，为后续改进提供对照参考。

具体做法是在 `Net/MyNet.py` 的 `AttenFFT.forward` 方法中，在 FFT 变换后、Softmax 前后分别打印张量的均值、标准差、最大值和最小值。特别关注复数 Softmax 后实部和虚部的分布是否出现数值异常（如接近零或爆炸）。

建议的调试代码插入位置如下（仅用于诊断，后续删除）：

```python
# 在 AttenFFT.forward 中，FFT2 变换后添加：
print(f"[DEBUG] FFT output - real mean: {x_fft.real.mean():.4f}, imag mean: {x_fft.imag.mean():.4f}")
print(f"[DEBUG] FFT output - real std: {x_fft.real.std():.4f}, imag std: {x_fft.imag.std():.4f}")

# 在复数 Softmax 后添加：
print(f"[DEBUG] After complex softmax - real mean: {attn.real.mean():.4f}, imag mean: {attn.imag.mean():.4f}")
```

运行几个 batch 的推理，将打印结果保存为文本文件，作为改进前的频域数值行为基准。

---

## 二、代码改动总览：需要修改的文件与位置

本次改进涉及的代码修改集中在以下文件和类中。改动范围经过严格限定，不触及网络的整体编码-融合-解码架构。

### 2.1 需要修改的文件清单

| 文件路径 | 修改类型 | 涉及的类/函数 | 改动说明 |
|---------|---------|-------------|---------|
| `Net/MyNet.py` | **重写** | `class AttenFFT` | 用 FSDA 逻辑替换核心的频域注意力计算 |
| `Net/MyNet.py` | **适配** | `class ASI` | 确认 ASI 调用 AttenFFT 的接口不变 |
| `Net/MyNet.py` | **新增** | `class FSDA`（新类） | 新增幅值-相位解耦注意力模块 |
| `Net/MyNet.py` | **可选新增** | `class ECALayer`（新类） | 轻量通道注意力，替代 FSDA 内部的 SE 双全连接层 |

### 2.2 不需要修改的文件

`Net/pvt.py`、`Net/MobileViT.py`、`Net/interact.py`、`models/loss_vif.py`、`train.py`、`models/model_plain.py`、`config.json` 均不需要修改。损失函数和训练流水线完全复用原始配置。

---

## 三、分步改动指南

### 3.1 第一步：理解 AttenFFT 的当前实现

打开 `Net/MyNet.py`，找到 `class AttenFFT`。当前实现的核心流程如下：

```
输入 x: [B, dim, H, W]
  → DWConv3×3 生成 q_s, k_s, v_s（三个独立的 depthwise conv，均为 3×3）
  → 对 q_s, k_s, v_s 分别做 torch.fft.fft2 得到复数张量
  → 多头拆分后计算 Q×K^T（复数域点积）
  → 对复数注意力矩阵的实部和虚部分别做 Softmax（问题所在）
  → 加权求和后 IFFT2 → 取绝对值回到实数域
  → 同时有一个 weight 分支：FFT(x) → Conv1×1→BN→ReLU→Conv1×1→Sigmoid 生成频域权重
  → 两路输出 cat → Conv1×1(2*dim → dim) 得到最终输出
```

需要重点关注的问题点：`custom_complex_normalization` 函数对复数的实部/虚部分别做 Softmax，这在数学上是不严谨的。

### 3.2 第二步：编写 FSDA 模块

在 `Net/MyNet.py` 中新增 `class FSDA` 模块。FSDA 的核心思想是将 FFT 输出的复数频谱显式分解为幅度谱和相位谱，分别通过通道注意力网络进行增强，避免在复数域上强行应用 Softmax。

以下是完整的 FSDA 实现代码，请在 `Net/MyNet.py` 文件中、`class AttenFFT` 定义之前插入：

```python
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
            nn.ReLU(inplace=True),
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
        x_fft = torch.fft.fft2(x, norm='ortho')  # [B, C, H, W] 复数

        # Step 2: 显式分解幅度谱和相位谱
        mag = torch.abs(x_fft)           # [B, C, H, W] 实数，幅度
        pha = torch.angle(x_fft)         # [B, C, H, W] 实数，相位（弧度）

        # Step 3: 幅度谱通道注意力增强（在实数域操作，数学严谨）
        mag_feat = self.mag_enhance(mag)         # Conv1×1 + BN + ReLU
        mag_feat = self.mag_attn(mag_feat)       # ECA 通道注意力
        mag_out = self.mag_proj(mag_feat) + mag  # 残差连接：增强 + 原始

        # Step 4: 相位谱通道注意力增强（在实数域操作，数学严谨）
        pha_feat = self.pha_enhance(pha)
        pha_feat = self.pha_attn(pha_feat)
        pha_out = self.pha_proj(pha_feat) + pha  # 残差连接

        # Step 5: 极坐标重建回复数频谱
        # F_out = mag_out * exp(j * pha_out)
        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        x_fft_enhanced = torch.complex(real, imag)

        # Step 6: IFFT2 回到空间域
        out = torch.fft.ifft2(x_fft_enhanced, norm='ortho').real  # 取实部

        return out
```

### 3.3 第三步：用 FSDA 替换 ASI 中的 AttenFFT 调用

在 `class ASI` 中，原始代码创建了四个 AttenFFT 实例（`self.fft1` 到 `self.fft4`）。需要将它们全部替换为 FSDA。

找到 `class ASI` 的 `__init__` 方法，将以下四行：

```python
self.fft1 = AttenFFT(self.dim)
self.fft2 = AttenFFT(self.dim)
self.fft3 = AttenFFT(self.dim)
self.fft4 = AttenFFT(self.dim)
```

替换为：

```python
self.fft1 = FSDA(self.dim)
self.fft2 = FSDA(self.dim)
self.fft3 = FSDA(self.dim)
self.fft4 = FSDA(self.dim)
```

**关键验证点**：检查 ASI 的 forward 方法中调用 `self.fft1(x1)` 的方式。FSDA 的 forward 接口与 AttenFFT 完全相同——输入 `[B, dim//4, H, W]`，输出 `[B, dim//4, H, W]`——因此 ASI 的 forward 方法无需任何修改。

### 3.4 第四步：处理原始 AttenFFT 类

有两种策略可以选择。

**策略 A（推荐用于消融实验阶段）**：保留原始 `class AttenFFT`，仅修改 ASI 中的实例化代码。这样可以随时在 FSDA 和 AttenFFT 之间切换，方便消融实验。

**策略 B（最终提交时）**：在所有实验完成且确认 FSDA 优于 AttenFFT 后，删除 `class AttenFFT` 及其内部的 `custom_complex_normalization` 函数，清理代码。

实验阶段建议采用策略 A，在 ASI 的 `__init__` 中通过一个布尔参数控制使用哪种频域注意力：

```python
class ASI(nn.Module):
    def __init__(self, dim, use_fsda=True):
        super(ASI, self).__init__()
        self.dim = dim // 4
        # ... 膨胀卷积保持不变 ...
        FreqModule = FSDA if use_fsda else AttenFFT
        self.fft1 = FreqModule(self.dim)
        self.fft2 = FreqModule(self.dim)
        self.fft3 = FreqModule(self.dim)
        self.fft4 = FreqModule(self.dim)
        # ... 其余不变 ...
```

---

## 四、修改后的完整性验证

代码修改完成后，在进入正式训练之前，必须通过以下三项验证。

### 4.1 形状一致性验证

编写一个简单的测试脚本，验证改进后的网络输入输出形状不变：

```python
import torch
from Net.MyNet import MyNet

model = MyNet(in_chans=1, hidden_chans=[48, 96, 192], out_chans=1)
x = torch.randn(2, 1, 256, 256)
y = torch.randn(2, 1, 256, 256)
out = model(x, y)
assert out.shape == (2, 1, 256, 256), f"输出形状错误: {out.shape}"
print("形状验证通过")
```

### 4.2 梯度流验证

确认 FSDA 模块的所有参数都能接收到梯度：

```python
model = MyNet(in_chans=1, hidden_chans=[48, 96, 192], out_chans=1)
x = torch.randn(2, 1, 256, 256, requires_grad=False)
y = torch.randn(2, 1, 256, 256, requires_grad=False)
out = model(x, y)
loss = out.mean()
loss.backward()

for name, param in model.named_parameters():
    if 'fft' in name or 'fsda' in name.lower():
        assert param.grad is not None, f"参数 {name} 梯度为 None"
        assert not torch.isnan(param.grad).any(), f"参数 {name} 梯度含 NaN"
        print(f"[OK] {name}: grad norm = {param.grad.norm():.6f}")

print("梯度流验证通过")
```

### 4.3 参数量与 FLOPs 对比

使用 `thop` 或 `fvcore` 计算改进前后的参数量和 FLOPs 变化：

```python
from thop import profile

model_old = MyNet(in_chans=1, hidden_chans=[48, 96, 192], out_chans=1)
# 将 ASI 中的 use_fsda 设为 False 来构建原始版本

model_new = MyNet(in_chans=1, hidden_chans=[48, 96, 192], out_chans=1)
# use_fsda=True（默认）

dummy_x = torch.randn(1, 1, 256, 256)
dummy_y = torch.randn(1, 1, 256, 256)

flops_old, params_old = profile(model_old, inputs=(dummy_x, dummy_y))
flops_new, params_new = profile(model_new, inputs=(dummy_x, dummy_y))

print(f"原始模型: {params_old/1e6:.2f}M params, {flops_old/1e9:.2f}G FLOPs")
print(f"改进模型: {params_new/1e6:.2f}M params, {flops_new/1e9:.2f}G FLOPs")
print(f"参数量变化: {(params_new - params_old)/1e6:+.2f}M")
print(f"FLOPs 变化: {(flops_new - flops_old)/1e9:+.2f}G")
```

FSDA 使用 Conv1×1 + ECA 替代原始 AttenFFT 中的 DWConv + 复数 Softmax + Cat + Conv1×1(2dim→dim)，预期参数量变化不大（可能略有增加或持平），FLOPs 应基本持平或略降。

---

## 五、消融实验设计

本次改进的消融实验分为三组，按照由粗到细的粒度逐步验证每个设计决策的贡献。

### 5.1 实验组 A：主效果验证（必做）

| 实验编号 | 配置 | 说明 |
|---------|------|------|
| A0 | 原始 AttenFFT（基线） | 复数 Softmax 频域注意力 |
| A1 | FSDA（幅值-相位解耦 + ECA） | 完整的改进方案 |
| A2 | FSDA（幅值-相位解耦 + SE） | 用 SE 替代 ECA，对比通道注意力方案 |

三组实验使用完全相同的训练配置（学习率、batch size、epoch 数、随机种子）。每组实验在 MSRS、TNO、RoadScene 三个数据集上报告全部 10 项融合质量指标。

**核心比较**：A1 vs A0 验证 FSDA 替换 AttenFFT 的整体收益；A1 vs A2 验证 ECA 相对于 SE 的轻量化优势。

### 5.2 实验组 B：频域处理方式消融（推荐做）

| 实验编号 | 配置 | 说明 |
|---------|------|------|
| B1 | 仅增强幅度谱，相位谱直接透传 | 验证幅度谱增强的独立贡献 |
| B2 | 仅增强相位谱，幅度谱直接透传 | 验证相位谱增强的独立贡献 |
| B3 | 幅度谱 + 相位谱同时增强（即完整 FSDA） | 验证双分支协同效应 |

实现方式：在 FSDA 的 forward 中通过条件判断控制。例如 B1 实验中，`pha_out = pha`（跳过相位增强分支）。

### 5.3 实验组 C：残差连接消融（可选做）

| 实验编号 | 配置 | 说明 |
|---------|------|------|
| C1 | 有残差连接（`mag_out = enhanced + mag`） | 默认方案 |
| C2 | 无残差连接（`mag_out = enhanced`） | 验证残差连接对训练稳定性的贡献 |

观察 C2 是否出现训练不稳定（loss 震荡或 NaN），以此证明残差连接在频域增强中的必要性。

---

## 六、训练策略与超参数

### 6.1 训练配置

**与基线完全一致的参数**：这些参数在改进实验中不做任何调整，以保证对比公平性。

学习率 5e-5、batch size 2、训练图像尺寸 256×256、EMA 衰减 0.999、损失函数 `fusion_loss_vif`（权重 λ₁=10, λ₂=20, λ₃=5, λ₄=1）。优化器和学习率调度策略与原始论文保持一致。

**训练轮数**：与基线相同（300 epochs），不提前截断。

**随机种子**：固定为 3 个不同的种子（如 42, 123, 2024），每个实验配置跑 3 次取均值和标准差，避免单次实验的偶然性。

### 6.2 可能需要微调的参数

FSDA 本身不引入额外的需要调参的超参数——ECA 的卷积核大小由通道数自适应计算，BN 和 ReLU 无超参数。但需要关注以下两点：

第一，FFT 的归一化模式。代码中使用 `norm='ortho'`（正交归一化），这与原始 AttenFFT 中是否一致需要核查。如果原始代码使用的是默认归一化（无 `norm` 参数），则 FSDA 中也应保持一致，否则幅度谱的数值范围会不同，影响 BN 层的统计量。

第二，幅度谱和相位谱的数值范围。幅度谱为非负实数，相位谱为 [-π, π]。FSDA 中的 BN + ReLU 对幅度谱是自然适配的（非负值经 ReLU 后不变），但对相位谱可能存在问题——ReLU 会将负相位截断为零。建议的替代方案是将相位分支的激活函数从 ReLU 改为 Tanh（输出范围 [-1, 1]）或直接去掉激活函数。这可以作为一个额外的消融变量。

---

## 七、评估与结果分析

### 7.1 定量指标记录模板

每个实验配置需要填写以下表格（每个数据集一张）：

| 指标 | 基线 A0 | FSDA+ECA A1 | FSDA+SE A2 | Δ(A1-A0) | Δ(A2-A0) |
|------|---------|------------|------------|----------|----------|
| EN | | | | | |
| MI | | | | | |
| FMI | | | | | |
| QAB/F | | | | | |
| SCD | | | | | |
| AG | | | | | |
| SF | | | | | |
| SD | | | | | |
| VIF | | | | | |
| MS-SSIM | | | | | |
| Params (M) | 5.70 | 5.69 | | | |
| FLOPs (G) | 28.42 | 28.36 | | | |
| 推理延迟 (ms) | | | | | |

### 7.2 定性可视化对比

除了定量指标外，还需要生成可视化对比图。选择 3-5 张典型场景的融合结果进行并排展示，重点关注以下方面：

边缘清晰度——观察融合图像中建筑边缘、人物轮廓等高频结构是否比基线更加锐利。这直接反映 FSDA 对高频分量的保持效果。

纹理保真度——观察地面纹理、植被纹理等中频信息是否得到更好的保留。频域解耦注意力应该能更精确地控制各频率分量的能量分配。

伪影抑制——观察融合图像中是否存在振铃效应或棋盘格伪影。如果 FSDA 的频域处理不当，可能在 IFFT 后引入新的伪影，这是需要警惕的失败模式。

### 7.3 频域中间特征可视化

为了直观理解 FSDA 的工作机制，建议在推理时保存并可视化以下中间结果：

原始幅度谱 `mag`、增强后幅度谱 `mag_out`、两者的差值 `mag_out - mag`（即 FSDA 学到的幅度修正量）。将差值图按通道取均值后显示为热力图，可以直观看到 FSDA 在哪些频率区域进行了增强或抑制。

同样地，保存相位谱的修正量 `pha_out - pha`，观察相位修正是否集中在边缘区域对应的频率分量上。

---

## 八、常见问题与排查清单

### 8.1 训练中出现 NaN

如果 FSDA 替换后训练出现 NaN，按以下顺序排查：

第一，检查 `torch.angle()` 的输入。当 FFT 输出的复数幅度接近零时，`torch.angle()` 可能产生不稳定的梯度。解决方案是在计算 `mag` 时添加一个小的 epsilon：`mag = torch.abs(x_fft) + 1e-8`。

第二，检查极坐标重建步骤。`torch.cos` 和 `torch.sin` 本身不会产生 NaN，但如果 `pha_out` 的值域被 BN 层拉伸到很大的范围，重建后的复数幅值可能爆炸。解决方案是将相位分支的 BN 替换为 LayerNorm，或在 `pha_out` 上添加 `torch.clamp(-π, π)`。

第三，检查梯度范数。在训练循环中添加梯度裁剪 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`，观察是否解决问题。

### 8.2 指标不升反降

如果 FSDA 替换后指标下降，可能的原因和排查方向如下：

第一，FFT 归一化模式不匹配。确认 FSDA 中的 `norm='ortho'` 与原始 AttenFFT 的设置一致。不同归一化模式下幅度谱的数值范围差异很大。

第二，相位分支激活函数不当。如前所述，ReLU 可能截断负相位。尝试将相位分支的 ReLU 替换为不截断负值的激活函数（如 GELU 或 Tanh），或直接移除激活函数。

第三，ECA 核大小不合适。当通道数为 12（ASI 中 dim=48 时 dim//4=12）时，ECA 自适应计算的核大小可能过小。手动检查 `k = int(abs((math.log2(12) + 1) / 2))` 的值，确认其不小于 3。

### 8.3 训练速度明显变慢

FSDA 引入了两次 FFT2/IFFT2 操作以及两个 ECA 分支，计算量略高于原始 AttenFFT。如果训练速度下降超过 15%，可以考虑以下优化：

将 FSDA 的 FFT2 和 IFFT2 设置为不跟踪梯度的中间步骤（不推荐，会阻断频域参数的梯度流），或者使用 `torch.cuda.amp` 混合精度训练加速 FFT 运算。注意 PyTorch 的 FFT 在 float16 下可能有数值精度问题，建议 FFT/IFFT 操作保持 float32。

---

## 九、实验时间线建议

| 阶段 | 预计耗时 | 产出 |
|------|---------|------|
| 基线复现与指标记录 | 2-3 天 | 基线指标表、调试日志 |
| FSDA 模块编码与形状/梯度验证 | 1 天 | 通过验证的新代码 |
| 实验组 A（主效果，3 配置 × 3 种子） | 5-7 天 | 9 组训练结果 |
| 实验组 B（频域消融，3 配置 × 1 种子） | 3-4 天 | 3 组训练结果 |
| 实验组 C（残差消融，可选） | 1-2 天 | 2 组训练结果 |
| 结果整理、可视化、分析 | 2 天 | 完整实验报告 |
| **合计** | **约 14-19 天** | |

建议在实验组 A 完成后即进行初步分析。如果 A1 相对 A0 的提升不显著（各指标提升均小于 0.5%），应优先排查第八节中列出的常见问题，而非直接推进实验组 B 和 C。

---

## 十、后续衔接：为 P1 改进做准备

当 FSDA 的消融实验完成并确认有效后，下一步将进入 P1 优先级的改进——CMA 替换 MEF 的逐元素乘法融合、IDC 替换 ESI 的多核卷积。

在进行 P1 改进时，建议以"FSDA 已集成"的模型作为新的基线（而非回退到原始模型），这样可以逐步累积改进收益。但消融实验中仍需包含"仅 FSDA"和"FSDA + CMA"的对比组，以隔离各改进项的独立贡献。

如果 FSDA 的改进效果不显著甚至为负，应将原始 AttenFFT 恢复为默认配置，独立推进 P1 改进，并在后续的全组合实验中重新评估 FSDA 与其他改进项的协同效应。
