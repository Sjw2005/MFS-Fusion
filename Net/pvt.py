import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math


class Mlp(nn.Module):
    '''
    这是 PVTv2 的 FFN，但加入了 DWConv，让 MLP 里也有局部空间感知
    IN：x: [B, N, C]，N=H*W 的 token 序列，H, W: 当前 stage 的空间尺寸
    '''

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        # x: [B, N, C]，N=H*W
        super().__init__()
        # 若未显式传出通道，保持与输入通道一致（残差分支需要同维度）
        out_features = out_features or in_features
        # 中间隐藏层默认与输入通道一致，可通过 mlp_ratio 在外部放大
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features) # [B, N, hidden]
        self.dwconv = DWConv(hidden_features) # depthwise 3x3 —> [B, N, hidden]
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear: # liner==True时在fc1和dwconv之间加入ReLU激活函数
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # Step1: token 维线性映射，逐 token 处理，不改变 token 个数 N
        x = self.fc1(x)
        if self.linear:
            # linear 模式下额外引入 ReLU，增强非线性
            x = self.relu(x)
        # Step2: 把 token 还原成特征图做深度可分离卷积，再转回 token
        x = self.dwconv(x, H, W)
        # Step3: 激活 + dropout
        x = self.act(x)
        x = self.drop(x)
        # Step4: 投影回输出通道
        x = self.fc2(x)
        x = self.drop(x)
        # OUT: [B, N, out_features]
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
    # 多头注意力要求 dim 能整除头数
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                # 空间下采样分支：降低 K/V 的 token 数量，降低注意力计算量
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            # Linear Attention 变体：先池化到固定 7x7，再 1x1 conv 生成更稳定的 K/V
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # IN: x [B, N, C], 其中 N = H * W
        B, N, C = x.shape
        # Q: [B, heads, N, C_head]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            # print("false")
            if self.sr_ratio > 1:
                # 序列 -> 特征图，做空间下采样
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                # 下采样后再展平回序列，K/V token 数由 N 变为 N'
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                # kv: [2, B, heads, N', C_head]
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                # 不下采样时，K/V 与 Q 使用同样 token 长度 N
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            # linear=True 时走固定分辨率池化分支，减少输入大小变化对注意力的影响
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k/v: [B, heads, N_or_N', C_head]
        k, v = kv[0], kv[1]

        # 注意力权重: [B, heads, N, N_or_N']
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 注意力聚合后恢复为 [B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)

        # OUT: [B, N, C]
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # Transformer Block 第1个残差：Norm -> Attention -> DropPath -> Add
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # Transformer Block 第2个残差：Norm -> MLP -> DropPath -> Add
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        # OUT: 形状保持 [B, N, C]
        return x


class OverlapPatchEmbed(nn.Module):
    """
    用带有重叠patch的步进卷积空间特征图转换为patch嵌入，通过3x3核与stride-2操作保留局部空间上下文
    投影将空间分辨率降低一半，同时增加通道维度以匹配该阶段的嵌入维度。
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        # 理论 patch 数（实际以 forward 输出的 H/W 为准）
        self.num_patches = self.H * self.W
        # 重叠卷积投影：空间降采样 + 通道升维
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # IN: x [B, Cin, H, W]
        x = self.proj(x)
        # 记录卷积后空间尺寸，后续 Block 里需要 H/W 来恢复二维结构
        _, _, H, W = x.shape
        # [B, C, H, W] -> [B, N, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        # OUT: token 序列 + 对应空间尺寸
        return x, H, W

class OverlapPatchEmbed_notproj(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # 该变体不做卷积投影，直接把输入特征图展开为 token
        # x = self.proj(x)
        # short 保留原特征，便于外部做旁路/残差使用
        short=x
        _, _, H, W = x.shape
        # [B, C, H, W] -> [B, N, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        # OUT: 原图特征 + token + 空间尺寸
        return short,x, H, W


class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=1000, embed_dims=[48, 96, 192, 384],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=True):
        super().__init__()
        # 金字塔分层配置：每个 stage 各自有 embed_dim / depth / head / sr_ratio
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        # DropPath 概率按深度线性递增
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            # 每一级先做 patch embedding，默认 stride=2 进行下采样
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=3,
                                            stride=2 if i == 0 else 2,
                                            # stride=2,
                                            # patch_size=7 if i == 0 else 3,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            # 该 stage 内部堆叠若干个 Transformer Block
            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            # 通过 setattr 动态注册为 patch_embed1/2/3..., block1/2/3...
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # IN: x [B, C, H, W]
        B = x.shape[0]
        # out 收集每个 stage 的二维特征图，供下游融合或解码器使用
        out=[]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            # patch embedding: [B,C,H,W] -> [B,N,Ci]
            short,x, H, W = patch_embed(x)
            # 当前 stage 的所有 Transformer block
            for blk in block:
                x = blk(x, H, W)
            # token 归一化
            x = norm(x)
            # [B,N,Ci] -> [B,Ci,H,W]，转回二维特征图
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            out.append(x)

        # return x.mean(dim=1)
        return out

    def forward(self, x):
        # 主干前向：返回金字塔多尺度特征列表
        x = self.forward_features(x)

        # x = self.head(x)

        return x

class PyramidVisionTransformerV2_one(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=1000, embed_dims=[48, 96, 192, 384],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=True):
        super().__init__()
        # 与 PyramidVisionTransformerV2 类似，但 patch_embed 使用 notproj 版本
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            # stride=1：该分支不做下采样（保持空间分辨率）
            patch_embed = OverlapPatchEmbed_notproj(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=3,
                                            stride=1,
                                            # patch_size=7 if i == 0 else 3,
                                            # stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            # 动态注册每个 stage 的模块
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # IN: x [B, C, H, W]
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            # patch embedding 后得到 token
            x, H, W = patch_embed(x)
            # 逐 block 更新 token
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            # if i != self.num_stages - 1:
            #     x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            # token 恢复为二维特征图，作为下一 stage 输入
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            # print(x.shape)

        # return x.mean(dim=1)
        return x

    def forward(self, x):
        # 返回最后一级二维特征图
        x = self.forward_features(x)

        # x = self.head(x)

        return x


class DWConv(nn.Module):
    '''
    在MLP中引入局部空间信息
    IN：x 形状为 [B, N, C]（序列格式），以及当前图像的高 H 和宽 W。
    '''
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        # groups=dim 表示每个通道单独卷积（depthwise conv）
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W) # 序列——>图像
        x = self.dwconv(x)  # depthwise 3x3 conv
        x = x.flatten(2).transpose(1, 2) # 图像——>序列

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    # 该工具函数常用于加载旧权重：把线性 patchify 权重 reshape 成卷积核格式
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            # [out_dim, 3*patch*patch] -> [out_dim, 3, patch, patch]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict
# print("__________________________________jian_dan_pvt______________________________________________")
# model=PyramidVisionTransformerV2(linear=True)
# a=torch.randn(1,1,128,128)
# out=model(a)
# print(out[0].shape,out[1].shape,out[2].shape,out[3].shape)

@register_model
def pvt_v2_b0(pretrained=False, **kwargs):
    # b0: 最轻量配置，通道和深度都较小
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b1(pretrained=False, **kwargs):
    # b1: 相比 b0 提升通道规模
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b2(pretrained=False, **kwargs):
    # b2: 常用配置，stage3 深度为 6
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b3(pretrained=False, **kwargs):
    # b3: 进一步加深 stage3
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b4(pretrained=False, **kwargs):
    # b4: 更深模型，计算量更大
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b5(pretrained=False, **kwargs):
    # b5: 深度最大之一，mlp_ratio 改为 4
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b2_li(pretrained=False, **kwargs):
    # b2_li: 在 b2 的基础上启用 linear=True 分支
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=True, **kwargs)
    model.default_cfg = _cfg()

    return model