# test_pvt_debug.py
import torch
from Net.pvt import PyramidVisionTransformerV2

def shape_hook(name):
    def _hook(module, inp, out):
        in_shapes = [tuple(t.shape) for t in inp if isinstance(t, torch.Tensor)]
        if isinstance(out, torch.Tensor):
            out_shapes = [tuple(out.shape)]
        elif isinstance(out, (list, tuple)):
            out_shapes = [tuple(t.shape) for t in out if isinstance(t, torch.Tensor)]
        else:
            out_shapes = [type(out)]
        print(f"[{name}] in={in_shapes} out={out_shapes}")
    return _hook

def main():
    torch.manual_seed(0)

    # 对齐 MyNet 里的 backbone 配置
    model = PyramidVisionTransformerV2(
        in_chans=48,
        embed_dims=[48, 96, 192],
        num_heads=[1, 2, 4],
        mlp_ratios=[4, 4, 4],
        depths=[2, 2, 2],
        sr_ratios=[8, 4, 2],
        num_stages=3,
        linear=True
    ).eval()

    # 给关键模块挂 hook
    for i in range(model.num_stages):
        getattr(model, f"patch_embed{i+1}").register_forward_hook(shape_hook(f"patch_embed{i+1}"))
        getattr(model, f"norm{i+1}").register_forward_hook(shape_hook(f"norm{i+1}"))
        blocks = getattr(model, f"block{i+1}")
        blocks[0].register_forward_hook(shape_hook(f"block{i+1}[0]"))

    x = torch.randn(1, 48, 128, 128)

    # 不直接走 model(x)，先手动走 stage，避免 forward_features 内部返回值不一致问题
    B = x.shape[0]
    outs = []
    for i in range(model.num_stages):
        patch_embed = getattr(model, f"patch_embed{i+1}")
        block = getattr(model, f"block{i+1}")
        norm = getattr(model, f"norm{i+1}")

        x, H, W = patch_embed(x)
        print(f"Stage{i+1} tokens: {x.shape}, H={H}, W={W}")

        for j, blk in enumerate(block):
            x = blk(x, H, W)
            print(f"Stage{i+1} block{j} -> {x.shape}")

        x = norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        print(f"Stage{i+1} feat map -> {x.shape}")
        outs.append(x)

    print("Final pyramid:")
    for i, o in enumerate(outs, 1):
        print(f"  out{i}: {tuple(o.shape)}")

if __name__ == "__main__":
    main()