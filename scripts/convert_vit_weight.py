import click
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.models import ViT_B_16_Weights

from tubevit.model import TubeViT


@click.command()
@click.option("-nc", "--num-classes", type=int, default=101, help="num of classes of dataset.")
@click.option("-f", "--frames-per-clip", type=int, default=32, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(224, 224), help="frame per clip.")
@click.option(
    "-o",
    "--output-path",
    type=click.Path(),
    default="tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt",
    help="output model weight name.",
)
@click.option(
    "--interpolated-kernels",
    is_flag=True,
    default=False,
    show_default=True,
    help="Produce weights for the interpolated-kernel variant (single shared kernel). "
    "Default produces independent per-tube weights (paper main results).",
)
def main(num_classes, frames_per_clip, video_size, output_path, interpolated_kernels):
    x = np.random.random((1, 3, frames_per_clip, video_size[0], video_size[1]))
    x = Tensor(x)
    print("x: ", x.shape)

    y = np.random.randint(0, 1, size=(1, num_classes))
    y = Tensor(y)
    print("y: ", y.shape)

    model = TubeViT(
        num_classes=num_classes,
        video_shape=x.shape[1:],
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        interpolated_kernels=interpolated_kernels,
    )

    weights = ViT_B_16_Weights.DEFAULT.get_state_dict(progress=True)

    # inflate ViT-B/16 2D patch-embed weight → (768, 3, 8, 8, 8)
    conv_proj_weight = weights["conv_proj.weight"]
    conv_proj_weight = F.interpolate(conv_proj_weight, (8, 8), mode="bilinear")
    conv_proj_weight = torch.unsqueeze(conv_proj_weight, dim=2)
    conv_proj_weight = conv_proj_weight.repeat(1, 1, 8, 1, 1)
    conv_proj_weight = conv_proj_weight / 8.0

    # remove missmatch parameters
    weights.pop("encoder.pos_embedding")
    weights.pop("heads.head.weight")
    weights.pop("heads.head.bias")

    model.load_state_dict(weights, strict=False)

    tokenizer = model.sparse_tubes_tokenizer
    if interpolated_kernels:
        tokenizer.conv_proj_weight = torch.nn.Parameter(conv_proj_weight)
    else:
        # initialize each per-tube weight by interpolating the inflated base kernel to that tube's size
        for i, k in enumerate(tokenizer.kernel_sizes):
            w = conv_proj_weight if i == 0 else F.interpolate(conv_proj_weight, k, mode="trilinear")
            tokenizer.conv_proj_weights[i] = torch.nn.Parameter(w.clone())

    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    main()
