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
def main(num_classes, frames_per_clip, video_size, output_path):
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
    )

    weights = ViT_B_16_Weights.DEFAULT.get_state_dict(progress=True)

    # inflated vit path convolution layer weight
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
    model.sparse_tubes_tokenizer.conv_proj_weight = torch.nn.Parameter(conv_proj_weight, requires_grad=True)

    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    main()
