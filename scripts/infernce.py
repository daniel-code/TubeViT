import click
import torch
from torchcodec.decoders import VideoDecoder
from torchvision.transforms import v2

from tubevit.model import TubeViTLightningModule


@click.command()
@click.argument("video-path")
@click.option("-m", "--model-path", type=click.Path(exists=True), required=True, help="path to model weight.")
@click.option("--label-path", type=click.Path(exists=True), required=True, help="path to classInd.txt.")
@click.option("-f", "--frames-per-clip", type=int, default=32, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(224, 224), help="frame per clip.")
@click.option(
    "--interpolated-kernels",
    is_flag=True,
    default=False,
    show_default=True,
    help="Share one conv kernel across tubes via trilinear interpolation. "
    "Only used when loading a raw .pt file; ignored for .ckpt (hparams are restored automatically).",
)
def main(
    video_path,
    model_path,
    label_path,
    frames_per_clip,
    video_size,
    interpolated_kernels,
):
    with open(label_path, "r") as f:
        labels = f.read().splitlines()
        labels = list(map(lambda x: x.split(" ")[-1], labels))

    transform = v2.Compose(
        [
            v2.Resize(size=video_size[0], antialias=True),
            v2.CenterCrop(video_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.Lambda(lambda x: x.permute(1, 0, 2, 3)),  # TCHW→CTHW
        ]
    )

    decoder = VideoDecoder(video_path)
    fps = decoder.metadata.average_fps
    duration = decoder.metadata.duration_seconds
    clip_duration = 2.0
    video_data = []
    for i in range(10):
        if clip_duration * (i + 1) > duration:
            break
        start_frame = int(clip_duration * i * fps)
        end_frame = int(clip_duration * (i + 1) * fps)
        indices = torch.linspace(start_frame, end_frame - 1, frames_per_clip).long()
        frames = decoder.get_frames_at(indices=indices).data  # (T, C, H, W) uint8
        video_data.append(transform(frames))

    video_data = torch.stack(video_data)

    # Accept either a Lightning .ckpt (from scripts/train.py) or a raw state_dict
    # .pt (from scripts/convert_vit_weight.py). The latter is the inflated ViT-B
    # seed — its classifier head is random, so predictions on it are meaningless.
    if str(model_path).endswith(".ckpt"):
        model = TubeViTLightningModule.load_from_checkpoint(model_path)
    else:
        model = TubeViTLightningModule(
            num_classes=len(labels),
            video_shape=(3, frames_per_clip, video_size[0], video_size[1]),
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            weight_path=model_path,
            interpolated_kernels=interpolated_kernels,
        )
    model.eval()
    with torch.no_grad():
        prediction = model.predict_step(batch=(video_data, None), batch_idx=0)
    print(video_data.shape)
    print("Predict:", labels[torch.argmax(torch.sum(prediction["y_prob"], dim=0)).to("cpu").item()])


if __name__ == "__main__":
    main()
