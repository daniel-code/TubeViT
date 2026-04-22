import sys

import click
import torch
import torchvision.transforms._functional_tensor as _tv_functional_tensor

# isort: off
# pytorchvideo 0.1.5 imports torchvision.transforms.functional_tensor, which was
# made private (_functional_tensor) in torchvision >=0.17. Alias it before any
# pytorchvideo import so the package loads.
sys.modules.setdefault("torchvision.transforms.functional_tensor", _tv_functional_tensor)

from pytorchvideo.data.encoded_video import EncodedVideo  # noqa: E402
from pytorchvideo.transforms import (  # noqa: E402
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose, Lambda  # noqa: E402
from torchvision.transforms._transforms_video import (  # noqa: E402
    CenterCropVideo,
    NormalizeVideo,
)

from tubevit.model import TubeViTLightningModule  # noqa: E402

# iosrt: on


@click.command()
@click.argument("video-path")
@click.option("-m", "--model-path", type=click.Path(exists=True), required=True, help="path to model weight.")
@click.option("--label-path", type=click.Path(exists=True), required=True, help="path to classInd.txt.")
@click.option("-f", "--frames-per-clip", type=int, default=32, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(224, 224), help="frame per clip.")
def main(
    video_path,
    model_path,
    label_path,
    frames_per_clip,
    video_size,
):
    with open(label_path, "r") as f:
        labels = f.read().splitlines()
        labels = list(map(lambda x: x.split(" ")[-1], labels))

    # Compose video data transforms
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(frames_per_clip),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ShortSideScale(size=video_size[0]),
                CenterCropVideo(crop_size=video_size),
            ]
        ),
    )

    # Load video
    video = EncodedVideo.from_path(video_path)
    # Get clip
    clip_start_sec = 0.0  # secs
    clip_duration = 2.0  # secs
    duration = video.duration
    video_data = []
    for i in range(10):
        if clip_start_sec + clip_duration * (i + 1) <= duration:
            data = video.get_clip(
                start_sec=clip_start_sec + clip_duration * i, end_sec=clip_start_sec + clip_duration * (i + 1)
            )
            data = transform(data)
            video_data.append(data["video"])

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
        )
    model.eval()
    with torch.no_grad():
        prediction = model.predict_step(batch=(video_data, None), batch_idx=0)
    print(video_data.shape)
    print("Predict:", labels[torch.argmax(torch.sum(prediction["y_prob"], dim=0)).to("cpu").item()])


if __name__ == "__main__":
    main()
