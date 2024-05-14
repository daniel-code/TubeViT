import click
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo

from tubevit.model import TubeViTLightningModule


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
                ShortSideScale(
                    size=video_size[0]
                ),
                CenterCropVideo(crop_size=video_size)
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
            data = video.get_clip(start_sec=clip_start_sec + clip_duration * i,
                                  end_sec=clip_start_sec + clip_duration * (i + 1))
            data = transform(data)
            video_data.append(data['video'])

    video_data = torch.stack(video_data)
    model = TubeViTLightningModule.load_from_checkpoint(model_path)
    prediction = model.predict_step(batch=(video_data, None), batch_idx=0)
    print(video_data.shape)
    print('Predict:', labels[torch.argmax(torch.sum(prediction['y_prob'], dim=0)).to('cpu').item()])


if __name__ == "__main__":
    main()
