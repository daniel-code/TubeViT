import os
import pickle

import click
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
from torchvision.transforms import v2

from tubevit.dataset import DomainTaggedDataset, Imagenette2Dataset, MyUCF101

# Maps Imagenette2 WordNet folder names → human-readable labels
_IMAGENETTE_NAMES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _unnormalize(frame: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation and clamp to [0, 1] for imshow."""
    return (frame * _IMAGENET_STD + _IMAGENET_MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


@click.command()
@click.option("-r", "--dataset-root", type=click.Path(exists=True), required=True, help="Path to UCF-101 dataset.")
@click.option("-a", "--annotation-path", type=click.Path(exists=True), required=True, help="Path to ucfTrainTestlist.")
@click.option("--label-path", type=click.Path(exists=True), required=True, help="Path to classInd.txt.")
@click.option(
    "--image-dataset-path",
    type=click.Path(exists=True),
    required=True,
    help="Root of Imagenette2 dataset (contains train/ and val/).",
)
@click.option("-b", "--batch-size", type=int, default=8, help="Number of samples to display.")
@click.option("-f", "--frames-per-clip", type=int, default=32, help="Frames per video clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(224, 224))
@click.option("--num-workers", type=int, default=0)
@click.option("--seed", type=int, default=42)
def main(
    dataset_root,
    annotation_path,
    label_path,
    image_dataset_path,
    batch_size,
    frames_per_clip,
    video_size,
    num_workers,
    seed,
):
    pl.seed_everything(seed)

    with open(label_path) as f:
        ucf_labels = list(map(lambda x: x.split(" ")[-1], f.read().splitlines()))

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    video_transform = v2.Compose(
        [
            v2.Lambda(lambda x: x.permute(0, 3, 1, 2)),  # THWC→TCHW
            v2.Resize(size=video_size, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=imagenet_mean, std=imagenet_std),
            v2.Lambda(lambda x: x.permute(1, 0, 2, 3)),  # TCHW→CTHW
        ]
    )

    image_transform = v2.Compose(
        [
            v2.Resize(size=video_size, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    metadata_file = "ucf101-train-meta.pickle"
    precomputed_metadata = None
    if os.path.exists(metadata_file):
        with open(metadata_file, "rb") as f:
            precomputed_metadata = pickle.load(f)

    video_set = MyUCF101(
        root=dataset_root,
        annotation_path=annotation_path,
        _precomputed_metadata=precomputed_metadata,
        frames_per_clip=frames_per_clip,
        train=True,
        output_format="THWC",
        transform=video_transform,
    )

    if not os.path.exists(metadata_file):
        with open(metadata_file, "wb") as f:
            pickle.dump(video_set.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    image_set = Imagenette2Dataset(
        root=os.path.join(image_dataset_path, "train"),
        frames_per_clip=frames_per_clip,
        transform=image_transform,
    )
    imagenette_labels = [_IMAGENETTE_NAMES.get(c, c) for c in image_set.classes]

    joint_set = ConcatDataset([DomainTaggedDataset(video_set, 0), DomainTaggedDataset(image_set, 1)])

    sampler = RandomSampler(joint_set, num_samples=batch_size)
    loader = DataLoader(joint_set, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    videos, labels, domains = next(iter(loader))
    # videos: (B, C, T, H, W)

    n_cols = 8  # frames to show per sample
    frame_indices = torch.linspace(0, frames_per_clip - 1, n_cols).long()

    fig, axs = plt.subplots(batch_size, n_cols, figsize=(n_cols * 1.5, batch_size * 1.5))
    if batch_size == 1:
        axs = axs[None]

    for row, (video, label_t, domain_t) in enumerate(zip(videos, labels, domains)):
        domain_val = domain_t.item()
        label_val = label_t.item()
        class_name = ucf_labels[label_val] if domain_val == 0 else imagenette_labels[label_val]
        prefix = "[V]" if domain_val == 0 else "[I]"
        axs[row][0].set_ylabel(f"{prefix} {class_name}", fontsize=6, rotation=0, labelpad=60, va="center")

        for col, t in enumerate(frame_indices):
            frame = video[:, t]  # (C, H, W)
            axs[row][col].imshow(_unnormalize(frame))
            axs[row][col].set_xticks([])
            axs[row][col].set_yticks([])

    fig.suptitle("[V] = UCF-101 video   [I] = Imagenette2 image (repeated frames)", fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
