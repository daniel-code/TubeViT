import os
import pickle

import click
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorchvideo.transforms import Normalize, Permute, RandAugment
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo

from tubevit.dataset import MyUCF101


@click.command()
@click.option("-r", "--dataset-root", type=click.Path(exists=True), required=True, help="path to dataset.")
@click.option("-a", "--annotation-path", type=click.Path(exists=True), required=True, help="path to dataset.")
@click.option("--label-path", type=click.Path(exists=True), required=True, help="path to classInd.txt.")
@click.option("-b", "--batch-size", type=int, default=32, help="batch size.")
@click.option("-f", "--frames-per-clip", type=int, default=32, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(224, 224), help="frame per clip.")
@click.option("--num-workers", type=int, default=0)
@click.option("--seed", type=int, default=42, help="random seed.")
def main(dataset_root, video_size, annotation_path, label_path, frames_per_clip, batch_size, num_workers, seed):
    pl.seed_everything(seed)
    with open(label_path, "r") as f:
        labels = f.read().splitlines()
        labels = list(map(lambda x: x.split(" ")[-1], labels))

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = T.Compose(
        [
            ToTensorVideo(),  # C, T, H, W
            Permute(dims=[1, 0, 2, 3]),  # T, C, H, W
            RandAugment(magnitude=10, num_layers=2),
            Permute(dims=[1, 0, 2, 3]),  # C, T, H, W
            T.Resize(size=video_size),
            Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    train_metadata_file = "ucf101-train-meta.pickle"
    train_precomputed_metadata = None
    if os.path.exists(train_metadata_file):
        with open(train_metadata_file, "rb") as f:
            train_precomputed_metadata = pickle.load(f)

    train_set = MyUCF101(
        root=dataset_root,
        annotation_path=annotation_path,
        _precomputed_metadata=train_precomputed_metadata,
        frames_per_clip=frames_per_clip,
        train=True,
        output_format="THWC",
        transform=train_transform,
    )

    if not os.path.exists(train_metadata_file):
        with open(train_metadata_file, "wb") as f:
            pickle.dump(train_set.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    train_sampler = RandomSampler(train_set, num_samples=len(train_set) // 10)
    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        sampler=train_sampler,
    )

    x, y = next(iter(train_dataloader))

    x = x.permute(0, 2, 3, 4, 1)  # CTHW->THWC

    fig, axs = plt.subplots(batch_size // 4, 8, figsize=(batch_size // 4, 8))
    for i in range(batch_size // 4):
        axs[i][0].set_title(labels[y[i]])
        for j in range(8):
            axs[i][j].imshow(x[i][j])
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
