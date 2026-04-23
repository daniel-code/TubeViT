import os
import pickle

import click
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from tubevit.dataset import Imagenette2Dataset, MyUCF101
from tubevit.model import TubeViTLightningModule


@click.command()
@click.option("-r", "--dataset-root", type=click.Path(exists=True), required=True, help="path to dataset.")
@click.option("-a", "--annotation-path", type=click.Path(exists=True), required=True, help="path to dataset.")
@click.option("-nc", "--num-classes", type=int, default=101, help="num of classes of dataset.")
@click.option("-b", "--batch-size", type=int, default=32, help="batch size.")
@click.option("-f", "--frames-per-clip", type=int, default=32, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(224, 224), help="frame per clip.")
@click.option("--max-epochs", type=int, default=10, help="max epochs.")
@click.option("--num-workers", type=int, default=0)
@click.option("--fast-dev-run", type=bool, is_flag=True, show_default=True, default=False)
@click.option("--seed", type=int, default=42, help="random seed.")
@click.option("--preview-video", type=bool, is_flag=True, show_default=True, default=False, help="Show input video")
@click.option(
    "--lr", type=float, default=1e-4, show_default=True, help="Base LR (paper: 5e-5 for ViT-B, 1e-5 for ViT-L/H)."
)
@click.option(
    "--weight-decay",
    type=float,
    default=0.001,
    show_default=True,
    help="Adam weight decay (paper: 0.001 for B, 1e-5 for L/H).",
)
@click.option(
    "--warmup-steps", type=int, default=0, show_default=True, help="Linear warmup steps for LR schedule (paper: 10000)."
)
@click.option(
    "--interpolated-kernels",
    is_flag=True,
    default=False,
    show_default=True,
    help="Share one conv kernel across tubes via trilinear interpolation (paper ablation). "
    "Required when loading tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt.",
)
@click.option(
    "--image-dataset-path",
    type=click.Path(exists=True),
    default=None,
    help="Root of Imagenette2 dataset (e.g. data/raw/imagenette2-320). Enables joint training.",
)
@click.option("--image-num-classes", type=int, default=10, show_default=True, help="Num classes for image dataset.")
def main(
    dataset_root,
    annotation_path,
    num_classes,
    batch_size,
    frames_per_clip,
    video_size,
    max_epochs,
    num_workers,
    fast_dev_run,
    seed,
    preview_video,
    lr,
    weight_decay,
    warmup_steps,
    interpolated_kernels,
    image_dataset_path,
    image_num_classes,
):
    pl.seed_everything(seed)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = v2.Compose(
        [
            v2.Lambda(lambda x: x.permute(0, 3, 1, 2)),  # THWC→TCHW uint8
            v2.Resize(size=video_size, antialias=True),
            v2.RandAugment(num_ops=2, magnitude=10),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=imagenet_mean, std=imagenet_std),
            v2.Lambda(lambda x: x.permute(1, 0, 2, 3)),  # TCHW→CTHW
        ]
    )

    test_transform = v2.Compose(
        [
            v2.Lambda(lambda x: x.permute(0, 3, 1, 2)),  # THWC→TCHW uint8
            v2.Resize(size=video_size, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=imagenet_mean, std=imagenet_std),
            v2.Lambda(lambda x: x.permute(1, 0, 2, 3)),  # TCHW→CTHW
        ]
    )

    image_train_transform = v2.Compose(
        [
            v2.Resize(size=video_size, antialias=True),
            v2.RandAugment(num_ops=2, magnitude=10),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    image_test_transform = v2.Compose(
        [
            v2.Resize(size=video_size, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=imagenet_mean, std=imagenet_std),
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

    val_metadata_file = "ucf101-val-meta.pickle"
    val_precomputed_metadata = None
    if os.path.exists(val_metadata_file):
        with open(val_metadata_file, "rb") as f:
            val_precomputed_metadata = pickle.load(f)

    val_set = MyUCF101(
        root=dataset_root,
        annotation_path=annotation_path,
        _precomputed_metadata=val_precomputed_metadata,
        frames_per_clip=frames_per_clip,
        train=False,
        output_format="THWC",
        transform=test_transform,
    )

    if not os.path.exists(val_metadata_file):
        with open(val_metadata_file, "wb") as f:
            pickle.dump(val_set.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    image_train_dataloader = None
    image_val_dataloader = None
    if image_dataset_path is not None:
        image_train_set = Imagenette2Dataset(
            root=os.path.join(image_dataset_path, "train"),
            frames_per_clip=frames_per_clip,
            transform=image_train_transform,
        )
        image_val_set = Imagenette2Dataset(
            root=os.path.join(image_dataset_path, "val"),
            frames_per_clip=frames_per_clip,
            transform=image_test_transform,
        )
        image_train_dataloader = DataLoader(
            image_train_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        image_val_dataloader = DataLoader(
            image_val_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )
        print(f"Joint training: {len(image_train_set)} image samples, {image_num_classes} classes")

    x, y = next(iter(train_dataloader))
    print(x.shape)

    if preview_video:
        x = x.permute(0, 2, 3, 4, 1)
        fig, axs = plt.subplots(4, 8)
        for i in range(4):
            for j in range(8):
                axs[i][j].imshow(x[0][i * 8 + j])
                axs[i][j].set_xticks([])
                axs[i][j].set_yticks([])
        plt.tight_layout()
        plt.show()

    model = TubeViTLightningModule(
        num_classes=num_classes,
        video_shape=x.shape[1:],
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        weight_path="tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt",
        max_epochs=max_epochs,
        interpolated_kernels=interpolated_kernels,
        image_num_classes=image_num_classes if image_dataset_path is not None else None,
    )

    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="epoch")]
    logger = TensorBoardLogger("logs", name="TubeViT")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        fast_dev_run=fast_dev_run,
        logger=logger,
        callbacks=callbacks,
    )
    train_loaders = [train_dataloader]
    val_loaders = [val_dataloader]
    if image_train_dataloader is not None:
        train_loaders.append(image_train_dataloader)
        val_loaders.append(image_val_dataloader)

    trainer.fit(model, train_dataloaders=train_loaders, val_dataloaders=val_loaders)
    trainer.save_checkpoint("./models/tubevit_ucf101.ckpt")


if __name__ == "__main__":
    main()
