import os
import pickle

import click
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import RandomResizedCropVideo, RandomHorizontalFlipVideo, ToTensorVideo, \
    NormalizeVideo

from TubeViT.dataset import MyUCF101
from TubeViT.model import TubeViTLightningModule
from TubeViT.video_transforms import ResizedVideo


@click.command()
@click.option('-r', '--dataset-root', type=click.Path(exists=True), required=True, help='path to dataset.')
@click.option('-a', '--annotation-path', type=click.Path(exists=True), required=True, help='path to dataset.')
@click.option('-nc', '--num-classes', type=int, default=101, help='num of classes of dataset.')
@click.option('-b', '--batch-size', type=int, default=32, help='batch size.')
@click.option('-f', '--frames-per-clip', type=int, default=32, help='frame per clip.')
@click.option('-v', '--video-size', type=click.Tuple([int, int]), default=(224, 224), help='frame per clip.')
@click.option('--max-epochs', type=int, default=10, help='max epochs.')
@click.option('--num-workers', type=int, default=0)
@click.option('--fast-dev-run', type=bool, is_flag=True, show_default=True, default=False)
@click.option('--seed', type=int, default=42, help='random seed.')
@click.option('--preview-video', type=bool, is_flag=True, show_default=True, default=False, help='Show input video')
def main(dataset_root, annotation_path, num_classes, batch_size, frames_per_clip, video_size, max_epochs, num_workers,
         fast_dev_run, seed, preview_video):
    pl.seed_everything(seed)

    train_transform = T.Compose([
        ToTensorVideo(),
        RandomHorizontalFlipVideo(),
        RandomResizedCropVideo(size=video_size),
        NormalizeVideo(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
    ])

    test_transform = T.Compose([
        ToTensorVideo(),
        ResizedVideo(size=video_size),
        NormalizeVideo(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
    ])

    train_metadata_file = 'ucf101-train-meta.pickle'
    train_precomputed_metadata = None
    if os.path.exists(train_metadata_file):
        with open(train_metadata_file, 'rb') as f:
            train_precomputed_metadata = pickle.load(f)

    train_set = MyUCF101(
        root=dataset_root,
        annotation_path=annotation_path,
        _precomputed_metadata=train_precomputed_metadata,
        frames_per_clip=frames_per_clip,
        train=True,
        output_format='THWC',
        transform=train_transform,
    )

    if not os.path.exists(train_metadata_file):
        with open(train_metadata_file, 'wb') as f:
            pickle.dump(train_set.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    val_metadata_file = 'ucf101-val-meta.pickle'
    val_precomputed_metadata = None
    if os.path.exists(val_metadata_file):
        with open(val_metadata_file, 'rb') as f:
            val_precomputed_metadata = pickle.load(f)

    val_set = MyUCF101(
        root=dataset_root,
        annotation_path=annotation_path,
        _precomputed_metadata=val_precomputed_metadata,
        frames_per_clip=frames_per_clip,
        train=False,
        output_format='THWC',
        transform=test_transform,
    )

    if not os.path.exists(val_metadata_file):
        with open(val_metadata_file, 'wb') as f:
            pickle.dump(val_set.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    train_sampler = RandomSampler(train_set, num_samples=len(train_set) // 10)
    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        sampler=train_sampler,
    )

    val_sampler = RandomSampler(val_set, num_samples=len(val_set) // 10)
    val_dataloader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        sampler=val_sampler,
    )

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
        num_layers=4,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        lr=1e-4,
        weight_path='tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt',
    )

    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval='epoch')]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        fast_dev_run=fast_dev_run,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.save_checkpoint('./models/tubevit_ucf101.ckpt')


if __name__ == '__main__':
    main()
