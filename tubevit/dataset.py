from typing import Callable, Optional, Tuple

from torch import Tensor
from torchvision.datasets import UCF101, ImageFolder
from torchvision.transforms import v2


class MyUCF101(UCF101):
    def __init__(self, transform: Optional[Callable] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[self.indices[video_idx]][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label


class Imagenette2Dataset(ImageFolder):
    """Wraps ImageFolder to produce (C, T, H, W) pseudo-videos by repeating a
    single image T times along the temporal axis."""

    def __init__(self, root: str, frames_per_clip: int = 32, transform: Optional[Callable] = None):
        super().__init__(root, transform=None)
        self.clip_transform = transform
        self.frames_per_clip = frames_per_clip

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        path, label = self.samples[idx]
        pil_img = self.loader(path)
        img = v2.functional.to_image(pil_img)  # (C, H, W) uint8 Tensor
        if self.clip_transform is not None:
            img = self.clip_transform(img)  # (C, H, W) float32
        video = img.unsqueeze(1).repeat(1, self.frames_per_clip, 1, 1)  # (C, T, H, W)
        return video, label
