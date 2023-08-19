from typing import Callable, Optional, Tuple

from torch import Tensor
from torchvision.datasets import UCF101


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
