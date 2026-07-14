import random
import warnings
from collections import OrderedDict
from typing import Callable, Optional, Tuple

from torch import Tensor
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder
from torchvision.datasets import UCF101, ImageFolder
from torchvision.transforms import v2


class MyUCF101(UCF101):
    def __init__(self, transform: Optional[Callable] = None, decoder_cache_size: int = 4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = transform
        self._decoder_cache_size = decoder_cache_size
        self._decoder_cache: "OrderedDict[str, VideoDecoder]" = OrderedDict()

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        video, video_idx = self._get_clip_with_fallback(idx)
        label = self.samples[self.indices[video_idx]][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label

    def _get_video_clip(self, idx: int) -> Tuple[Tensor, int]:
        """Equivalent to `self.video_clips.get_clip(idx)`, but decodes only
        video. `VideoClips.get_clip` unconditionally also builds a torchcodec
        `AudioDecoder` and decodes audio samples for every single clip, even
        though this dataset always discards them. Besides being pure waste,
        each extra decoder held concurrently (e.g. while a DataLoader batch
        is being assembled) compounds a native memory leak in torchcodec
        that never gets released, so dropping the audio decode roughly
        quarters the leak rate.

        The remaining leak scales with how many `VideoDecoder`s get
        constructed/destroyed (confirmed empirically: RSS grows per item
        even with audio dropped), so `self._decoder_cache` keeps the last
        `decoder_cache_size` decoders alive keyed by video path and reuses
        them across clips instead of opening a fresh one every call."""
        video_clips = self.video_clips
        video_idx, clip_idx = video_clips.get_clip_location(idx)
        video_path = video_clips.video_paths[video_idx]
        clip_pts = video_clips.clips[video_idx][clip_idx]
        start_idx = int(clip_pts[0].item())
        end_idx = int(clip_pts[-1].item())

        decoder = self._decoder_cache.pop(video_path, None)
        if decoder is None:
            dimension_order = "NHWC" if video_clips.output_format == "THWC" else "NCHW"
            decoder = VideoDecoder(video_path, dimension_order=dimension_order)
        self._decoder_cache[video_path] = decoder  # re-insert as most-recently-used
        while len(self._decoder_cache) > self._decoder_cache_size:
            self._decoder_cache.popitem(last=False)

        try:
            video = decoder.get_frames_at(indices=list(range(start_idx, end_idx + 1))).data
        except RuntimeError:
            # Don't keep a decoder that just raised around for reuse.
            self._decoder_cache.pop(video_path, None)
            raise
        return video, video_idx

    def _get_clip_with_fallback(self, idx: int, max_attempts: int = 10) -> Tuple[Tensor, int]:
        """A number of UCF101 .avi files report more frames in their container
        metadata (`torchcodec`'s `decoder.metadata.num_frames`) than are
        actually decodable (corrupt/truncated trailing frames), which makes
        clip retrieval raise a RuntimeError for clips sampled near the end of
        those videos. Some videos have several consecutive unreadable
        windows, so retries jump to a random clip elsewhere in the dataset
        (rather than idx+1) to avoid getting stuck in the same bad video."""
        for attempt in range(max_attempts):
            try:
                return self._get_video_clip(idx)
            except RuntimeError as e:
                warnings.warn(f"Skipping unreadable clip {idx} ({e}); trying another clip.")
                idx = random.randrange(len(self.video_clips))
        raise RuntimeError(f"Failed to decode a valid clip after {max_attempts} attempts.")


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


class DomainTaggedDataset(Dataset):
    """Wraps any (video, label) dataset and appends an integer domain tag.

    Returns (video, label, domain) so joint-training batches can be split
    by domain inside training_step without a separate DataLoader per source.
    """

    def __init__(self, dataset: Dataset, domain: int):
        self.dataset = dataset
        self.domain = domain

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, int]:
        video, label = self.dataset[idx]
        return video, label, self.domain
