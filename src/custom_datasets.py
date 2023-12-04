from functools import partial 
from pathlib import Path 
from typing import List, Optional, Union

from torch.utils.data import Dataset 
from torchvision import transforms as T

from constants import Tensor
from trainer import cast_num_frames, identity, gif_to_tensor

class ControlDataset(Dataset):
    def __init__(
        self,
        source_directory: Path,
        image_size: Optional[int]=64,
        channels: Optional[int]=1,
        num_frames: Optional[int]=20,
        horizontal_flip: Optional[bool]=False,
        force_num_frames: Optional[bool]=True,
        valid_extensions: Optional[List[str]]=["gif"]
    ):
        super().__init__()
        self.source_directory: Path = source_directory
        self.image_size = image_size
        self.channels = channels
        self.example_paths: List[Path] = [path for extension in valid_extensions for path in Path(self.source_directory).glob(f"**/*.{extension}")]
        self.cast_num_frames_fn: callable = partial(cast_num_frames, frames=num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.example_paths)

    def __getitem__(self, index: Union[int, Tensor]) -> Tensor:
        path: Path = self.example_paths[index]
        video: Tensor = gif_to_tensor(path, self.channels, transform=self.transform)
        return self.cast_num_frames_fn(video)
