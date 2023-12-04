import argparse 
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader 
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer

from constants import Module, Tensor
from data_transforms import ControlDataset
from utils import DATA_DIRECTORY, setup_logger

parser = argparse.ArgumentParser()
parser.add_argument("--model-dimension", type=int, default=32)
parser.add_argument("--num-timesteps", "-t", type=int, default=1_000)
parser.add_argument("--batch-size", type=int, default=32)

transform: Module = Compose([
    Lambda(lambda value: (value * 2) - 1),
])

reverse_transform: Module = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
])

def main(args): 
    # configure logging 
    log = setup_logger(__name__) 

    # configure GPU support 
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    log.info(f"GPU support: {'YES' if torch.cuda.is_available() else 'NO'}")


    # load the dataset 
    dataset_path: Path = DATA_DIRECTORY / "videos.pkl" 
    dataset: Dataset = ControlDataset(dataset_path, transform, device) 

    # configure the model 
    model = Unet3D(dim=args.model_dimension, dim_mults = (1, 2, 4))
    model.to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size=dataset.frame_height,
        num_frames=dataset.frames_per_video,
        timesteps=args.num_timesteps,   
        loss_type='l1'
    )
    diffusion.to(device)

    trainer = Trainer(
        diffusion,
        (DATA_DIRECTORY / "gifs").as_posix(),                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
        train_batch_size = args.batch_size,
        train_lr = 1e-4,
        save_and_sample_every = 1000,
        train_num_steps = 10_000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True                        # turn on mixed precision
    )

    trainer.train(log_fn=log.info)

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
