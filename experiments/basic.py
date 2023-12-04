import argparse 
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader 
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer

from constants import Module, Tensor
from custom_tensorboard import SummaryWriter
from data_transforms import ControlDataset
from utils import DATA_DIRECTORY, TENSORBOARD_DIRECTORY, setup_logger, setup_experiment_directory, get_now_str

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="no_obstacles_64")

parser.add_argument("--model-dimension", type=int, default=64)
parser.add_argument("--num-timesteps", "-t", type=int, default=1000)
parser.add_argument("--sample-every", type=int, default=1000)
parser.add_argument("--num-epochs", type=int, default=10_000)

parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--channels", type=int, default=1)
parser.add_argument("--image-size", type=int, default=64)
parser.add_argument("--frames-per-video", type=int, default=20)

def main(args): 
    # configure logging 
    experiment_directory: Path = setup_experiment_directory("control_unconditional")
    log: logging.Logger = setup_logger(__name__, custom_handle=experiment_directory / "log.out")
    writer: SummaryWriter = SummaryWriter(log_dir=TENSORBOARD_DIRECTORY / get_now_str())

    # configure GPU support 
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    log.info(f"GPU support: {'YES' if torch.cuda.is_available() else 'NO'}")

    # configure the model 
    model = Unet3D(dim=args.model_dimension, channels=args.channels, dim_mults = (1, 2, 4, 8))
    model.to(device)

    # TODO add channels! 
    diffusion = GaussianDiffusion(
        model,
        channels=args.channels, 
        image_size=args.image_size,
        num_frames=args.frames_per_video,
        timesteps=args.num_timesteps,   
        loss_type='l1'
    )
    diffusion.to(device)

    trainer = Trainer(
        diffusion,
        (DATA_DIRECTORY / f"{args.data_dir}").as_posix(),                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
        train_batch_size = args.batch_size,
        train_lr = 1e-4,
        save_and_sample_every=args.sample_every,
        train_num_steps = args.num_epochs,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True                        # turn on mixed precision
    )

    trainer.train(log_fn=log.info, writer=writer)
    writer.close()

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
