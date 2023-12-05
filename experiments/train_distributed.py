import argparse 
import copy 
from functools import partial 
from pathlib import Path

from einops import rearrange
from PIL import Image
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms as T, utils
from video_diffusion_pytorch import Unet3D, GaussianDiffusion

from constants import Module, Tensor
from custom_datasets import ControlDataset
from custom_tensorboard import SummaryWriter
from trainer import TrainerConfig, Trainer
from train_utils import EMA, CHANNELS_TO_MODE, cycle, seek_all_images, gif_to_tensor, video_tensor_to_gif, num_to_groups, noop, cast_num_frames, identity, exists
from utils import DATA_DIRECTORY, TENSORBOARD_DIRECTORY, setup_logger, setup_experiment_directory, get_now_str

parser = argparse.ArgumentParser()
parser.add_argument_group(title="Architecture")
parser.add_argument("--model-dimension", type=int, default=64)
parser.add_argument("--num-timesteps", "-t", type=int, default=1000)

parser.add_argument_group(title="Diagnostics")
parser.add_argument("--sample-every", type=int, default=1000)
parser.add_argument("--num-to-sample", type=int, default=2)

parser.add_argument_group(title="Training")
parser.add_argument("--num-steps", type=int, default=10_000, help="Number of steps to train the model.")
parser.add_argument("--no-amp", action="store_true")
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--step-size", type=float, default=1e-04)
parser.add_argument("--ema-decay", type=float, default=0.995)
parser.add_argument("--update-ema-every", type=int, default=10)
parser.add_argument("--step-start-ema", type=int, default=2_000)
parser.add_argument("--gradient-accumulate-every", type=int, default=2)
parser.add_argument("--focus-present", type=float, default=0.)

parser.add_argument_group(title="Data")
parser.add_argument("--data-dir", type=str, default="no_obstacles_64")
parser.add_argument("--num-channels", type=int, default=1)
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
    log.info(f"GPUs Available: {torch.cuda.device_count()}")

    # configure the model 
    unet = Unet3D(dim=args.model_dimension, channels=args.num_channels, dim_mults = (1, 2, 4, 8))
    unet.to(device)

    # TODO add channels! 
    diffusion = GaussianDiffusion(
        unet,
        channels=args.num_channels, 
        image_size=args.image_size,
        num_frames=args.frames_per_video,
        timesteps=args.num_timesteps,   
        loss_type='l1'
    )
    diffusion.to(device)

    model = diffusion

    # configure exponential moving-average 
    ema = EMA(args.ema_decay)
    ema_model = copy.deepcopy(model)
    ema_model.load_state_dict(model.state_dict())

    # configure dataset 
    data_directory = DATA_DIRECTORY / f"{args.data_dir}"
    dataset: Dataset = ControlDataset(data_directory, args.image_size, channels=args.num_channels, num_frames=args.frames_per_video)
    num_videos: int = len(dataset)
    log.info(f"Found {num_videos=} videos at {data_directory=}")
    assert num_videos >= 1 
    dataloader: DataLoader = cycle(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True))

    # configure optimizer
    optimizer = Adam(model.parameters(), lr=args.step_size)
    step: int = 0 

    # gradient scaling 
    gradient_scaler: GradScaler = GradScaler(enabled=(not args.no_amp))
    max_gradient_norm: float = None

    # configure the trainer 
    trainer_config: TrainerConfig = TrainerConfig(
            model=model, 
            dataloader=dataloader, 
            optimizer=optimizer, 
            ema=ema, 
            ema_model=ema_model, 
            gradient_scaler=gradient_scaler,
            device_id=torch.cuda.current_device(), 
            checkpoint_every=args.sample_every, 
            log=log, 
            experiment_directory=experiment_directory, 
            num_to_sample=args.num_to_sample, 
            writer=writer, 
            gradient_accumulate_every=args.gradient_accumulate_every, 
            max_gradient_norm=max_gradient_norm, 
            present_focus_probability=args.focus_present
            )
    trainer: Trainer = Trainer(trainer_config)
    trainer.train(args.num_steps)
    writer.close()

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
