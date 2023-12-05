import argparse 
import copy 
from functools import partial 
import logging
import os 
from pathlib import Path
from typing import Optional

from einops import rearrange
from PIL import Image
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms as T, utils
from video_diffusion_pytorch import Unet3D, GaussianDiffusion
import torch.multiprocessing as mp 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group, destroy_process_group

from constants import Module, Tensor
from custom_datasets import ControlDataset
from custom_tensorboard import SummaryWriter
from trainer import TrainerConfig, Trainer
from train_utils import EMA, CHANNELS_TO_MODE, cycle, seek_all_images, gif_to_tensor, video_tensor_to_gif, num_to_groups, noop, cast_num_frames, identity, exists
from utils import DATA_DIRECTORY, TENSORBOARD_DIRECTORY, setup_logger, setup_experiment_directory, get_now_str

def ddp_setup(rank: int, world_size: int): 
    os.environ["MASTER_ADDR"] = "localhost" 
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

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

parser.add_argument_group(title="Distributed Training")
parser.add_argument("--distributed", action="store_true", help="Enable multi-gpu training.")
#parser.add_argument("--rank", type=int, default=0, help="Process rank.")

parser.add_argument_group(title="Data")
parser.add_argument("--data-dir", type=str, default="no_obstacles_64")
parser.add_argument("--num-channels", type=int, default=1)
parser.add_argument("--image-size", type=int, default=64)
parser.add_argument("--frames-per-video", type=int, default=20)

def main(rank: int, args, log, experiment_directory, world_size: Optional[int]=0): 
    # configure distributed training 
    if args.distributed: 
        print("running distributed!")
        print(f"Device {rank} online")
        log.info(f"Device {rank} online")
        ddp_setup(rank, world_size)
        log.info("Set up DDP")
        print("Set up DDP")

    # configure logging 
    if args.distributed and rank != 0: 
        writer: SummaryWriter = None
    else: 
        writer: SummaryWriter = SummaryWriter(log_dir=TENSORBOARD_DIRECTORY / get_now_str())

    if not args.distributed: 
        # configure GPU support 
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
        log.info(f"GPU support: {'YES' if torch.cuda.is_available() else 'NO'}")
        log.info(f"GPUs Available: {torch.cuda.device_count()}")

    # configure the model 
    unet = Unet3D(dim=args.model_dimension, channels=args.num_channels, dim_mults = (1, 2, 4, 8))

    if args.distributed: 
        unet.to(rank)
    else: 
        unet.to(device)

    diffusion = GaussianDiffusion(
        unet,
        channels=args.num_channels, 
        image_size=args.image_size,
        num_frames=args.frames_per_video,
        timesteps=args.num_timesteps,   
        loss_type='l1'
    )

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
    print(f"Found {num_videos=} videos at {data_directory=}")
    assert num_videos >= 1 

    if args.distributed: 
        dataloader: DataLoader = cycle(DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=DistributedSampler(dataset)))
    else: 
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
            device_id=rank if args.distributed else torch.cuda.current_device(), 
            checkpoint_every=args.sample_every, 
            log=log, 
            experiment_directory=experiment_directory, 
            num_to_sample=args.num_to_sample, 
            writer=writer, 
            distributed=args.distributed, 
            gradient_accumulate_every=args.gradient_accumulate_every, 
            max_gradient_norm=max_gradient_norm, 
            present_focus_probability=args.focus_present
            )
    trainer: Trainer = Trainer(trainer_config)
    trainer.train(args.num_steps)
    writer.close()
    destroy_process_group()

if __name__=="__main__": 
    args = parser.parse_args()
    experiment_directory: Path = setup_experiment_directory("control_unconditional")
    log: logging.Logger = setup_logger(__name__, custom_handle=experiment_directory / "log.out")

    if args.distributed: 
        world_size: int = torch.cuda.device_count()
        mp.spawn(main, args=(args, log, experiment_directory, world_size,), nprocs=world_size)

    main(0, args)
