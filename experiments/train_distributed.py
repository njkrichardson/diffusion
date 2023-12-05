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
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, enable_wrap, wrap

from constants import Module, Tensor
from custom_datasets import ControlDataset
from custom_tensorboard import SummaryWriter
from trainer import TrainerConfig, DataParallelTrainer, ShardedTrainer
from train_utils import EMA, CHANNELS_TO_MODE, cycle, seek_all_images, gif_to_tensor, video_tensor_to_gif, num_to_groups, noop, cast_num_frames, identity, exists
from utils import DATA_DIRECTORY, TENSORBOARD_DIRECTORY, setup_logger, setup_experiment_directory, get_now_str

def ddp_setup(): 
    init_process_group(backend="nccl")

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
parser.add_argument("--from-checkpoint", type=str, default=None, help="Path to experiment directory with snapshots.")
parser.add_argument("--shard-threshold", type=int, default=100, help="Threshold (in units of number of parameters) beyond which a module should be sharded.")

parser.add_argument_group(title="Data")
parser.add_argument("--data-dir", type=str, default="no_obstacles_64")
parser.add_argument("--num-channels", type=int, default=1)
parser.add_argument("--image-size", type=int, default=64)
parser.add_argument("--frames-per-video", type=int, default=20)

def main(args, experiment_directory): 
    log = setup_logger(__name__, custom_handle=experiment_directory / "log.out")
    log.info("logger online")
    log_fn = log.info 

    world_size = int(os.environ['WORLD_SIZE'])
    local_rank: int = int(os.environ["LOCAL_RANK"])
    rank: int = int(os.environ["RANK"])

    # configure distributed training 
    if args.distributed: 
        log_fn("running distributed!")
        log_fn(f"Device {os.environ['LOCAL_RANK']} online")
        log_fn(f"World size: {world_size}")
        ddp_setup()
        log_fn("Set up DDP")


    # configure logging 
    if args.distributed and int(os.environ["LOCAL_RANK"]) != 0: 
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
        unet.to(int(os.environ["LOCAL_RANK"]))
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
    log_fn(f"Found {num_videos=} videos at {data_directory=}")
    assert num_videos >= 1 

    if args.distributed: 
        dataloader: DataLoader = cycle(DataLoader(dataset, num_workers=2, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=DistributedSampler(dataset, rank=rank, num_replicas=world_size)))
    else: 
        dataloader: DataLoader = cycle(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True))

    # configure optimizer
    optimizer = Adam(model.parameters(), lr=args.step_size)
    step: int = 0 

    # gradient scaling 
    gradient_scaler: GradScaler = GradScaler(enabled=(not args.no_amp))
    max_gradient_norm: float = None

    # configure the trainer 
    sharding_policy: callable = partial(size_based_auto_wrap_policy, min_num_params=args.shard_threshold)
    trainer_config: TrainerConfig = TrainerConfig(
            model=model, 
            dataloader=dataloader, 
            optimizer=optimizer, 
            ema=ema, 
            ema_model=ema_model, 
            gradient_scaler=gradient_scaler,
            batch_size=args.batch_size, 
            num_to_sample=args.num_to_sample, 
            checkpoint_every=args.sample_every, 
            distributed=args.distributed, 
            log=log_fn, 
            experiment_directory=experiment_directory, 
            writer=writer, 
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
    if args.from_checkpoint is not None: 
        experiment_directory = Path(args.from_checkpoint)
    else: 
        experiment_directory: Path = setup_experiment_directory("control_unconditional", exists_ok=True)

    main(args, experiment_directory)
