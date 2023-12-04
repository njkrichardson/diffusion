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
from trainer import EMA, CHANNELS_TO_MODE, cycle, seek_all_images, gif_to_tensor, video_tensor_to_gif, num_to_groups, noop, cast_num_frames, identity, exists
from utils import DATA_DIRECTORY, TENSORBOARD_DIRECTORY, setup_logger, setup_experiment_directory, get_now_str

parser = argparse.ArgumentParser()
parser.add_argument_group(title="Architecture")
parser.add_argument("--model-dimension", type=int, default=64)
parser.add_argument("--num-timesteps", "-t", type=int, default=1000)

parser.add_argument_group(title="Diagnostics")
parser.add_argument("--sample-every", type=int, default=1000)
parser.add_argument("--num-to-sample", type=int, default=2)

parser.add_argument_group(title="Training")
parser.add_argument("--num-epochs", type=int, default=10_000)
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

class Trainer:
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        num_frames = 16,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        num_sample_rows = 4,
        max_grad_norm = None
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        self.ds = ControlDataset(folder, image_size, channels = channels, num_frames = num_frames)

        print(f'found {len(self.ds)} videos as gif files at {folder}')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.dl = cycle(DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    def train(
        self,
        prob_focus_present = 0.,
        focus_present_mask = None,
        log_fn = noop, 
        writer = None
    ):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()

                with autocast(enabled = self.amp):
                    loss = self.model(
                        data,
                        prob_focus_present = prob_focus_present,
                        focus_present_mask = focus_present_mask
                    )

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

            log_fn(f'{self.step}: {loss.item()}')
            
            if writer is not None: 
                writer.scalar("Objective", loss.item(), step=self.step)

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                num_samples = self.num_sample_rows ** 2
                batches = num_to_groups(num_samples, self.batch_size)

                all_videos_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_videos_list = torch.cat(all_videos_list, dim = 0)

                all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

                one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows)
                video_path = str(self.results_folder / str(f'{milestone}.gif'))
                video_tensor_to_gif(one_gif, video_path)
                self.save(milestone)

            self.step += 1

        log_fn('training completed')

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
    dataset: Dataset = ControlDataset(args.data_dir, args.image_size, channels=args.num_channels, num_frames=args.frames_per_video)
    num_videos: int = len(dataset)
    log.info(f"Found {num_videos=} videos at {args.data_dir=}")
    assert num_videos >= 1 
    dataloader: DataLoader = cycle(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True))

    # configure optimizer
    optimizer = Adam(model.parameters(), lr=args.step_size)
    step: int = 0 

    # gradient scaling 
    gradient_scaler: GradScaler = GradScaler(enabled=(not args.no_amp))
    max_gradient_norm: float = None

    while step < args.num_epochs:
        for i in range(args.gradient_accumulate_every):
            batch: Tensor = next(dataloader).cuda()

            with autocast(enabled=(not args.no_amp)):
                loss: Tensor = model(batch, prob_focus_present=args.focus_present, focus_present_mask=None)
                scaled_loss: Tensor = gradient_scaler.scale(loss / args.gradient_accumulate_every)
                scaled_loss.backward()

        log.info(f"Iteration [{step:05d}/{args.num_epochs:05d}]:\t{loss.item():0.4f}")
        
        if writer is not None: 
            writer.scalar("Objective", loss.item(), step=step)

        if exists(max_gradient_norm):
            gradient_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)

        gradient_scaler.step(optimizer)
        gradient_scaler.update()
        optimizer.zero_grad()

        if step % args.update_ema_every == 0:
            if step < args.step_start_ema:
                ema_model.load_state_dict(model.state_dict())
            else: 
                ema.update_model_average(ema_model, model)

        if step != 0 and step % args.sample_every == 0:
            milestone: int = step // args.sample_every
            num_samples: int = args.num_to_sample ** 2
            batches = num_to_groups(num_samples, args.batch_size)

            all_videos_list = list(map(lambda n: ema_model.sample(batch_size=n), batches))
            all_videos_list = torch.cat(all_videos_list, dim = 0)

            all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

            one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=args.num_to_sample)
            video_path = str(experiment_directory / str(f'{milestone}.gif'))
            video_tensor_to_gif(one_gif, video_path)
            checkpoint_data: dict = {
                'step': step,
                'model': model.state_dict(),
                'ema': ema_model.state_dict(),
                'scaler': scaler.state_dict()
            }
            torch.save(data, str(experiment_directory / f'checkpoint-{milestone}.pt'))

        step += 1

    log_fn('training completed')

    writer.close()

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
