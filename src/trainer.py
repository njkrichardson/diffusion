from dataclasses import dataclass 
from pathlib import Path
import logging
import os 
from typing import List, Optional

from einops import rearrange
import torch 
from torch.utils.data import Dataset, DataLoader 
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload

from constants import Module, Optimizer, Tensor
from custom_tensorboard import SummaryWriter
from train_utils import exists, num_to_groups, video_tensor_to_gif

@dataclass 
class TrainerConfig: 
    # Modules 
    model: Module 
    dataloader: DataLoader 
    optimizer: Optimizer 
    ema: Module 
    ema_model: Module 
    gradient_scaler: Module

    # Data 
    batch_size: int 
    num_to_sample: int

    # Compute and distributed training
    checkpoint_every: int 
    distributed: bool

    # Diagnostics 
    log: callable
    experiment_directory: Path
    writer: Optional[SummaryWriter]=None

    # EMA and gradient scaling 
    update_ema_every: Optional[int]=10
    step_start_ema: Optional[int]=2000
    gradient_accumulate_every: Optional[int]=2 
    max_gradient_norm: Optional[float] = None
    present_focus_probability: Optional[float]=0.
    present_focus_mask: Optional[Tensor]=None

class DataParallelTrainer: 
    """Generic training class with support for distributed data parallelism.
    """
    def __init__(self, config: TrainerConfig): 
        self.config = config 
        self.optimizer: Optimizer = self.config.optimizer

        self.local_rank: int = int(os.environ["LOCAL_RANK"])
        self.global_rank: int = int(os.environ["RANK"])

        # enable DDP 
        self.model: Module = self.config.model.to(self.local_rank)
        
        self.dataloader: DataLoader = self.config.dataloader
        self.ema: Module = self.config.ema 
        self.ema_model: Module = self.config.ema_model.to(self.local_rank)
        self.gradient_scaler: Module = self.config.gradient_scaler
        self.checkpoint_every: int = self.config.checkpoint_every
        self.log: callable = self.config.log 
        self.writer = self.config.writer 
        
        # device management 

        # fault tolerance 
        self.steps_completed: int = 0 
        snapshots: List[Path] = list(self.config.experiment_directory.glob("**/*snapshot*"))
        if len(snapshots): 
            snapshot_versions: List[int] = [int(snapshot.with_suffix('').as_posix().split('-')[-1]) for snapshot in snapshots]
            most_recent_snapshot: Path = snapshots[max(enumerate(snapshot_versions), key=lambda x: x[1])[0]]
            self._load_snapshot(most_recent_snapshot)

        if self.config.distributed: 
            self.model = DDP(self.model, device_ids=[self.local_rank])
            self.model = FullyShardedDataParallel(self.model(), fsdp_auto_wrap_policy=default_auto_wrap_policy, cpu_offload=CPUOffload(offload_params=True))


    def _load_snapshot(self, snapshot_path: Path) -> None: 
        # find most recent 
        snapshot: dict = torch.load(snapshot_path, map_location="cpu") 

        self.log(f"Loaded snapshot from: {snapshot_path.absolute().as_posix()}")

        self.steps_completed: int = snapshot['step']
        self.log(f"Snapshot steps completed: {self.steps_completed}")

        self.model.load_state_dict(snapshot['model'])
        self.ema_model.load_state_dict(snapshot['ema'])
        self.gradient_scaler.load_state_dict(snapshot['scaler'])

        self.model.to(self.local_rank)
        self.ema_model.to(self.local_rank)

    def _run_batch(self, batch: Tensor) -> None: 
        objective: Tensor = self.model(
                batch, 
                prob_focus_present=self.config.present_focus_probability, 
                focus_present_mask=self.config.present_focus_mask
                )
        scaled_objective: Tensor = self.gradient_scaler.scale(objective / self.config.gradient_accumulate_every)
        scaled_objective.backward()
        self.current_objective: float = objective.item()

    def _run_step(self, step: int): 
        self.optimizer.zero_grad()

        for _ in range(self.config.gradient_accumulate_every):
            batch: Tensor = next(self.dataloader).to(self.local_rank)
            self._run_batch(batch) 

        if exists(self.config.max_gradient_norm):
            self.gradient_scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_gradient_norm)

        self.gradient_scaler.step(self.optimizer)
        self.gradient_scaler.update()

    def _save_snapshot(self, step: int): 
        if step != 0 and step % self.config.checkpoint_every == 0 and self.global_rank == 0:
            milestone: int = step // self.config.checkpoint_every
            num_samples: int = self.config.num_to_sample ** 2
            batches = num_to_groups(num_samples, self.config.batch_size)

            all_videos_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
            all_videos_list = torch.cat(all_videos_list, dim = 0)
            all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

            one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=self.config.num_to_sample)
            video_path = str(self.config.experiment_directory / str(f'samples_{milestone}.gif'))
            video_tensor_to_gif(one_gif, video_path)
            self.log(f"Saved samples to: {video_path}")

            snapshot: dict = {
                'step': step,
                'model': self.model.module.state_dict() if self.config.distributed else self.model.state_dict(),
                'ema': self.ema_model.state_dict(),
                'scaler': self.gradient_scaler.state_dict()
            }
            snapshot_path: Path = self.config.experiment_directory / f"snapshot-{milestone}.pt"
            torch.save(snapshot, snapshot_path)
            self.log(f"Saved snapshot to: {snapshot_path.as_posix()}")

    def train(self, num_steps: int): 
        for step in range(self.steps_completed, num_steps): 
            self._run_step(step) 

            self.log(f"GPU{self.global_rank} | Iteration [{step:05d}/{num_steps:05d}] | Batch size: {self.config.batch_size} | Objective: {self.current_objective:0.4f}")

            if self.writer is not None: 
                self.writer.scalar("Objective", self.current_objective, step=step)

            if step % self.config.update_ema_every == 0:
                if step < self.config.step_start_ema:
                    self.ema_model.load_state_dict(self.model.module.state_dict())
                else: 
                    self.ema.update_model_average(self.ema_model, self.model)

            if step != 0 and step % self.config.checkpoint_every == 0:
                self._save_snapshot(step)

class DataParallelTrainer: 
    """Generic training class with support for distributed data parallelism.
    """
    def __init__(self, config: TrainerConfig): 
        self.config = config 
        self.optimizer: Optimizer = self.config.optimizer

        self.local_rank: int = int(os.environ["LOCAL_RANK"])
        self.global_rank: int = int(os.environ["RANK"])

        # enable DDP 
        self.model: Module = self.config.model.to(self.local_rank)
        
        self.dataloader: DataLoader = self.config.dataloader
        self.ema: Module = self.config.ema 
        self.ema_model: Module = self.config.ema_model.to(self.local_rank)
        self.gradient_scaler: Module = self.config.gradient_scaler
        self.checkpoint_every: int = self.config.checkpoint_every
        self.log: callable = self.config.log 
        self.writer = self.config.writer 
        
        # device management 

        # fault tolerance 
        self.steps_completed: int = 0 
        snapshots: List[Path] = list(self.config.experiment_directory.glob("**/*snapshot*"))
        if len(snapshots): 
            snapshot_versions: List[int] = [int(snapshot.with_suffix('').as_posix().split('-')[-1]) for snapshot in snapshots]
            most_recent_snapshot: Path = snapshots[max(enumerate(snapshot_versions), key=lambda x: x[1])[0]]
            self._load_snapshot(most_recent_snapshot)

        if self.config.distributed: 
            self.model = DDP(self.model, device_ids=[self.local_rank])
            self.model = FullyShardedDataParallel(self.model(), fsdp_auto_wrap_policy=default_auto_wrap_policy, cpu_offload=CPUOffload(offload_params=True))


    def _load_snapshot(self, snapshot_path: Path) -> None: 
        # find most recent 
        snapshot: dict = torch.load(snapshot_path, map_location="cpu") 

        self.log(f"Loaded snapshot from: {snapshot_path.absolute().as_posix()}")

        self.steps_completed: int = snapshot['step']
        self.log(f"Snapshot steps completed: {self.steps_completed}")

        self.model.load_state_dict(snapshot['model'])
        self.ema_model.load_state_dict(snapshot['ema'])
        self.gradient_scaler.load_state_dict(snapshot['scaler'])

        self.model.to(self.local_rank)
        self.ema_model.to(self.local_rank)

    def _run_batch(self, batch: Tensor) -> None: 
        objective: Tensor = self.model(
                batch, 
                prob_focus_present=self.config.present_focus_probability, 
                focus_present_mask=self.config.present_focus_mask
                )
        scaled_objective: Tensor = self.gradient_scaler.scale(objective / self.config.gradient_accumulate_every)
        scaled_objective.backward()
        self.current_objective: float = objective.item()

    def _run_step(self, step: int): 
        self.optimizer.zero_grad()

        for _ in range(self.config.gradient_accumulate_every):
            batch: Tensor = next(self.dataloader).to(self.local_rank)
            self._run_batch(batch) 

        if exists(self.config.max_gradient_norm):
            self.gradient_scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_gradient_norm)

        self.gradient_scaler.step(self.optimizer)
        self.gradient_scaler.update()

    def _save_snapshot(self, step: int): 
        if step != 0 and step % self.config.checkpoint_every == 0 and self.global_rank == 0:
            milestone: int = step // self.config.checkpoint_every
            num_samples: int = self.config.num_to_sample ** 2
            batches = num_to_groups(num_samples, self.config.batch_size)

            all_videos_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
            all_videos_list = torch.cat(all_videos_list, dim = 0)
            all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

            one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=self.config.num_to_sample)
            video_path = str(self.config.experiment_directory / str(f'samples_{milestone}.gif'))
            video_tensor_to_gif(one_gif, video_path)
            self.log(f"Saved samples to: {video_path}")

            snapshot: dict = {
                'step': step,
                'model': self.model.module.state_dict() if self.config.distributed else self.model.state_dict(),
                'ema': self.ema_model.state_dict(),
                'scaler': self.gradient_scaler.state_dict()
            }
            snapshot_path: Path = self.config.experiment_directory / f"snapshot-{milestone}.pt"
            torch.save(snapshot, snapshot_path)
            self.log(f"Saved snapshot to: {snapshot_path.as_posix()}")

    def train(self, num_steps: int): 
        for step in range(self.steps_completed, num_steps): 
            self._run_step(step) 

            self.log(f"GPU{self.global_rank} | Iteration [{step:05d}/{num_steps:05d}] | Batch size: {self.config.batch_size} | Objective: {self.current_objective:0.4f}")

            if self.writer is not None: 
                self.writer.scalar("Objective", self.current_objective, step=step)

            if step % self.config.update_ema_every == 0:
                if step < self.config.step_start_ema:
                    self.ema_model.load_state_dict(self.model.module.state_dict())
                else: 
                    self.ema.update_model_average(self.ema_model, self.model)

            if step != 0 and step % self.config.checkpoint_every == 0:
                self._save_snapshot(step)
