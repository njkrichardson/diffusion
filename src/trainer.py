from dataclasses import dataclass 
from pathlib import Path
import logging
from typing import Optional

from einops import rearrange
import torch 
from torch.utils.data import Dataset, DataLoader 
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP 

from constants import Module, Optimizer, Tensor
from custom_tensorboard import SummaryWriter
from train_utils import exists, num_to_groups, video_tensor_to_gif

@dataclass 
class TrainerConfig: 
    model: Module 
    dataloader: DataLoader 
    optimizer: Optimizer 
    ema: Module 
    ema_model: Module 
    gradient_scaler: Module
    device_id: int 
    checkpoint_every: int 
    log: logging.Logger 
    experiment_directory: Path
    num_to_sample: Optional[int]=2
    writer: Optional[SummaryWriter]=None
    distributed: Optional[bool]=False
    update_ema_every: Optional[int]=10
    step_start_ema: Optional[int]=2000
    gradient_accumulate_every: Optional[int]=2 
    max_gradient_norm: Optional[float] = None
    present_focus_probability: Optional[float]=0.
    present_focus_mask: Optional[Tensor]=None

class Trainer: 
    """Generic training class with support for distributed data parallelism.
    """
    def __init__(self, config: TrainerConfig): 
        self.config = config 
        self.optimizer: Optimizer = self.config.optimizer

        # enable DDP 
        self.model: Module = self.config.model.to(self.config.device_id)
        
        if self.config.distributed: 
            self.model = DDP(self.model, device_ids=[self.config.device_id])

        self.dataloader: DataLoader = self.config.dataloader
        self.ema: Module = self.config.ema 
        self.ema_model: Module = self.config.ema_model
        self.gradient_scaler: Module = self.config.gradient_scaler
        self.device_id: int = self.config.device_id
        self.checkpoint_every: int = self.config.checkpoint_every
        self.log: logging.Logger = self.config.log 
        self.writer = self.config.writer 

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
            batch: Tensor = next(self.dataloader).to(self.device_id)
            self._run_batch(batch) 

        if exists(self.config.max_gradient_norm):
            self.gradient_scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_gradient_norm)

        self.gradient_scaler.step(self.optimizer)
        self.gradient_scaler.update()

    def _save_checkpoint(self, step: int): 
        if step != 0 and step % self.config.sample_every == 0 and self.device_id == 0:
            milestone: int = step // self.config.sample_every
            num_samples: int = self.config.num_to_sample ** 2
            batches = num_to_groups(num_samples, self.config.batch_size)

            all_videos_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
            all_videos_list = torch.cat(all_videos_list, dim = 0)
            all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

            one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=self.config.num_to_sample)
            video_path = str(self.config.experiment_directory / str(f'samples_{milestone}.gif'))
            video_tensor_to_gif(one_gif, video_path)
            checkpoint_data: dict = {
                'step': step,
                'model': self.model.module.state_dict() if self.config.distributed else self.model.state_dict(),
                'ema': self.ema_model.state_dict(),
                'scaler': self.gradient_scaler.state_dict()
            }
            torch.save(checkpoint_data, str(self.config.experiment_directory / f'checkpoint-{milestone}.pt'))

    def train(self, num_steps: int): 
        for step in range(num_steps): 
            self._run_step(step) 

            self.log.info(f"GPU{self.config.device_id} | Iteration [{step:05d}/{num_steps:05d}] | Objective: {self.current_objective:0.4f}")
            print(f"GPU{self.config.device_id} | Iteration [{step:05d}/{num_steps:05d}] | Objective: {self.current_objective:0.4f}")

            if self.writer is not None: 
                self.writer.scalar("Objective", self.current_objective, step=step)

            if step % self.config.update_ema_every == 0:
                if step < self.config.step_start_ema:
                    self.ema_model.load_state_dict(self.model.module.state_dict())
                else: 
                    self.ema.update_model_average(self.ema_model, self.model)

            if step != 0 and step % self.config.checkpoint_every == 0:
                self._save_checkpoint(step)
