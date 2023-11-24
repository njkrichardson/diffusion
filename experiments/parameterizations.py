import argparse 
import logging
from pathlib import Path 

import torch 
from torch.optim import Adam 
from torch.utils.tensorboard import SummaryWriter

from constants import Tensor 
from custom_objectives import score_matching_objective
from forward import bind_diffusion, linear_schedule
from nnets import AffineNetConfig, AffineNet
from utils import TENSORBOARD_DIRECTORY, setup_logger, setup_experiment_directory

parser = argparse.ArgumentParser()

parser.add_argument_group("Forward diffusion options")
parser.add_argument("--num-timesteps", "-t", type=int, default=300, help="Number of steps taken to encode/diffuse data into noise codes.")

parser.add_argument_group("Training options")
parser.add_argument("--num-epochs", "--epochs", type=int, default=100, help="Number of epochs to run optimization.")
parser.add_argument("--step-size", "-lr", type=float, default=1e-03, help="Step size to use for gradient steps.")

parser.add_argument_group("Diagnostics")
parser.add_argument("--report-every", type=int, default=10, help="Log the current objective at this periodicity (in units of epochs).")

def main(args): 
    # configure logging 
    experiment_directory: Path = setup_experiment_directory("parameterizations")
    log: logging.Logger = setup_logger(__name__, custom_handle=experiment_directory / "log.out")
    writer: SummaryWriter = SummaryWriter(log_dir=TENSORBOARD_DIRECTORY)

    # configure GPU support 
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    log.info(f"GPU support: {'YES' if torch.cuda.is_available() else 'NO'}")

    # generate some synthetic data 
    data_distribution: callable = lambda shape: torch.randn(*shape) + 5.
    num_observations: int = 100 
    observation_dimension: int = 1 
    observations: Tensor = data_distribution((num_observations, observation_dimension)).to(device) 
    log.info(f"Sampled {num_observations} observations of dimension {observation_dimension}")

    # instantiate encoder 
    encoder: callable = bind_diffusion(args.num_timesteps, linear_schedule) 
    log.info(f"Configured diffusion process with {args.num_timesteps} timesteps")

    # instantiate decoder 
    decoder_config: AffineNetConfig = AffineNetConfig(input_dimension=observation_dimension) 
    decoder: AffineNet = AffineNet(decoder_config)
    decoder.to(device) 

    # configure optimizer 
    optimizer = Adam(decoder.parameters(), lr=args.step_size)

    for epoch in range(args.num_epochs): 
        epoch_objective: Tensor = torch.zeros(1)
        num_batches: int = 0

        # TODO batch 
        for i, batch in enumerate(observations): 
            optimizer.zero_grad()
            batch_size: int = batch.shape[0]
            batch.to(device) 

            # sample (uniformly) a time-step for each observation in the batch 
            timesteps: Tensor = torch.randint(0, args.num_timesteps, (batch_size,), device=device).long()

            # evaluate the objective 
            objective: Tensor = score_matching_objective(decoder, encoder, batch, timesteps)
            epoch_objective += objective 
            num_batches += 1 

            objective.backward()
            optimizer.step()

        if epoch % args.report_every == 0: 
            log.info(f"Epoch [{epoch:04d}/{args.num_epochs:04d}]\tObjective: {(epoch_objective / num_batches).item():0.3f}")

    # TODO model saving
    writer.close()

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
