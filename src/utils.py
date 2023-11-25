import datetime
import logging 
from pathlib import Path
import pickle 
from typing import Any, List, Sequence, Tuple, Union

import torch 

from constants import Tensor

# project directory configuration 
SOURCE_DIRECTORY: Path = Path(__file__).parent.absolute()
PROJECT_DIRECTORY: Path = SOURCE_DIRECTORY.parent.absolute() 
MEDIA_DIRECTORY: Path = PROJECT_DIRECTORY / "media" 
DATA_DIRECTORY: Path = PROJECT_DIRECTORY / "data" 
LOG_DIRECTORY: Path = PROJECT_DIRECTORY / "logs" 
TENSORBOARD_DIRECTORY: Path = PROJECT_DIRECTORY / "tensorboard_logs"

auto_built_directories: Tuple[Path] = (
        MEDIA_DIRECTORY, 
        DATA_DIRECTORY, 
        LOG_DIRECTORY, 
        TENSORBOARD_DIRECTORY, 
        )

for path in auto_built_directories: 
    path.mkdir(exist_ok=True) 

def serialize(obj: Any, location: Path):
    if location[-4:] != ".pkl":
        location += ".pkl"

    with open(location, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def deserialize(location: Path) -> Any:
    with open(location, "rb") as handle:
        result = pickle.load(handle)
    return result

def human_seconds_str(seconds: int) -> str:
    units: Tuple[str] = ("seconds", "milliseconds", "microseconds")
    power: int = 1

    for unit in units:
        if seconds > power:
            return f"{seconds:.1f} {unit}"

        seconds *= 1000

    return f"{int(seconds)} nanoseconds"

def get_now_str() -> str:
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def setup_experiment_directory(application: str) -> Path: 
    now: str = get_now_str()
    prefix_dir: Path = LOG_DIRECTORY / application

    if not prefix_dir.exists(): 
        prefix_dir.mkdir(exist_ok=False)

    directory: Path = prefix_dir / now 
    directory.mkdir(exist_ok=False)
    return directory

def level_from_args(args) -> int:
    """Determines an integer-valued logging level from an argparse namespace. 
    It is assumed that the namespace contains both a `debug` and `verbose` key, 
    otherwise a default logging level of logging.WARN is used. 
    """
    try: 
        level: int = logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARN
    except: 
        level: int = logging.WARN 
    
    return level

def setup_logger(name: str, level: int = logging.INFO, custom_handle: Path=None) -> logging.Logger:
    # --- create the entry point logger
    logger = logging.getLogger(name)

    if not getattr(logger, "handler_set", None):
        # --- add the file handler
        log_file: Path = custom_handle if custom_handle is not None else LOG_DIRECTORY / f"{get_now_str()}.out"
        file_handler = logging.FileHandler(log_file)

        # --- format the file handler
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler.setFormatter(fmt)

        # --- setup stream handler 
        console = logging.StreamHandler() 
        console.setLevel(level) 
        console.setFormatter(fmt) 

        # --- configure the logger
        logger.addHandler(file_handler)
        logger.addHandler(console)
        logger.setLevel(level)

        # --- don't add more handlers next time
        logger.handler_set = True
        logger.propagate = False

    return logger

def corrupt(encoder: callable, x: Tensor, timestep: Tensor) -> Tensor: 
    code: Tensor = encoder(x, timestep) 
    return code 
