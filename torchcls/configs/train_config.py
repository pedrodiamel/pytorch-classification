from dataclasses import dataclass, field
from typing import Any, List

from omegaconf import MISSING


@dataclass
class ProjectConfig:
    path: str = MISSING
    name_prefix: str = MISSING
    version: float = 0.01


@dataclass
class DataConfig:
    path: str = MISSING
    name_dataset: str = MISSING
    epochs: int = 10
    batch_size: int = 32
    workers: int = 3
    auto_balance: bool = False


@dataclass
class ModelCheckpointConf:
    print_freq: int = 100
    resume: str = MISSING
    snapshot: int = 5
    verbose: bool = True


@dataclass
class TrainConfig:
    lr: float = 1e-4
    momentun: float = 0.9
    cuda: bool = True
    gpu: int = 0
    parallel: bool = False
    arch: str = MISSING
    loss: str = MISSING
    opt: str = MISSING
    scheduler: str = MISSING
    numclass: int = 2
    numchannels: int = 3
    image_size: int = 28
    finetuning: bool = False


@dataclass
class AugmentationConfig:
    transforms_train: Any = None
    transforms_val: Any = None
    transforms_test: Any = None


@dataclass
class Config:
    project: ProjectConfig = ProjectConfig()
    data: DataConfig = DataConfig()
    trainer: TrainConfig = TrainConfig()
    checkpoint: ModelCheckpointConf = ModelCheckpointConf()
    seed: int = 123456  # Seed for generators
