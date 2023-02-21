# Exemplo
# python helper/train.py +configs=preactresnet18_v1

import augmentation
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from torchcls.configs.train_config import AugmentationConfig, Config
from torchcls.training import train


cs = ConfigStore.instance()
cs.store(name="config", node=Config)

# Create augmentation methods
# TODO February 21, 2023: Improve this
aug: AugmentationConfig = AugmentationConfig()
aug.transforms_train = augmentation.get_transforms_aug
aug.transforms_val = augmentation.get_transforms
aug.transforms_test = augmentation.get_transforms


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: Config) -> None:
    train(cfg, aug)


if __name__ == "__main__":
    main()
