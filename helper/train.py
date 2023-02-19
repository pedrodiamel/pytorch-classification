# Exemplo
# python helper/train.py +configs=preactresnet18_v1

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from torchcls.configs.train_config import Config
from torchcls.training import train


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: Config) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
