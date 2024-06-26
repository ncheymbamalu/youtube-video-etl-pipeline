from pathlib import Path, PosixPath

from omegaconf import DictConfig, ListConfig, OmegaConf


class Config:
    class Path:
        HOME_DIR: PosixPath = Path(__file__).parent.parent
        SRC_DIR: PosixPath = HOME_DIR / "src"
        DATA_DIR: PosixPath = HOME_DIR / "data"
        ARTIFACTS_DIR: PosixPath = HOME_DIR / "artifacts"
        NOTEBOOKS_DIR: PosixPath = HOME_DIR / "notebooks"


def load_config() -> DictConfig | ListConfig:
    return OmegaConf.load(Config.Path.HOME_DIR / "config.yaml")
