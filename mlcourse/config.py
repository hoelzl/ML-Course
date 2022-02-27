# %%
import logging
import platform
from multiprocessing import cpu_count
from pathlib import Path

from pydantic.env_settings import BaseSettings

import mlcourse.data as mlp_data
import mlcourse.saved_models as mlp_models


# %%
class Config(BaseSettings):
    # Paths
    data_dir_path = Path(mlp_data.__path__[0])
    model_dir_path = Path(mlp_models.__path__[0])

    # Logging
    logfile: Path = data_dir_path / "mlcourse.log"
    loglevel: int = logging.DEBUG

    # Multiprocessing
    max_processes: int = 8 if platform.system() == "Windows" else cpu_count()

    def __init__(self):
        super().__init__()
        logging.basicConfig(level=self.loglevel, filename=self.logfile)
        self.data_dir_path.mkdir(parents=True, exist_ok=True)
        self.model_dir_path.mkdir(parents=True, exist_ok=True)

    class Config:
        env_prefix = "MLP_"
        fields = {
            "data_dir_path": {
                "env": "DATA_DIR",
            },
            "model_dir_path": {
                "env": "MODEL_DIR",
            },
            "logfile": {
                "env": "LOGFILE",
            },
            "loglevel": {
                "env": "LOGLEVEL",
            },
            "max_processes": {
                "env": "MAX_PROCESSES",
            },
        }


# %%
