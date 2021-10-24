import os

from typing import List

from pydantic import BaseSettings


class Settings(BaseSettings):
    MODEL_DIR: str
    MODEL_FILE: str
    MODEL_SHAPE: List[int]
    MODEL_THRESHOLD: float
    REPLACE_MESSAGE: str
    NO_REPLACE_MESSAGE: str

    class Config:
        env_file = ".env"

