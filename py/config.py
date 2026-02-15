import os
from pydantic_settings import BaseSettings
import ast
from dotenv import load_dotenv
load_dotenv()

class Config(BaseSettings):

    DATA_PATH: str = os.getenv('DATA_PATH')
    MODEL_PATH: str = os.getenv('MODEL_PATH')

    ML_TASK: str = os.getenv('ML_TASK', "classifications")
    MODEL_FEATURE_NAME: str = os.getenv('MODEL_TASK_NAME', "vgg16")
    
    PRECEPTION_METRIC: str = os.getenv('PRECEPTION_METRIC', "safety")
    CITY_STUDIED: str = os.getenv('CITY_STUDIED', "New York")
    YEAR_STUDIED: int = int(os.getenv('YEAR_STUDIED', "2011"))
    DELTA: float  = float(os.getenv('DELTA', "0.42"))

    RANDOM_STATE: int = int(os.getenv('RANDOM_STATE', "42"))
