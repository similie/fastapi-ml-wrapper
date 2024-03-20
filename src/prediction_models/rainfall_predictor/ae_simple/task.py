
import os
from os import path, getcwd
import math
import torch
from datetime import timedelta
from pytz import UTC
import pandas as pd
from ..AllWeatherConfig import getAllWeatherMLConfig
from ..PredictionPostResponse import (
    RainfallPrediction,
    StationRainfallPrediction,
)
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
# model imports
from layers.model import Autoencoder
from mutils import generate_datetime_index
from predict import _predict, generate_predictions 


predictions = _predict(startDate, data)
