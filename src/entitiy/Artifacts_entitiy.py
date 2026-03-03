from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy
from keras.models import Sequential
@dataclass
class DataIngestionArtifacts:
    df:pd.DataFrame

@dataclass
class DataTransformationArtifacts:
    x_train_vectorized:numpy.ndarray
    x_test_vectorized:numpy.ndarray  
    y_train_encoded:numpy.ndarray
    y_test_encoded:numpy.ndarray  

@dataclass 
class ModelArtifacts:
    model:Sequential 