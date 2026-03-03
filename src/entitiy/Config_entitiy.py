from pathlib import Path
from dataclasses import dataclass
from src.entitiy.Artifacts_entitiy import DataIngestionArtifacts
@dataclass
class DataIngestionConfig:
    kaggle_dataset: str
    output_data_path: Path

@dataclass
class DataTransformationConfig:
    test_size:float
    train_ouput_path:Path
    test_output_path:Path
    pickle_encoder_obj_path:Path
    pickle_vectorizer_obj_path:Path


@dataclass
class ModelConfig:
    save_path:Path
    num_classes:int    
    epoch:int

@dataclass
class ModelEvalConfig:
    tracking_adress:str  
    plot_save_path:Path






