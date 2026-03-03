from src.entitiy.Artifacts_entitiy import DataIngestionArtifacts,DataTransformationArtifacts
from src.entitiy.Config_entitiy import DataIngestionConfig,DataTransformationConfig
from src.compenents import data_ingestion,data_transformation
from pathlib import Path
import yaml

def run_data_pipeline():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    ingestion_cfg = DataIngestionConfig(
    kaggle_dataset=cfg["data_ingestion"]["kaggle_dataset"],
    output_data_path=Path(cfg["data_ingestion"]["output_data_path"])
)

    ingestion=data_ingestion.DataIngestion(data_ingestion_config=ingestion_cfg)
    data_ingestion_artifacts=ingestion.Read_And_Write()

    transformation_cfg = DataTransformationConfig(
        test_size=cfg["data_transformation"]["test_size"],
        train_ouput_path=Path(cfg["data_transformation"]["train_output_path"]),
        test_output_path=Path(cfg["data_transformation"]["test_output_path"]),
        pickle_encoder_obj_path=Path(cfg["data_transformation"]["pickle_encoder_obj_path"]),
        pickle_vectorizer_obj_path=Path(cfg["data_transformation"]["pickle_vectorizer_obj_path"])
    )
    
    transformation=data_transformation.DataTransformation(data_transformation_config=transformation_cfg,data_ingestion_artifacts=data_ingestion_artifacts)

    data_transformation_artifacts=transformation.Initiate_Transformation()
    
    return data_transformation_artifacts

