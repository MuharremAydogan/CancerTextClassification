
import kagglehub
from src.entitiy.Config_entitiy import DataIngestionConfig
from src.entitiy.Artifacts_entitiy import DataIngestionArtifacts
from pathlib import Path
import pandas as pd
class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def Read_And_Write(self):

        dataset_dir = kagglehub.dataset_download(
            self.data_ingestion_config.kaggle_dataset
        )

        dataset_dir = Path(dataset_dir)

        csv_files = list(dataset_dir.glob("*.csv"))

        if len(csv_files) == 0:
            raise Exception("Dataset içinde csv bulunamadi")

        csv_path = csv_files[0]

        df = pd.read_csv(csv_path, encoding="latin1")

        output_path = self.data_ingestion_config.output_data_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)

        return DataIngestionArtifacts(df=df)