from src.entitiy.Artifacts_entitiy import DataIngestionArtifacts, DataTransformationArtifacts
from src.entitiy.Config_entitiy import DataTransformationConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import string
import nltk
import pandas as pd
import pickle

nltk.download("punkt")
nltk.download('stopwords')
from nltk.corpus import stopwords

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifacts: DataIngestionArtifacts):
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.data_transformation_config = data_transformation_config

    def DataReadAndReturn(self):
        
        df = self.data_ingestion_artifacts.df
        df.drop(["Unnamed: 0"], axis=1, inplace=True, errors="ignore")
        df.columns = ["label", "description"]  
        return df

    def DataPreprocessing(self, df: pd.DataFrame):
        
        stop_words = set(stopwords.words("english"))
        cleaned_texts = []

        for cumle in df["description"]:
            tokens = word_tokenize(cumle)
            filtered_tokens = [k.lower() for k in tokens if k.lower() not in stop_words and k not in string.punctuation]
            cleaned_texts.append(" ".join(filtered_tokens))

        df["description"] = cleaned_texts
        return df

    def Split_Data(self, df: pd.DataFrame):
        
        train_df, test_df = train_test_split(df, test_size=self.data_transformation_config.test_size, random_state=42)

        
        self.data_transformation_config.train_ouput_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_transformation_config.test_output_path.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(self.data_transformation_config.train_ouput_path, index=False)
        test_df.to_csv(self.data_transformation_config.test_output_path, index=False)

        return train_df, test_df

    def Vectorized_Data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        
        x_train = train_df.description.values
        y_train = train_df.label.values
        x_test = test_df.description.values
        y_test = test_df.label.values

        
        encoder = LabelEncoder()
        y_train_encoded = encoder.fit_transform(y_train)
        y_test_encoded = encoder.transform(y_test)

        
        self.data_transformation_config.pickle_encoder_obj_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_transformation_config.pickle_encoder_obj_path, "wb") as f:
            pickle.dump(encoder, f)

        
        vectorizer = TfidfVectorizer(max_features=10000)
        x_train_vectorized = vectorizer.fit_transform(x_train).toarray()
        x_test_vectorized = vectorizer.transform(x_test).toarray()

        
        self.data_transformation_config.pickle_vectorizer_obj_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_transformation_config.pickle_vectorizer_obj_path, "wb") as f:
            pickle.dump(vectorizer, f)

        return x_train_vectorized, x_test_vectorized, y_train_encoded, y_test_encoded

    def Initiate_Transformation(self):
        print("Text preprocessing started...")
        df = self.DataReadAndReturn()
        df = self.DataPreprocessing(df)
        train_df, test_df = self.Split_Data(df)
        x_train, x_test, y_train, y_test = self.Vectorized_Data(train_df, test_df)

        artifacts = DataTransformationArtifacts(
            x_train_vectorized=x_train,
            x_test_vectorized=x_test,
            y_train_encoded=y_train,
            y_test_encoded=y_test
        )

        print("Text transformation successfully...")
        return artifacts