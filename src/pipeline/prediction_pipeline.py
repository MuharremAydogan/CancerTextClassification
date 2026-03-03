from keras.models import load_model
from pathlib import Path
import pickle
import numpy as np
class ModelPredict:
    def __init__(self, model_path: Path, encoder_path: Path, tfidf_path: Path):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.tfidf_path = tfidf_path

        self.model = None
        self.label_encoder = None
        self.tfidf_encoder = None

    def get_object(self):

        if self.model is None:
            self.model = load_model(self.model_path)

        if self.label_encoder is None:
            with open(self.encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)

        if self.tfidf_encoder is None:
            with open(self.tfidf_path, "rb") as f:
                self.tfidf_encoder = pickle.load(f)

        return self.model, self.label_encoder, self.tfidf_encoder
    
    def predict(self, texts):

        model, label_encoder, tfidf = self.get_object()

        if isinstance(texts, str):
            texts = [texts]

        X = tfidf.transform(texts)

        preds = model.predict(X)

        
        if preds.shape[1] == 1:
            class_ids = (preds > 0.5).astype(int).ravel()
        else:
            class_ids = np.argmax(preds, axis=1)

        labels = label_encoder.inverse_transform(class_ids)

        return labels    
