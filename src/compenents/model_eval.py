from src.entitiy.Config_entitiy import ModelEvalConfig
from src.entitiy.Artifacts_entitiy import ModelArtifacts,DataTransformationArtifacts
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import mlflow.keras
import mlflow
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models.signature import infer_signature

class ModelEval:
    def __init__(self,eval_cfg:ModelEvalConfig,model_artifacts:ModelArtifacts,transformation_artifacts:DataTransformationArtifacts,num_classes:int):
        self.eval_cfg=eval_cfg
        self.model_artifacts=model_artifacts
        self.transformation_artifacts=transformation_artifacts
        self.num_classes=num_classes

        mlflow.set_tracking_uri(self.eval_cfg.tracking_adress)

    def Evaluation(self):
        x_test = self.transformation_artifacts.x_test_vectorized
        y_test = self.transformation_artifacts.y_test_encoded

        model = self.model_artifacts.model

        
        y_prob = model.predict(x_test)
        y_pred = y_prob.argmax(axis=1)

        
        acc_score = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        
        save_path = self.eval_cfg.plot_save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(range(self.num_classes)),
                    yticklabels=list(range(self.num_classes)))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(save_path)
        

        
        with mlflow.start_run():
            signature = infer_signature(x_test, y_test)
            mlflow.keras.log_model(model, name="kerasmodel", registered_model_name="kerasmodel", signature=signature)
            mlflow.log_metric("test_accuracy", acc_score)
            mlflow.log_artifact(str(save_path), artifact_path="confusion_matrix")

        print(f"Test Accuracy: {acc_score:.4f}")
        print(f"Confusion matrix saved at: {save_path}")
