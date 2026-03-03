from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from src.entitiy.Config_entitiy import ModelConfig
from src.entitiy.Artifacts_entitiy import DataTransformationArtifacts,ModelArtifacts
from pathlib import Path
from keras.models import load_model
class Model:
    def __init__(self,model_cfg_path:ModelConfig,data_transformation_artifacts:DataTransformationArtifacts):
        self.model_cfg=model_cfg_path
        self.data_transformation_artifacts=data_transformation_artifacts

    def Model_Create_And_Fit(self):
        x_train=self.data_transformation_artifacts.x_train_vectorized
        x_test=self.data_transformation_artifacts.x_test_vectorized
        y_train=self.data_transformation_artifacts.y_train_encoded
        y_test=self.data_transformation_artifacts.y_test_encoded
        num_classes=self.model_cfg.num_classes
        input_dim=x_train.shape[1]
        
        y_train_categorical=to_categorical(y_train,num_classes=num_classes)
        y_test_categorical=to_categorical(y_test,num_classes=num_classes)

        model = Sequential()
        model.add(Dense(128, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))

        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  

        model.fit(
            x_train, y_train_categorical,
            validation_data=(x_test, y_test_categorical),
            epochs=self.model_cfg.epoch,
            batch_size=32
        )
        
        save_path=self.model_cfg.save_path
        save_path.parent.mkdir(parents=True,exist_ok=True)
        model.save(save_path)

        model_artifacts=ModelArtifacts(model)  
        
        return model_artifacts

    def model_load(self):
        path=self.model_cfg.save_path
        return load_model(path)


        