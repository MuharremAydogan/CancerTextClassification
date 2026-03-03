from src.pipeline import data_pipeline,model_pipeline


transformation_artifacts=data_pipeline.run_data_pipeline()

print(transformation_artifacts.x_train_vectorized.shape)

model_artifacts=model_pipeline.run_pipeline(transformation_artifacts=transformation_artifacts,yaml_path="config.yaml")