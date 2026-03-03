from src.compenents.model import Model
from src.compenents.model_eval import ModelEval
from src.entitiy.Artifacts_entitiy import ModelArtifacts,DataTransformationArtifacts
from src.entitiy.Config_entitiy import ModelConfig,ModelEvalConfig
from pathlib import Path
import yaml

def run_pipeline(transformation_artifacts:DataTransformationArtifacts,yaml_path:Path):
    with open(yaml_path,"r") as f:
        cfg=yaml.safe_load(f)
    model_cfg=ModelConfig(epoch=cfg["model"]["epoch"],num_classes=cfg["model"]["num_classes"],save_path=Path(cfg["model"]["save_path"]))
    model=Model(model_cfg_path=model_cfg,data_transformation_artifacts=transformation_artifacts)
    model_artifacts=model.Model_Create_And_Fit()
    model_eval_cfg=ModelEvalConfig(tracking_adress=cfg["model_eval"]["tracking_uri"],plot_save_path=Path(cfg["model_eval"]["plot_save_path"]))
    model_eval=ModelEval(eval_cfg=model_eval_cfg,model_artifacts=model_artifacts,transformation_artifacts=transformation_artifacts,num_classes=cfg["model_eval"]["num_classes"])
    model_eval.Evaluation()