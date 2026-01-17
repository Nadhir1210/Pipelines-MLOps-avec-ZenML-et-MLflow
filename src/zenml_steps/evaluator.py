from zenml import step
from ultralytics import YOLO
import mlflow
import os

@step(experiment_tracker="local_mlflow_tracker")
def evaluator(model_path: str, data_path: str) -> float:
    """Evaluator step for YOLOv8."""
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found for evaluation.")
        return 0.0
        
    model = YOLO(model_path)
    metrics = model.val(data=data_path)
    
    # YOLO returns a Results object with speed, map, etc.
    map50 = metrics.results_dict.get('metrics/mAP50(B)', 0)
    
    # Log evaluation metrics to MLflow
    mlflow.log_metric("eval_mAP50", map50)
    
    return float(map50)
