from zenml import step
from ultralytics import YOLO
import mlflow
from pathlib import Path
import os

@step(experiment_tracker="mlflow_tracker")
def trainer(
    data_path: str,
    epochs: int = 3,
    imgsz: int = 320,
    lr0: float = 0.005,
    model_type: str = "yolov8n.pt",
    seed: int = 42
) -> str:
    """Trainer step for YOLOv8."""
    
    run_name = f"zenml_{Path(model_type).stem}_e{epochs}_sz{imgsz}_lr{lr0}"
    
    # Mlflow parameters are automatically logged if experiment_tracker is active
    mlflow.log_params({
        "epochs": epochs,
        "imgsz": imgsz,
        "lr0": lr0,
        "model": model_type,
        "seed": seed
    })

    model = YOLO(model_type)
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        lr0=lr0,
        seed=seed,
        project="runs",
        name=run_name,
        verbose=False
    )
    
    # Path to the best model (using actual save_dir to handle naming conflicts)
    best_model_path = os.path.normpath(os.path.join(results.save_dir, "weights", "best.pt"))
    print(f"Model trained and saved at: {best_model_path}")

    # Log artifacts to MLflow
    if os.path.exists(best_model_path):
        mlflow.log_artifact(best_model_path, artifact_path="weights")
        
    return best_model_path
