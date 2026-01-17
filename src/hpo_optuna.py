# src/hpo_optuna.py
import os
import optuna
import mlflow
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from src.utils import set_global_seed

# MLflow setup
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
EXPERIMENT_NAME = "yolo_hpo_optuna"
mlflow.set_experiment(EXPERIMENT_NAME)

def log_yolo_metrics(run_dir: Path):
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    last = df.iloc[-1].to_dict()
    metrics = {}
    candidates = {
        "precision": ["metrics/precision(B)", "metrics/precision"],
        "recall":    ["metrics/recall(B)", "metrics/recall"],
        "mAP50":     ["metrics/mAP50(B)", "metrics/mAP50"],
    }
    for m, cols in candidates.items():
        for c in cols:
            if c in last:
                val = float(last[c])
                metrics[m] = val
                mlflow.log_metric(m, val)
                break
    return metrics

def objective(trial):
    # Hyperparameters to tune
    imgsz = trial.suggest_categorical("imgsz", [320, 416, 640])
    lr0 = trial.suggest_float("lr0", 1e-4, 1e-1, log=True)
    batch = trial.suggest_categorical("batch", [8, 16])
    
    run_name = f"trial_{trial.number}_sz{imgsz}_lr{lr0:.4f}"
    
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params({
            "imgsz": imgsz,
            "lr0": lr0,
            "batch": batch,
            "epochs": 3,  # Fixed for fast HPO
            "model": "yolov8n.pt"
        })
        
        model = YOLO("yolov8n.pt")
        results = model.train(
            data="data/tiny_coco.yaml",
            epochs=3,
            imgsz=imgsz,
            lr0=lr0,
            batch=batch,
            seed=42,
            project="runs_hpo",
            name=run_name,
            verbose=False
        )
        
        # Extract metrics
        metrics = log_yolo_metrics(Path(results.save_dir))
        mAP50 = metrics.get("mAP50", 0.0)
        
        # Log weight
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            mlflow.log_artifact(str(best_pt), artifact_path="weights")
            
        return mAP50

if __name__ == "__main__":
    set_global_seed(42)
    
    # Bayesian search (TPE)
    study = optuna.create_study(direction="maximize", study_name="yolo_bayesian_hpo")
    
    print("Starting Optuna HPO trials (10 trials)...")
    study.optimize(objective, n_trials=10)
    
    print("\nBest Trial:")
    print(f"  Value (mAP50): {study.best_value}")
    print(f"  Params: {study.best_params}")
    
    # Log best params to the master run
    with mlflow.start_run(run_name="optuna_best_summary"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_mAP50", study.best_value)
