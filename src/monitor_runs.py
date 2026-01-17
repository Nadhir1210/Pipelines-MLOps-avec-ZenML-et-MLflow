# src/monitor_runs.py
import mlflow
import pandas as pd
import argparse
import sys
import os

def monitor_metrics(experiment_name="cv_yolo_tiny", threshold=0.1):
    """
    Simplified monitoring: compares the latest run's mAP50 
    against the average of previous runs.
    """
    client = mlflow.tracking.MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"Experiment '{experiment_name}' not found.")
            return
            
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=11
        )
    except Exception as e:
        print(f"Error connecting to MLflow: {e}")
        return

    if len(runs) < 2:
        print("Not enough runs to perform monitoring comparison.")
        return

    latest_run = runs[0]
    previous_runs = runs[1:]
    
    latest_map = latest_run.data.metrics.get("mAP50", 0.0)
    
    # Calculate average mAP50 from history
    hist_maps = [r.data.metrics.get("mAP50", 0.0) for r in previous_runs if "mAP50" in r.data.metrics]
    
    if not hist_maps:
        print("No historical mAP50 found for comparison.")
        return
        
    avg_map = sum(hist_maps) / len(hist_maps)
    diff = (latest_map - avg_map) / (avg_map + 1e-6)

    print(f"--- Monitoring Report for '{experiment_name}' ---")
    print(f"Latest mAP50: {latest_map:.4f}")
    print(f"Baseline (Avg): {avg_map:.4f}")
    print(f"Drift/Diff: {diff*100:.2f}%")

    # Tracking Monitoring Results as a new run in a 'monitoring' experiment
    mlflow.set_experiment("system_monitoring")
    with mlflow.start_run(run_name=f"monitor_{latest_run.info.run_id[:8]}"):
        mlflow.log_metric("drift_percentage", diff)
        mlflow.log_metric("latest_performance", latest_map)
        mlflow.set_tag("source_run_id", latest_run.info.run_id)
        
        if diff < -threshold:
            print("WARNING: Significant performance drop detected!")
            mlflow.set_tag("status", "ALERT")
        else:
            print("Status: Performance is stable.")
            mlflow.set_tag("status", "OK")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="cv_yolo_tiny")
    parser.add_argument("--threshold", type=float, default=0.2, help="Drift threshold (0.2 = 20%)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # In CI/CD dry-run mode, we just check imports
    if args.dry_run:
        print("Monitoring script check: OK")
        sys.exit(0)

    # Set MLflow tracking URI if not set
    if "MLFLOW_TRACKING_URI" not in os.environ:
        os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

    monitor_metrics(args.experiment, args.threshold)
