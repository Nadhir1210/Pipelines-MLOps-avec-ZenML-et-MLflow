from src.zenml_pipelines.yolo_pipeline import yolo_training_pipeline

def main():
    """Run a small grid of pipeline configurations."""
    configs = [
        {"imgsz": 320, "lr0": 0.005},
        {"imgsz": 416, "lr0": 0.005},
        {"imgsz": 320, "lr0": 0.01},
        {"imgsz": 416, "lr0": 0.01},
    ]

    for cfg in configs:
        print(f"--- Running ZenML Pipeline with imgsz={cfg['imgsz']}, lr0={cfg['lr0']} ---")
        yolo_training_pipeline.with_options(
            run_name=f"yolo_grid_sz{cfg['imgsz']}_lr{cfg['lr0']}"
        )(
            epochs=3,
            imgsz=cfg['imgsz'],
            lr0=cfg['lr0']
        )

if __name__ == "__main__":
    main()
