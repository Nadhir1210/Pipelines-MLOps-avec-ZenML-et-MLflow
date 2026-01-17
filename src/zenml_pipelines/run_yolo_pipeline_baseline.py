from src.zenml_pipelines.yolo_pipeline import yolo_training_pipeline

def main():
    """Run the baseline pipeline."""
    yolo_training_pipeline(
        epochs=3,
        imgsz=320,
        lr0=0.005
    )

if __name__ == "__main__":
    main()
