from src.zenml_pipelines.yolo_pipeline import yolo_training_pipeline

def main():
    """Run the baseline pipeline."""
    yolo_training_pipeline.with_options(
        run_name="yolo_baseline_{date}_{time}"
    )(
        epochs=3,
        imgsz=320,
        lr0=0.005
    )

if __name__ == "__main__":
    main()
