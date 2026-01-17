from zenml import pipeline
from ..zenml_steps.data_loader import data_loader
from ..zenml_steps.trainer import trainer
from ..zenml_steps.evaluator import evaluator

@pipeline
def yolo_training_pipeline(
    epochs: int = 3,
    imgsz: int = 320,
    lr0: float = 0.005,
):
    """Pipeline to train and evaluate YOLO."""
    data_path = data_loader()
    model_path = trainer(data_path=data_path, epochs=epochs, imgsz=imgsz, lr0=lr0)
    evaluator(model_path=model_path, data_path=data_path)
