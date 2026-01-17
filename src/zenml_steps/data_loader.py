from zenml import step
import os

@step
def data_loader() -> str:
    """Loader for the tiny_coco dataset."""
    data_path = "data/tiny_coco.yaml"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset config {data_path} not found.")
    return data_path
