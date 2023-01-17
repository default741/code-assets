from src.run_config import dt_conf, fs_conf
from src.data_transform import DataTransform
from src.feature_selection import FeatureSelection


class Manager:

    def __init__(self) -> None:
        pass

    def run(self) -> None:
        transformed_data = DataTransform().compile_pipeline(**dt_conf)
        feature_selection = FeatureSelection().compile_selection(**fs_conf)
