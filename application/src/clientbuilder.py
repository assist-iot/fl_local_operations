from abc import ABC, abstractmethod


class FlowerClientTrainingBuilder(ABC):

    @property
    @abstractmethod
    def prepare_training(self) -> None:
        pass


class FlowerClientInferenceBuilder(ABC):

    @property
    @abstractmethod
    def prepare_inference(self) -> None:
        pass
