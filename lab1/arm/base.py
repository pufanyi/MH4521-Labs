from abc import ABC, abstractmethod

class Arm(ABC):
    @abstractmethod
    def mean(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def pull(self) -> float:
        raise NotImplementedError
