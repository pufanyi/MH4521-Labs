from abc import ABC, abstractmethod

class Arm(ABC):
    @abstractmethod
    def mean(self) -> float:
        pass

    @abstractmethod
    def pull(self) -> float:
        pass
