from abc import ABC, abstractmethod

from core.usecase.dto import LatentTraversalAnalysisRequest, LatentTraversalAnalysisResponse


class LatentTraversalAnalyserUseCase(ABC):
    @abstractmethod
    def analyse(self, request: LatentTraversalAnalysisRequest)->LatentTraversalAnalysisResponse:
        pass