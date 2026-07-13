from dataclasses import asdict

from core.domain import LatentTraversalInput
from core.repository import InferenceRepository
from core.service import ModelService, ScalerService, LatentTraversalService
from core.dto import LatentTraversalAnalysisRequest
from core.usecase import LatentTraversalAnalysisUseCase

if __name__ == '__main__':

    model_service = ModelService()
    scaler_service = ScalerService()
    repository = InferenceRepository()

    lt_service = LatentTraversalService(model_service, repository)
    lt_usecase = LatentTraversalAnalysisUseCase(lt_service, scaler_service, repository)

    request = LatentTraversalAnalysisRequest(
        inputs=[LatentTraversalInput(dimension=1, sigma_range=(-2, 2))]
    )

    response = lt_usecase.run(request)
    print(asdict(response))

