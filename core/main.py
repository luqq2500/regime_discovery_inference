from pprint import pprint

from core.repository import InferenceRepository
from core.service import ModelService, ScalerService, LatentTraversalService
from core.dto import LatentTraversalInput, LatentTraversalStatsFilter, LatentTraversalAnalysisRequest
from core.usecase import LatentTraversalAnalysisUseCase

if __name__ == '__main__':

    model_service = ModelService()
    scaler_service = ScalerService()
    repository = InferenceRepository()
    lt_service = LatentTraversalService(model_service, repository)

    lt_usecase = LatentTraversalAnalysisUseCase(lt_service, scaler_service, repository)

    lt_inputs = [
        LatentTraversalInput(3, (-3.5, 3.5)),
        LatentTraversalInput(6, (-3.5, 3.5)),
    ]
    request = LatentTraversalAnalysisRequest(lt_inputs)
    response = lt_usecase.run(request)
    pprint(response.get_payload())

