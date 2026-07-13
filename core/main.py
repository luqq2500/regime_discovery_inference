from dataclasses import asdict
from pprint import pprint

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
        dimensions=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        sigma_range=(-3, 3),
        top_n=3
    )

    response = lt_usecase.run(request)
    for output in response.outputs:
        print(f'Regime: {output.dimension}')
        print(f'Degree of Freedom: {output.degree_of_freedom}')
        print(f'Sigma Range: {output.sweeps}')
        pprint(output.feature_metrics)