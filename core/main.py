import inspect
from pprint import pprint

from core.repository.repository import InferenceRepository
from core.service.service import ModelService, ScalerService, LatentTraversalService
from core.usecase.dto import LatentTraversalAnalysisRequest
from core.usecase.usecase import LatentTraversalAnalysisUseCase


def wire_lta_usecase():
    model_service = ModelService()
    scaler_service = ScalerService()
    repository = InferenceRepository()
    lt_service = LatentTraversalService(model_service, repository)
    return LatentTraversalAnalysisUseCase(lt_service, scaler_service, repository)

def wire_repository()->InferenceRepository:
    return InferenceRepository()


if __name__ == '__main__':

    model_service = ModelService()
    scaler_service = ScalerService()
    repository = InferenceRepository()
    lt_service = LatentTraversalService(model_service, repository)
    
    lt_usecase = LatentTraversalAnalysisUseCase(lt_service, scaler_service, repository)

    request = LatentTraversalAnalysisRequest(
        dimensions=[1, 2],
        sigma_range=(-3, 3),
        top_n=10
    )

    
    response = lt_usecase.analyse(request)
    for output in response.outputs:
        pprint(output)

    methods = inspect.getmembers(repository, predicate=inspect.ismethod)

    for name, method in methods:
        # Skip the initialization method
        if name == "__init__":
            continue

        print(f"\n=== Executing: {name}() ===")
        try:
            # Handle methods requiring specific arguments
            if name == "get_dof_by_dimension":
                # Example: pass index 0 to fetch the first degree of freedom
                output = method(dimension=0)
            else:
                # Execute zero-argument getter methods dynamically
                output = method()

            print(f"Type: {type(output)}")
            print(f"Output:\n{output}")

        except Exception as e:
            print(f"Error executing {name}: {e}")
