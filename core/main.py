from core.repository import InferenceRepository
from core.usecase.dto import LTAConfig, LatentTraversalStatsFilter, LTARequest
from core.usecase.lta import LatentTraversalUseCase
from rich import print as rich_print

if __name__ == '__main__':

    #repository = AssetRepository()

    repository = InferenceRepository()
    print(repository.get_embeddings())


    '''
    lt = LatentTraversalUseCase(repository)


    config = LTAConfig(
        dimension=1,
        sigma_range=(-2, 2),
        filter=LatentTraversalStatsFilter(
            snr_threshold=0.1,
            top_n=5
        )
    )

    request = LTARequest([config])

    response = lt.execute(request)
    for result in response.outputs:
        print(f'Dimension: {result.dimension}\n')
        rich_print(f'Payload: {result.get_payload()}\n')
    '''
