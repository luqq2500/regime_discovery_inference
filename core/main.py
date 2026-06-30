from core.repository import AssetRepository
from core.usecase.dto import LatentTraversalRequest, LatentTraversalStatsFilter
from core.usecase.core import LatentTraversalUseCase
from rich import print as rich_print

if __name__ == '__main__':

    asset_repository = AssetRepository()
    lt = LatentTraversalUseCase(asset_repository)
    request = LatentTraversalRequest(
        sigma_range=(-10, -2),
        filter=LatentTraversalStatsFilter(
            std_threshold=None,
            snr_threshold=1.0,
            top_n=5
        )
    )

    response = lt.execute(request)
    for result in response.outputs:
        print(f'Dimension: {result.dimension}\n')
        rich_print(f'Payload: {result.get_payload()}\n')
