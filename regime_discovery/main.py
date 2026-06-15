from regime_discovery.config import prepare_latent_traversal_config
from regime_discovery.usecase.latent_traversal.dto import LatentTraversalRequest, LatentTraversalStatsFilter
from regime_discovery.usecase.latent_traversal.usecase import LatentTraversalUseCase

if __name__ == '__main__':
    config = prepare_latent_traversal_config('asset')
    lt = LatentTraversalUseCase(config)
    request = LatentTraversalRequest(
        sigma=2,
        filter=LatentTraversalStatsFilter(
            filter_columns=['mean', 'std'],
            std_threshold=1.0,
            snr_threshold=0.1,
            sort_column=('snr', 5),
        )
    )
    response = lt.execute(request)

    for result in response.response:
        print(result.dimension)
        print(result.stats_filtered_rescaled)
        print(f'\n')
