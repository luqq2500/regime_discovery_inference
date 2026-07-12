import os
from typing import Any

import pandas as pd
import torch

class InferenceRepository:
    def __init__(self, path='assets', asset_file='asset.pt'):
        try:
            asset = torch.load(os.path.join(path, asset_file), map_location=torch.device('cpu'), weights_only=False)
            self.embeddings = asset['embeddings']
            self.model_params = asset['model_params']
            self.degree_of_freedoms = asset['degree_of_freedoms']
            self.feature = asset['data']
            self.feature_columns = self.feature.columns

        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as e:
            raise RuntimeError(f'Inference repository initialization encountered error: {e}')

    def get_embeddings(self):
        return self.embeddings
    def get_model_params(self)->dict[str, Any]:
        return self.model_params
    def get_dimensions(self)->int:
        return self.model_params['latent_dim']
    def get_dofs(self)->list[float]:
        return self.degree_of_freedoms
    def get_dof_by_dimension(self, dimension: int)->float:
        return self.degree_of_freedoms[dimension]
    def get_feature(self)->pd.DataFrame:
        return self.feature
    def get_feature_columns(self)->list[str]:
        return self.feature_columns
    def get_feature_assets(self)->list[str]:
        assets = [column.split("-")[0] for column in self.feature_columns]
        return list(set(assets))
    def get_feature_horizons(self)->list[str]:
        horizons = [column.split("-", 1)[1] for column in self.feature_columns]
        return list(set(horizons))