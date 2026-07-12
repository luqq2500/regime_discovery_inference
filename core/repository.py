import os
from typing import Any

import torch
from sklearn.preprocessing import StandardScaler

class InferenceRepository:
    def __init__(self, path='assets', asset_file='asset.pt', encoder_file='encoder.pt', decoder_file='decoder.pt'):
        try:
            asset = torch.load(os.path.join(path, asset_file), map_location=torch.device('cpu'), weights_only=False)
            self.encoder = torch.jit.load(os.path.join(path, encoder_file))
            self.decoder = torch.jit.load(os.path.join(path, decoder_file))
            self.embeddings = asset['embeddings']
            self.model_params = asset['model_params']
            self.degree_of_freedom = asset['degree_of_freedoms']
            scaler_type = asset['scaler_type']
            scaler_weights = asset['scaler_weights']
            if scaler_type == 'StandardScaler':
                scaler = StandardScaler()
                scaler.mean_ = scaler_weights['mean']
                scaler.scale_ = scaler_weights['scale']
                scaler.var_ = scaler_weights['var']
            else:
                scaler = None
            self.scaler = scaler
            self.feature = asset['data']
            self.feature_columns = self.feature.columns

        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as e:
            raise RuntimeError(f'Inference repository initialization encountered error: {e}')

    def get_encoder(self, is_eval:bool=True) -> torch.nn.Module:
        return self.encoder.eval() if is_eval else self.encoder
    def get_decoder(self, is_eval:bool=True) -> torch.nn.Module:
        return self.decoder.eval() if is_eval else self.decoder
    def get_embeddings(self):
        return self.embeddings
    def get_model_params(self)->dict[str, Any]:
        return self.model_params
    def get_dimensions(self)->int:
        return self.model_params['latent_dim']
    def get_degree_of_freedom(self, dimension: int)->float:
        return self.degree_of_freedom[dimension]
    def get_scaler(self):
        return self.scaler
    def get_feature(self):
        return self.feature
    def get_feature_columns(self):
        return self.feature_columns
    def get_feature_assets(self):
        assets = [column.split("-")[0] for column in self.feature_columns]
        return list(set(assets))
    def get_feature_horizons(self):
        horizons = [column.split("-", 1)[1] for column in self.feature_columns]
        return list(set(horizons))