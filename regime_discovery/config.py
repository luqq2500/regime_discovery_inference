import torch
from sklearn.preprocessing import StandardScaler

from regime_discovery.usecase.latent_traversal.dto import LatentTraversalConfig


def prepare_latent_traversal_config(config_path) -> LatentTraversalConfig:
  config = torch.load(f'{config_path}/config.pt', map_location=torch.device('cpu'), weights_only=False)
  model = torch.jit.load(f'{config_path}/model.pt')
  scaler = load_scaler(config['scaler_type'], config['scaler_weights'])
  col_names = config['column_names']
  return LatentTraversalConfig(model=model, scaler=scaler, col_names=col_names)


def load_scaler(scaler_type, weights):
  scaler = None
  if scaler_type == 'StandardScaler':
    scaler = StandardScaler()
    scaler.mean_ = weights['mean']
    scaler.scale_ = weights['scale']
    scaler.var_ = weights['var']
    scaler.n_features_in_ = weights['n_features_in_']
  return scaler

