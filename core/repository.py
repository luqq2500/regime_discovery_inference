import torch
from sklearn.preprocessing import StandardScaler

class AssetRepository:
    def __init__(self, path='assets', config_file='config.pt', model_file='model.pt'):
        config = torch.load(f'{path}/{config_file}', map_location=torch.device('cpu'), weights_only=False)
        model = torch.jit.load(f'{path}/{model_file}')
        scaler_type = config['scaler_type']
        scaler_weights = config['scaler_weights']
        scaler = None
        if scaler_type == 'StandardScaler':
            scaler = StandardScaler()
            scaler.mean_ = scaler_weights['mean']
            scaler.scale_ = scaler_weights['scale']
            scaler.var_ = scaler_weights['var']
        self.model = model
        self.degree_of_freedom = config['nu']
        self.feature_columns = config['column_names']
        self.scaler = scaler
        self._validate()

    def _validate(self):
        if self.model is None:
            raise ValueError('No model specified')
        if self.degree_of_freedom is None:
            raise ValueError('No degree of freedom specified')
        if self.scaler is None:
            raise ValueError('No scaler specified')
        if self.feature_columns is None:
            raise ValueError('No feature columns specified')


    def get_model(self):
        return self.model
    def get_degree_of_freedom(self):
        return self.degree_of_freedom
    def get_scaler(self):
        return self.scaler
    def get_feature_columns(self):
        return self.feature_columns
    def get_feature_assets(self):
        return [column.split("-")[0] for column in self.feature_columns]
    def get_feature_horizons(self):
        return [column.split("-", 1)[1] for column in self.feature_columns]